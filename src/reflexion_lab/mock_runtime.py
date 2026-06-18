from __future__ import annotations
import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}
_LAST_CALL_METRICS = {"token_count": 0, "latency_ms": 0}

def pop_call_metrics() -> dict[str, int]:
    metrics = dict(_LAST_CALL_METRICS)
    _record_metrics(0, 0)
    return metrics

def _record_metrics(token_count: int, latency_ms: int) -> None:
    _LAST_CALL_METRICS["token_count"] = max(0, int(token_count))
    _LAST_CALL_METRICS["latency_ms"] = max(0, int(latency_ms))

def _runtime_mode() -> str:
    return os.getenv("REFLEXION_RUNTIME", "mock").strip().lower()

def _base_qid(qid: str) -> str:
    return qid.split("_", 1)[0]

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if _runtime_mode() in {"llm", "ollama", "openai"}:
        return _llm_actor_answer(example, attempt_id, agent_type, reflection_memory)
    start = time.perf_counter()
    base_qid = _base_qid(example.qid)
    if base_qid not in FIRST_ATTEMPT_WRONG:
        answer = example.gold_answer
    elif agent_type == "react":
        answer = FIRST_ATTEMPT_WRONG[base_qid]
    elif attempt_id == 1 and not reflection_memory:
        answer = FIRST_ATTEMPT_WRONG[base_qid]
    else:
        answer = example.gold_answer
    _record_metrics(_estimate_tokens(example.question, answer, *reflection_memory), _elapsed_ms(start))
    return answer

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if _runtime_mode() in {"llm", "ollama", "openai"}:
        return _llm_evaluator(example, answer)
    start = time.perf_counter()
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        result = JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
        _record_metrics(_estimate_tokens(example.gold_answer, answer, result.reason), _elapsed_ms(start))
        return result
    if normalize_answer(answer) == "london":
        result = JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
        _record_metrics(_estimate_tokens(example.gold_answer, answer, result.reason, *result.missing_evidence), _elapsed_ms(start))
        return result
    result = JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])
    _record_metrics(_estimate_tokens(example.gold_answer, answer, result.reason, *result.missing_evidence, *result.spurious_claims), _elapsed_ms(start))
    return result

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if _runtime_mode() in {"llm", "ollama", "openai"}:
        return _llm_reflector(example, attempt_id, judge)
    start = time.perf_counter()
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if _base_qid(example.qid) == "hp2" else "Verify the final entity against the second paragraph before answering."
    result = ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)
    _record_metrics(_estimate_tokens(judge.reason, result.lesson, result.next_strategy), _elapsed_ms(start))
    return result

def _llm_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    user_prompt = {
        "question": example.question,
        "context": [chunk.model_dump() for chunk in example.context],
        "agent_type": agent_type,
        "attempt_id": attempt_id,
        "reflection_memory": reflection_memory,
    }
    text, metrics = _call_llm(ACTOR_SYSTEM, json.dumps(user_prompt, ensure_ascii=False))
    _record_metrics(metrics["token_count"], metrics["latency_ms"])
    return text.strip().strip('"')

def _llm_evaluator(example: QAExample, answer: str) -> JudgeResult:
    user_prompt = {
        "question": example.question,
        "gold_answer": example.gold_answer,
        "predicted_answer": answer,
        "context": [chunk.model_dump() for chunk in example.context],
    }
    text, metrics = _call_llm(EVALUATOR_SYSTEM, json.dumps(user_prompt, ensure_ascii=False))
    payload = _extract_json(text)
    result = JudgeResult.model_validate(payload)
    _record_metrics(metrics["token_count"], metrics["latency_ms"])
    return result

def _llm_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    user_prompt = {
        "question": example.question,
        "context": [chunk.model_dump() for chunk in example.context],
        "attempt_id": attempt_id,
        "evaluator_feedback": judge.model_dump(),
    }
    text, metrics = _call_llm(REFLECTOR_SYSTEM, json.dumps(user_prompt, ensure_ascii=False))
    payload = _extract_json(text)
    result = ReflectionEntry.model_validate({"attempt_id": attempt_id, **payload})
    _record_metrics(metrics["token_count"], metrics["latency_ms"])
    return result

def _call_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict[str, int]]:
    mode = _runtime_mode()
    if mode == "openai" or (mode == "llm" and os.getenv("OPENAI_API_KEY")):
        return _call_openai_compatible(system_prompt, user_prompt)
    return _call_ollama(system_prompt, user_prompt)

def _call_openai_compatible(system_prompt: str, user_prompt: str) -> tuple[str, dict[str, int]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when REFLEXION_RUNTIME=openai.")
    base_url = os.getenv("REFLEXION_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": 0,
    }
    start = time.perf_counter()
    response = _post_json(f"{base_url}/chat/completions", payload, {"Authorization": f"Bearer {api_key}"})
    latency_ms = _elapsed_ms(start)
    text = response["choices"][0]["message"]["content"]
    token_count = int(response.get("usage", {}).get("total_tokens", _estimate_tokens(system_prompt, user_prompt, text)))
    return text, {"token_count": token_count, "latency_ms": latency_ms}

def _call_ollama(system_prompt: str, user_prompt: str) -> tuple[str, dict[str, int]]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    start = time.perf_counter()
    response = _post_json(f"{base_url}/api/chat", payload, {})
    latency_ms = _elapsed_ms(start)
    text = response["message"]["content"]
    token_count = int(response.get("prompt_eval_count", 0)) + int(response.get("eval_count", 0))
    if token_count == 0:
        token_count = _estimate_tokens(system_prompt, user_prompt, text)
    return text, {"token_count": token_count, "latency_ms": latency_ms}

def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST")
    timeout = float(os.getenv("REFLEXION_LLM_TIMEOUT", "30"))
    retries = max(1, int(os.getenv("REFLEXION_LLM_RETRIES", "2")))
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(attempt)
                continue
    raise RuntimeError(
        f"LLM request failed for {url} after {retries} attempt(s): {last_error}. "
        "If this is an SSL handshake timeout, check VPN/proxy/firewall access to api.openai.com "
        "or set REFLEXION_OPENAI_BASE_URL to an OpenAI-compatible endpoint you can reach."
    ) from last_error

def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"LLM did not return JSON: {text}") from None
        return json.loads(text[start : end + 1])

def _estimate_tokens(*parts: str) -> int:
    return max(1, sum(max(1, len(part.split())) for part in parts))

def _elapsed_ms(start: float) -> int:
    return max(1, round((time.perf_counter() - start) * 1000))
