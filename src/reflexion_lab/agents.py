from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, pop_call_metrics, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            actor_metrics = pop_call_metrics()
            judge = evaluator(example, answer)
            judge_metrics = pop_call_metrics()
            token_estimate = actor_metrics["token_count"] + judge_metrics["token_count"] + judge.token_count
            latency_ms = actor_metrics["latency_ms"] + judge_metrics["latency_ms"] + judge.latency_ms
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break
            
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, judge)
                reflection_metrics = pop_call_metrics()
                reflection_memory.append(f"Lesson: {reflection.lesson} Next strategy: {reflection.next_strategy}")
                reflections.append(reflection)
                trace.reflection = reflection
                trace.token_estimate += reflection_metrics["token_count"] + reflection.token_count
                trace.latency_ms += reflection_metrics["latency_ms"] + reflection.latency_ms
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        base_qid = example.qid.split("_", 1)[0]
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, FAILURE_MODE_BY_QID.get(base_qid, "wrong_final_answer"))
        return RunRecord(qid=example.qid, difficulty=example.difficulty, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
