from __future__ import annotations
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

FAILURE_MODE_KEYS = [
    "none",
    "entity_drift",
    "incomplete_multi_hop",
    "wrong_final_answer",
    "looping",
    "reflection_overfit",
]

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {"count": len(rows), "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4), "avg_attempts": round(mean(r.attempts for r in rows), 4), "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2), "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2)}
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {"em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4), "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4), "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2), "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)}
    return summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    totals: Counter = Counter()
    by_agent: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        totals[record.failure_mode] += 1
        by_agent[record.agent_type][record.failure_mode] += 1
    totals_by_mode = {failure_mode: totals.get(failure_mode, 0) for failure_mode in FAILURE_MODE_KEYS}
    by_agent_by_mode = {
        agent: {failure_mode: counter.get(failure_mode, 0) for failure_mode in FAILURE_MODE_KEYS}
        for agent, counter in by_agent.items()
    }
    return {**totals_by_mode, "by_agent": by_agent_by_mode}

def estimate_cost(records: list[RunRecord]) -> dict:
    input_price_per_1k_tokens = float(os.getenv("REFLEXION_INPUT_PRICE_PER_1K_TOKENS", "0.00015"))
    output_price_per_1k_tokens = float(os.getenv("REFLEXION_OUTPUT_PRICE_PER_1K_TOKENS", "0.0006"))
    legacy_price_per_1k_tokens = float(os.getenv("REFLEXION_PRICE_PER_1K_TOKENS", "0.0"))
    usd_to_vnd = float(os.getenv("REFLEXION_USD_TO_VND", "25000"))
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)

    cost: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        total_tokens = sum(r.token_estimate for r in rows)
        total_latency_ms = sum(r.latency_ms for r in rows)
        # AttemptTrace currently stores total tokens. Split them for cost display
        # using a conservative QA workload estimate: most tokens are prompt/context.
        estimated_input_tokens = round(total_tokens * 0.7)
        estimated_output_tokens = total_tokens - estimated_input_tokens
        estimated_token_cost_usd = (
            (total_tokens / 1000) * legacy_price_per_1k_tokens
            if legacy_price_per_1k_tokens > 0
            else (estimated_input_tokens / 1000) * input_price_per_1k_tokens
            + (estimated_output_tokens / 1000) * output_price_per_1k_tokens
        )
        cost[agent_type] = {
            "num_runs": len(rows),
            "total_tokens": total_tokens,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "avg_tokens_per_run": round(total_tokens / len(rows), 2) if rows else 0,
            "total_latency_ms": total_latency_ms,
            "total_running_time_seconds": round(total_latency_ms / 1000, 3),
            "avg_latency_ms": round(total_latency_ms / len(rows), 2) if rows else 0,
            "input_price_per_1k_tokens_usd": input_price_per_1k_tokens,
            "output_price_per_1k_tokens_usd": output_price_per_1k_tokens,
            "legacy_price_per_1k_tokens_usd": legacy_price_per_1k_tokens,
            "estimated_token_cost_usd": round(estimated_token_cost_usd, 6),
            "estimated_token_cost_vnd": round(estimated_token_cost_usd * usd_to_vnd),
        }
    if "react" in cost and "reflexion" in cost:
        cost["delta_reflexion_minus_react"] = {
            "total_tokens": cost["reflexion"]["total_tokens"] - cost["react"]["total_tokens"],
            "estimated_input_tokens": cost["reflexion"]["estimated_input_tokens"] - cost["react"]["estimated_input_tokens"],
            "estimated_output_tokens": cost["reflexion"]["estimated_output_tokens"] - cost["react"]["estimated_output_tokens"],
            "total_latency_ms": cost["reflexion"]["total_latency_ms"] - cost["react"]["total_latency_ms"],
            "total_running_time_seconds": round(cost["reflexion"]["total_running_time_seconds"] - cost["react"]["total_running_time_seconds"], 3),
            "estimated_token_cost_usd": round(cost["reflexion"]["estimated_token_cost_usd"] - cost["react"]["estimated_token_cost_usd"], 6),
            "estimated_token_cost_vnd": round(cost["reflexion"]["estimated_token_cost_vnd"] - cost["react"]["estimated_token_cost_vnd"]),
        }
    return cost

def _metrics(rows: list[RunRecord]) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row.is_correct)
    total_tokens = sum(row.token_estimate for row in rows)
    total_latency_ms = sum(row.latency_ms for row in rows)
    return {
        "count": total,
        "correct": correct,
        "incorrect": total - correct,
        "em": round(correct / total, 4) if total else 0,
        "avg_attempts": round(mean(row.attempts for row in rows), 4) if rows else 0,
        "avg_tokens": round(total_tokens / total, 2) if total else 0,
        "avg_latency_ms": round(total_latency_ms / total, 2) if total else 0,
        "total_tokens": total_tokens,
        "total_latency_ms": total_latency_ms,
        "total_running_time_seconds": round(total_latency_ms / 1000, 3),
    }

def _group_metrics(records: list[RunRecord], key: str) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[str(getattr(record, key))].append(record)
    return {group: _metrics(rows) for group, rows in sorted(grouped.items())}

def build_benchmark_details(records: list[RunRecord], max_trace_examples: int = 12) -> dict:
    by_agent = _group_metrics(records, "agent_type")
    by_difficulty: dict[str, dict] = defaultdict(dict)
    for record in records:
        by_difficulty[record.difficulty].setdefault(record.agent_type, []).append(record)
    by_difficulty_metrics = {
        difficulty: {agent_type: _metrics(rows) for agent_type, rows in sorted(agent_rows.items())}
        for difficulty, agent_rows in sorted(by_difficulty.items())
    }

    candidates = sorted(
        records,
        key=lambda record: (
            record.is_correct and not record.reflections,
            -len(record.reflections),
            record.agent_type != "reflexion",
            record.failure_mode,
            record.difficulty,
            record.qid,
        ),
    )
    interesting = []
    seen_signatures = set()
    for record in candidates:
        base_qid = record.qid.split("_", 1)[0]
        signature = (record.agent_type, record.failure_mode, record.difficulty, base_qid)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        interesting.append(record)
        if len(interesting) >= max_trace_examples:
            break
    if len(interesting) < max_trace_examples:
        for record in candidates:
            if record in interesting:
                continue
            interesting.append(record)
            if len(interesting) >= max_trace_examples:
                break
    trace_examples = [
        {
            "qid": record.qid,
            "difficulty": record.difficulty,
            "agent_type": record.agent_type,
            "question": record.question,
            "gold_answer": record.gold_answer,
            "predicted_answer": record.predicted_answer,
            "is_correct": record.is_correct,
            "attempts": record.attempts,
            "failure_mode": record.failure_mode,
            "token_estimate": record.token_estimate,
            "latency_ms": record.latency_ms,
            "traces": [trace.model_dump() for trace in record.traces],
            "reflections": [reflection.model_dump() for reflection in record.reflections],
        }
        for record in interesting
    ]
    return {
        "by_agent": by_agent,
        "by_difficulty": by_difficulty_metrics,
        "trace_examples": trace_examples,
        "run_config": {
            "runtime": os.getenv("REFLEXION_RUNTIME", "mock"),
            "openai_model": os.getenv("OPENAI_MODEL", ""),
            "ollama_model": os.getenv("OLLAMA_MODEL", ""),
            "llm_timeout_seconds": float(os.getenv("REFLEXION_LLM_TIMEOUT", "30")),
            "llm_retries": int(os.getenv("REFLEXION_LLM_RETRIES", "2")),
            "input_price_per_1k_tokens_usd": float(os.getenv("REFLEXION_INPUT_PRICE_PER_1K_TOKENS", "0.00015")),
            "output_price_per_1k_tokens_usd": float(os.getenv("REFLEXION_OUTPUT_PRICE_PER_1K_TOKENS", "0.0006")),
            "usd_to_vnd": float(os.getenv("REFLEXION_USD_TO_VND", "25000")),
        },
    }

def build_report(records: list[RunRecord], dataset_name: str, mode: str = "mock") -> ReportPayload:
    examples = [{"qid": r.qid, "difficulty": r.difficulty, "agent_type": r.agent_type, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer, "is_correct": r.is_correct, "attempts": r.attempts, "failure_mode": r.failure_mode, "reflection_count": len(r.reflections)} for r in records]
    return ReportPayload(meta={"dataset": dataset_name, "mode": mode, "num_records": len(records), "agents": sorted({r.agent_type for r in records})}, summary=summarize(records), cost_estimate=estimate_cost(records), benchmark_details=build_benchmark_details(records), failure_modes=failure_breakdown(records), examples=examples, extensions=["structured_evaluator", "reflection_memory", "benchmark_report_json", "mock_mode_for_autograding"], discussion="Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. ReAct is cheaper and faster because it makes one answer attempt, while Reflexion spends extra calls on evaluator feedback and reflection memory when the first answer fails. In this benchmark, compare EM, attempts, tokens, and running time together: Reflexion is worth the extra cost when it recovers multi-hop mistakes that ReAct leaves unresolved.")

def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)

def _agent_detail_table(details: dict) -> str:
    rows = []
    for agent_type, metrics in details.get("by_agent", {}).items():
        rows.append(
            f"| {agent_type} | {metrics.get('count', 0)} | {metrics.get('correct', 0)} | {metrics.get('incorrect', 0)} | {_fmt(metrics.get('em', 0))} | {_fmt(metrics.get('avg_attempts', 0))} | {_fmt(metrics.get('avg_tokens', 0))} | {_fmt(metrics.get('avg_latency_ms', 0))} | {_fmt(metrics.get('total_running_time_seconds', 0))} |"
        )
    return "\n".join(rows)

def _difficulty_table(details: dict) -> str:
    rows = []
    for difficulty, agents in details.get("by_difficulty", {}).items():
        for agent_type, metrics in agents.items():
            rows.append(
                f"| {difficulty} | {agent_type} | {metrics.get('count', 0)} | {metrics.get('correct', 0)} | {metrics.get('incorrect', 0)} | {_fmt(metrics.get('em', 0))} | {_fmt(metrics.get('avg_attempts', 0))} | {_fmt(metrics.get('avg_tokens', 0))} | {_fmt(metrics.get('avg_latency_ms', 0))} |"
            )
    return "\n".join(rows)

def _trace_examples_md(details: dict) -> str:
    sections = []
    for item in details.get("trace_examples", []):
        trace_lines = []
        for trace in item.get("traces", []):
            reflection = trace.get("reflection") or {}
            reflection_text = ""
            if reflection:
                reflection_text = f" Reflection: {reflection.get('lesson', '')} Next: {reflection.get('next_strategy', '')}"
            trace_lines.append(
                f"- Attempt {trace.get('attempt_id')}: score={trace.get('score')}, answer={trace.get('answer')}, reason={trace.get('reason')}.{reflection_text}"
            )
        sections.append(
            "\n".join(
                [
                    f"### {item.get('qid')} ({item.get('agent_type')}, {item.get('difficulty')})",
                    f"- Question: {item.get('question')}",
                    f"- Gold: {item.get('gold_answer')}",
                    f"- Prediction: {item.get('predicted_answer')}",
                    f"- Correct: {item.get('is_correct')} | Attempts: {item.get('attempts')} | Failure mode: {item.get('failure_mode')}",
                    *trace_lines,
                ]
            )
        )
    return "\n\n".join(sections)

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    cost = report.cost_estimate
    react_cost = cost.get("react", {})
    reflexion_cost = cost.get("reflexion", {})
    cost_delta = cost.get("delta_reflexion_minus_react", {})
    details = report.benchmark_details
    by_agent = details.get("by_agent", {})
    react_details = by_agent.get("react", {})
    reflexion_details = by_agent.get("reflexion", {})
    run_config = details.get("run_config", {})
    agent_detail_rows = _agent_detail_table(details)
    difficulty_rows = _difficulty_table(details)
    trace_examples = _trace_examples_md(details)
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Run Configuration
| Key | Value |
|---|---|
| Runtime | {run_config.get('runtime', report.meta['mode'])} |
| OpenAI model | {run_config.get('openai_model', '') or 'n/a'} |
| Ollama model | {run_config.get('ollama_model', '') or 'n/a'} |
| LLM timeout (s) | {run_config.get('llm_timeout_seconds', 0)} |
| LLM retries | {run_config.get('llm_retries', 0)} |
| Input price / 1K tokens (USD) | {run_config.get('input_price_per_1k_tokens_usd', 0)} |
| Output price / 1K tokens (USD) | {run_config.get('output_price_per_1k_tokens_usd', 0)} |
| USD to VND | {run_config.get('usd_to_vnd', 0)} |

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Accuracy Overview
| Agent | Correct / Total | Incorrect / Total | Accuracy |
|---|---:|---:|---:|
| ReAct | {react_details.get('correct', 0)} / {react_details.get('count', 0)} | {react_details.get('incorrect', 0)} / {react_details.get('count', 0)} | {_fmt(react_details.get('em', 0))} |
| Reflexion | {reflexion_details.get('correct', 0)} / {reflexion_details.get('count', 0)} | {reflexion_details.get('incorrect', 0)} / {reflexion_details.get('count', 0)} | {_fmt(reflexion_details.get('em', 0))} |

## Detailed Agent Metrics
| Agent | Count | Correct | Incorrect | EM | Avg attempts | Avg tokens | Avg latency (ms) | Total running time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{agent_detail_rows}

## ReAct vs Reflexion Comparison
| Aspect | ReAct Agent | Reflexion Agent |
|---|---|---|
| Core loop | Single answer attempt, then evaluator checks the result. | Multiple attempts; failed attempts create reflection memory for the next answer. |
| Strength | Lower token usage and lower running time. | Better recovery from partial multi-hop answers, entity drift, and unsupported final answers. |
| Weakness | Cannot self-correct after a wrong first answer. | Costs more tokens and latency when reflection is needed. |
| Best use case | Simple questions or tight latency/cost budget. | Harder multi-hop questions where a second attempt can fix reasoning mistakes. |

## Cost Estimate Including Running Time
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| Runs | {react_cost.get('num_runs', 0)} | {reflexion_cost.get('num_runs', 0)} | 0 |
| Total tokens | {react_cost.get('total_tokens', 0)} | {reflexion_cost.get('total_tokens', 0)} | {cost_delta.get('total_tokens', 0)} |
| Estimated input tokens | {react_cost.get('estimated_input_tokens', 0)} | {reflexion_cost.get('estimated_input_tokens', 0)} | {cost_delta.get('estimated_input_tokens', 0)} |
| Estimated output tokens | {react_cost.get('estimated_output_tokens', 0)} | {reflexion_cost.get('estimated_output_tokens', 0)} | {cost_delta.get('estimated_output_tokens', 0)} |
| Avg tokens / run | {react_cost.get('avg_tokens_per_run', 0)} | {reflexion_cost.get('avg_tokens_per_run', 0)} | {delta.get('tokens_abs', 0)} |
| Total running time (s) | {react_cost.get('total_running_time_seconds', 0)} | {reflexion_cost.get('total_running_time_seconds', 0)} | {cost_delta.get('total_running_time_seconds', 0)} |
| Avg latency (ms) | {react_cost.get('avg_latency_ms', 0)} | {reflexion_cost.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |
| Input price / 1K tokens (USD) | {react_cost.get('input_price_per_1k_tokens_usd', 0)} | {reflexion_cost.get('input_price_per_1k_tokens_usd', 0)} | 0 |
| Output price / 1K tokens (USD) | {react_cost.get('output_price_per_1k_tokens_usd', 0)} | {reflexion_cost.get('output_price_per_1k_tokens_usd', 0)} | 0 |
| Estimated token cost (USD) | {react_cost.get('estimated_token_cost_usd', 0)} | {reflexion_cost.get('estimated_token_cost_usd', 0)} | {cost_delta.get('estimated_token_cost_usd', 0)} |
| Estimated token cost (VND) | {react_cost.get('estimated_token_cost_vnd', 0)} | {reflexion_cost.get('estimated_token_cost_vnd', 0)} | {cost_delta.get('estimated_token_cost_vnd', 0)} |

## Breakdown By Difficulty
| Difficulty | Agent | Count | Correct | Incorrect | EM | Avg attempts | Avg tokens | Avg latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{difficulty_rows}

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Detailed Trace Examples
{trace_examples}

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
