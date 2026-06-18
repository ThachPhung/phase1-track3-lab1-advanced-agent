from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Iterable
from .schemas import QAExample, RunRecord

def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def load_dataset(path: str | Path) -> list[QAExample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [QAExample.model_validate(_coerce_qa_example(item)) for item in raw]

def _coerce_qa_example(item: dict[str, Any]) -> dict[str, Any]:
    if {"qid", "difficulty", "question", "gold_answer", "context"}.issubset(item):
        return item
    if {"_id", "question", "answer", "context"}.issubset(item):
        return {
            "qid": item["_id"],
            "difficulty": _coerce_difficulty(item.get("level")),
            "question": item["question"],
            "gold_answer": item["answer"],
            "context": [_coerce_context_chunk(chunk) for chunk in item["context"]],
        }
    return item

def _coerce_difficulty(level: Any) -> str:
    if level in {"easy", "medium", "hard"}:
        return level
    return "medium"

def _coerce_context_chunk(chunk: Any) -> dict[str, str]:
    if isinstance(chunk, dict):
        return chunk
    if isinstance(chunk, list) and len(chunk) == 2:
        title, sentences = chunk
        text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
        return {"title": str(title), "text": text}
    return {"title": "Untitled", "text": str(chunk)}

def save_jsonl(path: str | Path, records: Iterable[RunRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
