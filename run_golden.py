from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich import print

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)
load_dotenv()


@app.command()
def main(
    dataset: str = "data/golden_test.json",
    out_dir: str = "outputs/golden_test",
    reflexion_attempts: int = 3,
    limit: int = 0,
) -> None:
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        raise typer.BadParameter(f"Missing Golden Test Set file: {dataset_path}")

    examples = load_dataset(dataset_path)
    if limit > 0:
        examples = examples[:limit]
        print(f"[yellow]Using first {len(examples)} examples for smoke test[/yellow]")
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)

    react_records = [react.run(example) for example in examples]
    reflexion_records = [reflexion.run(example) for example in examples]
    all_records = react_records + reflexion_records

    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=dataset_path.name, mode=os.getenv("REFLEXION_RUNTIME", "mock"))
    json_path, md_path = save_report(report, out_path)

    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(f"[green]Saved[/green] {out_path / 'react_runs.jsonl'}")
    print(f"[green]Saved[/green] {out_path / 'reflexion_runs.jsonl'}")
    print(json.dumps(report.summary, indent=2))
    print(f"\nRun autograde: python3 autograde.py --report-path {json_path}")


if __name__ == "__main__":
    app()
