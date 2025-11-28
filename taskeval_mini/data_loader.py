from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class TaskExample:
    task_id: str
    input_text: str
    context: Optional[str]
    expected_output: Dict[str, Any]


def load_tasks(path: Path) -> List[TaskExample]:
    """Load task examples from a JSONL file."""
    tasks: List[TaskExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append(
                TaskExample(
                    task_id=obj["task_id"],
                    input_text=obj["input_text"],
                    context=obj.get("context"),
                    expected_output=obj["expected_output"],
                )
            )
    return tasks


def package_data_path(filename: str) -> Path:
    """Resolve a data file inside the package data directory."""
    return Path(__file__).resolve().parent / "data" / filename
