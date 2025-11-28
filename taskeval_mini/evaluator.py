from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .config import AVAILABLE_MODELS, ModelConfig
from .data_loader import TaskExample, load_tasks, package_data_path
from .metrics import (
    ExampleMetrics,
    aggregate_metrics,
    compare_structured_outputs,
)
from .model_runner import create_model_runner


def evaluate_model(
    model_name: str,
    output_dir: Path,
    tasks_path: Optional[Path] = None,
) -> None:
    """Run evaluation for a given model and write results to CSV and JSON.

    Args:
        model_name: key from AVAILABLE_MODELS.
        output_dir: directory where results are written.
        tasks_path: optional custom path to tasks JSONL file; if None, uses packaged data.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS)}")

    config: ModelConfig = AVAILABLE_MODELS[model_name]
    runner = create_model_runner(config)

    if tasks_path is None:
        tasks_path = package_data_path("tasks_support.jsonl")

    tasks: List[TaskExample] = load_tasks(tasks_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_example: List[ExampleMetrics] = []
    raw_preds_path = output_dir / f"{model_name}_predictions.jsonl"
    metrics_csv_path = output_dir / f"{model_name}_metrics.csv"
    summary_json_path = output_dir / f"{model_name}_summary.json"

    with raw_preds_path.open("w", encoding="utf-8") as pred_f, metrics_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as csv_f:
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(
            [
                "task_id",
                "field_accuracy",
                "exact_match",
                "schema_compliant",
                "num_fields",
            ]
        )

        for ex in tasks:
            predicted = runner.generate(ex.input_text, ex.context)
            # Write raw prediction for inspection
            pred_f.write(
                json.dumps(
                    {
                        "task_id": ex.task_id,
                        "input_text": ex.input_text,
                        "context": ex.context,
                        "expected_output": ex.expected_output,
                        "predicted_output": predicted,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            field_acc, exact_match, schema_ok = compare_structured_outputs(
                ex.expected_output, predicted
            )
            ex_metrics = ExampleMetrics(
                task_id=ex.task_id,
                field_accuracy=field_acc,
                exact_match=exact_match,
                schema_compliant=schema_ok,
            )
            per_example.append(ex_metrics)
            csv_writer.writerow(
                [
                    ex_metrics.task_id,
                    f"{ex_metrics.field_accuracy:.4f}",
                    ex_metrics.exact_match,
                    ex_metrics.schema_compliant,
                    ex_metrics.num_fields,
                ]
            )

    summary = aggregate_metrics(per_example)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    print(f"Evaluated model '{model_name}' on {summary.num_examples} examples.")
    print(f"Average field accuracy: {summary.avg_field_accuracy:.3f}")
    print(f"Exact match rate: {summary.exact_match_rate:.3f}")
    print(f"Schema compliance rate: {summary.schema_compliance_rate:.3f}")
