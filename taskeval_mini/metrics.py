from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


EXPECTED_FIELDS = ["intent", "priority", "requires_human", "target_system", "sla_hours"]


@dataclass
class ExampleMetrics:
    task_id: str
    field_accuracy: float
    exact_match: int
    schema_compliant: int
    num_fields: int = field(default=len(EXPECTED_FIELDS))


@dataclass
class AggregateMetrics:
    num_examples: int
    avg_field_accuracy: float
    exact_match_rate: float
    schema_compliance_rate: float


def compare_structured_outputs(
    expected: Dict[str, Any],
    predicted: Dict[str, Any],
) -> Tuple[float, int, int]:
    """Compare two structured outputs and compute metrics.

    Returns:
        field_accuracy: fraction of fields that match exactly.
        exact_match: 1 if all fields match exactly, else 0.
        schema_compliant: 1 if all expected fields are present and have non-null values, else 0.
    """
    correct = 0
    total = len(EXPECTED_FIELDS)
    schema_ok = 1

    for field in EXPECTED_FIELDS:
        if field not in predicted:
            schema_ok = 0
            continue
        if predicted[field] is None:
            schema_ok = 0
        if predicted[field] == expected.get(field):
            correct += 1

    field_accuracy = correct / total if total > 0 else 0.0
    exact_match = 1 if correct == total else 0

    return field_accuracy, exact_match, schema_ok


def aggregate_metrics(per_example: List[ExampleMetrics]) -> AggregateMetrics:
    n = len(per_example)
    if n == 0:
        return AggregateMetrics(
            num_examples=0,
            avg_field_accuracy=0.0,
            exact_match_rate=0.0,
            schema_compliance_rate=0.0,
        )

    avg_field_acc = sum(e.field_accuracy for e in per_example) / n
    exact_rate = sum(e.exact_match for e in per_example) / n
    schema_rate = sum(e.schema_compliant for e in per_example) / n

    return AggregateMetrics(
        num_examples=n,
        avg_field_accuracy=avg_field_acc,
        exact_match_rate=exact_rate,
        schema_compliance_rate=schema_rate,
    )
