from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the project root (the folder that contains the taskeval_mini package)
# is on sys.path so that "import taskeval_mini" works when running this script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taskeval_mini.evaluator import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TaskEval Mini evaluation.")
    parser.add_argument(
        "--model",
        type=str,
        default="rule_based",
        help="Model name to evaluate (default: rule_based).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to write results into.",
    )
    parser.add_argument(
        "--tasks-path",
        type=str,
        default="",
        help="Optional path to a custom tasks JSONL file.",
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks_path) if args.tasks_path else None
    results_dir = Path(args.results_dir)
    evaluate_model(args.model, results_dir, tasks_path)


if __name__ == "__main__":
    main()
