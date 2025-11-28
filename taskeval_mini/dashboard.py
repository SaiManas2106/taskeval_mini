from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def load_summary_files(results_dir: Path):
    summaries = []
    for path in results_dir.glob("*_summary.json"):
        model_name = path.stem.replace("_summary", "")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["model_name"] = model_name
        summaries.append(data)
    return summaries


def load_metrics_csv(results_dir: Path):
    tables = {}
    for path in results_dir.glob("*_metrics.csv"):
        model_name = path.stem.replace("_metrics", "")
        df = pd.read_csv(path)
        tables[model_name] = df
    return tables


def main():
    st.title("TaskEval Mini – LLM Benchmark Dashboard")

    results_dir = Path("results")
    if not results_dir.exists():
        st.info("No results directory found. Run the evaluation script first.")
        return

    summaries = load_summary_files(results_dir)
    if not summaries:
        st.info("No summary files found in results/.")
        return

    summary_df = pd.DataFrame(summaries)
    st.subheader("Model level summary")
    st.dataframe(summary_df.set_index("model_name"))

    st.subheader("Per example metrics")
    tables = load_metrics_csv(results_dir)
    model_names = sorted(tables.keys())
    if not model_names:
        st.info("No metrics CSV files found.")
        return

    selected_model = st.selectbox("Select model", model_names)
    st.dataframe(tables[selected_model])

    st.caption(
        "This dashboard is a lightweight view over CSV and JSON summaries. "
        "Extend it with more plots as you add new metrics or tasks."
    )


if __name__ == "__main__":
    main()
