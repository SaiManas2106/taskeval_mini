
# TaskEval Mini

TaskEval Mini is a small benchmark for testing how well LLMs turn support-style requests into structured JSON actions.  
It includes a tiny JSONL dataset, a rule-based baseline, optional OpenAI model support, and simple metrics and summaries.

## Setup

```bash
git clone https://github.com/<your-username>/taskeval-mini.git
cd taskeval-mini

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
````

## Run evaluation

Rule-based baseline (no API key needed):

```bash
python scripts/run_eval.py --model rule_based
```

This creates a `results/` folder with:

* `<model>_predictions.jsonl`
* `<model>_metrics.csv`
* `<model>_summary.json`

## Optional: OpenAI model

Set your API key and run:

```bash
# in a new terminal after setting the key
# Windows
setx OPENAI_API_KEY "your-key-here"
# macOS / Linux
# export OPENAI_API_KEY="your-key-here"

python scripts/run_eval.py --model openai_gpt4o
```

## Dashboard

After running at least one evaluation:

```bash
streamlit run taskeval_mini/dashboard.py
```

The dashboard reads from `results/` and shows a simple summary and per-example metrics.

