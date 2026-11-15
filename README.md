# Transformer Health Monitoring AI Agent

## What is here

| File | Task | What it does |
|---|---|---|
| `generate_data.py` | 1 | Simulates 10,800 one-minute records (7 variables + label) and writes `transformer_data.csv`. |
| `preprocess_features.py` | 1, 2 | Cleans the data (missing values, outliers, noise), normalises it, makes the time-aware train/validation/test split, engineers 7 features and selects features by correlation. Writes `features.csv`. |
| `agent.py` | 3, 4 | The agent prototype: tools `get_sensor_data()`, `predict_fault()`, `get_maintenance_history()`, a clearly labelled local LLM stub (no API key, no network), the orchestrator and the guardrails, with a Thought–Action–Observation loop. |
| `evaluate.py` | 4 | Runs the classifier on the held-out test split and the full agent on five scenarios, and prints accuracy, precision, recall and response-quality scores. |

## Requirements

Python 3.9 or newer. **Standard library only** — no packages to install,
so the code runs on any machine with Python 3.

## How to run

Run the commands in order, from inside this folder:

```bash
python3 generate_data.py       # creates transformer_data.csv (10,800 rows)
python3 preprocess_features.py # creates features.csv
python3 agent.py --scenario overheating   # demo: one full TAO loop
python3 evaluate.py            # prints the metrics used in the report
```

Other scenarios for the demo: `normal`, `overload`, `cooling_failure`,
`sensor_failure`.

## Notes

- The random seed is fixed (42), so the dataset and the metrics are
  reproducible: re-running the scripts gives the same numbers.
- The LLM in `agent.py` is a **stub**, labelled as such in the code and in
  the report. It formats reasoning from the input features instead of
  calling a remote model, so the prototype runs offline. Replace
  `LocalLLMStub.reason()` with a real LLM API call (same prompt, same
  JSON reply shape) for a live deployment.
- The agent never switches equipment by itself: any switching action is
  only proposed, and the guardrail layer records that human operator
  approval is required first (see Task 5 in the report).
