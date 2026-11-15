#!/usr/bin/env python3
"""
Evaluation of the agent and classifier. Standard library only.

1. Loads features.csv, runs predict_fault() from agent.py over the
   time-aware TEST split (the last 15% of records, never seen in
   normalisation fitting), and reports overall accuracy and per-class
   precision and recall against the ground-truth labels.
2. Runs the full agent (tools + LLM stub + guardrails) on five scenarios
   - normal, overload, overheating, cooling_failure, sensor_failure -
   and scores response quality on four checks that can be tested from
   the reply itself: it names the condition, gives a cause, states a
   confidence, and gives a concrete next action.
3. Prints every number used in the report (Section 7). Nothing here is
   hand-written: re-run this file and the numbers change only if the
   code or the data change.

Usage:  python3 evaluate.py
"""
import csv
from collections import Counter, defaultdict

from agent import Agent, SCENARIOS, predict_fault

LABELS = ["normal", "overload", "overheating", "cooling_failure",
          "sensor_failure"]


def load_test_rows():
    with open("features.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    for r in rows:
        for c in ("oil_temp_c", "load_pct", "temp_rise_rate_15",
                  "temp_load_ratio"):
            r[c] = float(r[c])
        r["sensor_suspect"] = int(float(r["sensor_suspect"]))
    return rows


def main():
    rows = load_test_rows()
    print(f"Test set: {len(rows)} records "
          f"({rows[0]['timestamp']} to {rows[-1]['timestamp']})")
    print("True label counts:", dict(Counter(r["label"] for r in rows)))

    # --- Classification metrics on the test split. ---
    confusion = Counter()
    for r in rows:
        pred, _, _ = predict_fault(r)
        confusion[(r["label"], pred)] += 1
    correct = sum(v for (t, p), v in confusion.items() if t == p)
    accuracy = correct / len(rows)
    print(f"\nOverall accuracy: {correct}/{len(rows)} = {accuracy:.4f}")
    print(f"\n{'class':16s} {'n':>5s} {'precision':>10s} {'recall':>8s}")
    per_class = {}
    for label in LABELS:
        tp = confusion[(label, label)]
        fp = sum(v for (t, p), v in confusion.items()
                 if p == label and t != label)
        fn = sum(v for (t, p), v in confusion.items()
                 if t == label and p != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = (prec, rec)
        print(f"{label:16s} {tp + fn:5d} {prec:10.3f} {rec:8.3f}")
    print("\nConfusion (true -> predicted):")
    for label in LABELS:
        preds = {p: confusion[(label, p)] for p in LABELS
                 if confusion[(label, p)]}
        print(f"  {label:16s} -> {preds}")

    # --- Five-scenario agent test + response-quality scoring. ---
    print("\nFive-scenario agent test:")
    agent = Agent()
    quality_scores = {}
    scenario_correct = 0
    for name, snap in SCENARIOS.items():
        result = agent.diagnose(snap, verbose=False)
        d = result["diagnosis"]
        checks = {
            "names condition": d["condition"] in
                ("normal", "overload", "overheating", "cooling_failure",
                 "sensor_failure"),
            "gives a cause": len(d["likely_cause"]) > 20,
            "states confidence": 0.0 <= d["confidence"] <= 1.0,
            "gives a concrete action": len(d["recommended_action"]) > 20,
        }
        score = sum(checks.values())
        quality_scores[name] = score
        ok = d["condition"] == name
        scenario_correct += ok
        print(f"  {name:16s} diagnosed as {d['condition']:16s} "
              f"({'CORRECT' if ok else 'WRONG'}, confidence "
              f"{d['confidence']:.2f}) - response quality {score}/4, "
              f"checks passed: {[k for k, v in checks.items() if v]}")
    mean_quality = sum(quality_scores.values()) / len(quality_scores)
    print(f"\nScenario accuracy: {scenario_correct}/5")
    print(f"Mean response quality: {mean_quality:.2f} / 4")
    print(f"\nSUMMARY FOR REPORT: accuracy={accuracy:.3f} "
          f"scenario_accuracy={scenario_correct}/5 "
          f"mean_response_quality={mean_quality:.2f}/4")
    for label in LABELS:
        p, r = per_class[label]
        print(f"  {label}: precision={p:.3f} recall={r:.3f}")


if __name__ == "__main__":
    main()
