#!/usr/bin/env python3
"""
Agent prototype for the transformer health monitoring system.
Standard library only. No real LLM call is made and no API key is needed:
`LocalLLMStub` is CLEARLY labelled as a stub - it formats the same
structured reasoning an LLM would be asked to produce, from the features
and the rule-based diagnosis, so the prototype runs offline and its
behaviour is fully reproducible. In a live deployment this stub is replaced
by a real LLM API call with the same prompt and the same JSON reply shape.

Architecture (see the report, Figure 1):
    Sensors / SCADA -> Orchestrator -> Tools -> Rule classifier
                                        -> LLM stub (reasoning)
                                        -> Memory (history + cases)
                                        -> Guardrails -> Operator

Tools: get_sensor_data(), predict_fault(), get_maintenance_history().
Guardrails: no switching action is ever executed without human approval;
every diagnosis is checked against hard engineering limits, and a
disagreement is logged and resolved in favour of the stricter result.

Usage:
    python3 agent.py              (runs the overheating TAO loop demo)
    python3 agent.py --scenario overheating
    Scenarios: normal, overload, overheating, cooling_failure,
               sensor_failure
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime

DATAFILE = "features.csv"

# Hard engineering limits (IEC 60076-7 loading guide, summarised).
LIMIT_OIL_WARN = 75.0      # deg C: warning
LIMIT_OIL_CRITICAL = 90.0  # deg C: critical
LIMIT_LOAD_WARN = 100.0    # % of rating
LIMIT_TEMP_RISE = 8.0      # deg C rise in 15 minutes: rapid heating

MAINTENANCE_HISTORY = {
    "T-114": [
        {"date": "2026-03-12", "work": "Oil filtration and dissolved-gas test; gas levels normal."},
        {"date": "2026-06-28", "work": "Radiator cleaning; one cooling fan replaced."},
        {"date": "2026-09-15", "work": "Bushing inspection; no defects found."},
    ],
    "T-087": [
        {"date": "2026-05-04", "work": "Fan motor replaced after overheating alarm."},
        {"date": "2026-10-02", "work": "Oil top-up; temperature probe recalibrated."},
    ],
}

# Representative one-minute snapshots (from the simulated dataset's regimes).
SCENARIOS = {
    "normal": {"transformer_id": "T-114", "timestamp": "2026-11-03 10:00",
               "voltage_kv": 11.0, "current_a": 8.2, "oil_temp_c": 56.0,
               "load_pct": 62.0, "power_factor": 0.94, "ambient_temp_c": 27.0,
               "load_rolling_mean_15": 61.0, "temp_rise_rate_15": 0.8,
               "current_rms_15": 8.3, "load_factor_60": 0.88,
               "peak_hour_flag": 0, "voltage_deviation": 0.0,
               "temp_load_ratio": 0.90, "sensor_suspect": 0},
    "overload": {"transformer_id": "T-114", "timestamp": "2026-11-02 19:30",
                 "voltage_kv": 11.0, "current_a": 14.9, "oil_temp_c": 78.0,
                 "load_pct": 112.0, "power_factor": 0.91, "ambient_temp_c": 30.0,
                 "load_rolling_mean_15": 109.0, "temp_rise_rate_15": 4.5,
                 "current_rms_15": 14.6, "load_factor_60": 0.97,
                 "peak_hour_flag": 1, "voltage_deviation": 0.0,
                 "temp_load_ratio": 0.70, "sensor_suspect": 0},
    "overheating": {"transformer_id": "T-114", "timestamp": "2026-11-04 15:20",
                    "voltage_kv": 10.9, "current_a": 13.0, "oil_temp_c": 93.0,
                    "load_pct": 98.0, "power_factor": 0.92, "ambient_temp_c": 33.0,
                    "load_rolling_mean_15": 97.0, "temp_rise_rate_15": 9.5,
                    "current_rms_15": 12.9, "load_factor_60": 0.96,
                    "peak_hour_flag": 0, "voltage_deviation": -0.1,
                    "temp_load_ratio": 0.95, "sensor_suspect": 0},
    "cooling_failure": {"transformer_id": "T-087", "timestamp": "2026-11-05 03:10",
                        "voltage_kv": 11.1, "current_a": 7.6, "oil_temp_c": 84.0,
                        "load_pct": 58.0, "power_factor": 0.95, "ambient_temp_c": 22.0,
                        "load_rolling_mean_15": 57.0, "temp_rise_rate_15": 11.0,
                        "current_rms_15": 7.5, "load_factor_60": 0.90,
                        "peak_hour_flag": 0, "voltage_deviation": 0.1,
                        "temp_load_ratio": 1.45, "sensor_suspect": 0},
    "sensor_failure": {"transformer_id": "T-114", "timestamp": "2026-11-03 08:40",
                       "voltage_kv": 11.0, "current_a": 8.4, "oil_temp_c": 150.0,
                       "load_pct": 63.0, "power_factor": 0.94, "ambient_temp_c": 26.0,
                       "load_rolling_mean_15": 62.0, "temp_rise_rate_15": 45.0,
                       "current_rms_15": 8.3, "load_factor_60": 0.89,
                       "peak_hour_flag": 0, "voltage_deviation": 0.0,
                       "temp_load_ratio": 2.38, "sensor_suspect": 1},
}


# ---------------------------------------------------------------- tools
def get_sensor_data(transformer_id, timestamp=None):
    """Tool 1: latest one-minute telemetry for a transformer.

    In this prototype it reads the prepared features.csv; in deployment it
    queries the SCADA gateway. Returns None when no reading exists.
    """
    try:
        with open(DATAFILE) as f:
            rows = [r for r in csv.DictReader(f)]
    except FileNotFoundError:
        return None
    if timestamp:
        for r in rows:
            if r["timestamp"] == timestamp:
                return r
        return None
    return rows[-1] if rows else None


def predict_fault(features):
    """Tool 2: rule-based fault classifier over the engineered features.

    Rules follow the engineering limits in the report. Order matters:
    a broken sensor is checked first, because no other reading can be
    trusted once the probe itself is suspect.
    Returns (fault, confidence, reason).
    """
    oil = features["oil_temp_c"]
    load = features["load_pct"]
    rise = features["temp_rise_rate_15"]
    ratio = features["temp_load_ratio"]
    # Sensor failure is judged from the data-quality flag kept during
    # preprocessing: cleaning removes spikes, so the cleaned temperature
    # alone cannot show a broken probe.
    if features.get("sensor_suspect", 0) == 1 or oil > 120 or oil < 0:
        return ("sensor_failure", 0.97,
                "the temperature probe is reporting impossible values or "
                "a frozen value, so the probe itself is suspect")
    if oil >= LIMIT_OIL_CRITICAL:
        return ("overheating", 0.95,
                f"oil temperature {oil:.0f} C is above the critical limit "
                f"of {LIMIT_OIL_CRITICAL:.0f} C")
    if oil >= 76 and load < 78 and ratio > 1.20:
        return ("cooling_failure", 0.90,
                "oil is hot for the load being carried (temperature/load "
                "ratio above 1.20) while the load is ordinary, which "
                "points to failed cooling, not load")
    if load >= LIMIT_LOAD_WARN:
        return ("overload", 0.93,
                f"load {load:.0f}% is above the 100% rating")
    if oil >= LIMIT_OIL_WARN or rise > LIMIT_TEMP_RISE:
        return ("overheating", 0.72,
                "oil temperature or its rise rate is above the warning "
                "limit and needs watching")
    return ("normal", 0.96, "all readings are inside normal limits")


def get_maintenance_history(transformer_id):
    """Tool 3: past maintenance jobs for a transformer (memory)."""
    return MAINTENANCE_HISTORY.get(transformer_id, [])


# ------------------------------------------------------- LLM stub
class LocalLLMStub:
    """STUB - stands in for a real LLM API call (no key, no network).

    It receives the same prompt the real model would receive (features,
    classifier result, maintenance history) and returns the same JSON
    shape: condition, likely cause, confidence, recommended action.
    Its wording is assembled from the evidence, so it cannot invent a
    reading that is not in the input - the guardrail layer then checks
    the reply against the hard limits before anything is shown.
    """

    def reason(self, snapshot, fault, confidence, reason, history):
        tid = snapshot["transformer_id"]
        oil, load = snapshot["oil_temp_c"], snapshot["load_pct"]
        actions = {
            "normal": "No action needed. Continue routine monitoring.",
            "overload": "Shift part of the load to a neighbouring feeder "
                        "if possible, warn the operator, and re-check in "
                        "15 minutes. If load stays above 100% for an hour, "
                        "arrange a planned load transfer.",
            "overheating": "Raise a critical alert now. Reduce load if an "
                           "approved switching plan exists and a human "
                           "operator approves it. Send a crew to inspect "
                           "cooling and check the oil.",
            "cooling_failure": "Raise a high-priority alert. Send a crew "
                               "to inspect radiators and fans at first "
                               "light and consider de-loading the unit "
                               "until it is checked.",
            "sensor_failure": "Do not act on the temperature reading. "
                              "Raise a sensor-fault alert, verify the "
                              "probe on site, and use load and current "
                              "until the probe is repaired.",
        }
        causes = {
            "normal": "load and temperature are both inside normal limits",
            "overload": "demand has risen above the transformer rating",
            "overheating": "sustained high load has driven oil "
                           "temperature past its critical limit",
            "cooling_failure": "heat is not being removed although the "
                               "load is ordinary, so cooling has likely failed",
            "sensor_failure": "the temperature probe is reporting values "
                              "no real transformer could produce",
        }
        past = (" The most recent maintenance was: " +
                history[-1]["work"] if history else "")
        return {
            "condition": fault,
            "likely_cause": causes[fault] + "." + past,
            "confidence": round(confidence, 2),
            "evidence": reason + f" (oil {oil:.0f} C, load {load:.0f}%).",
            "recommended_action": actions[fault],
            "note": "STUB reply: produced locally from the input features, "
                    "not by a remote language model.",
        }


# ------------------------------------------------------------- guardrails
def guardrail_check(diagnosis, snapshot):
    """Hard-limit check + human-approval rule. Returns (ok, notes)."""
    notes = []
    oil = snapshot["oil_temp_c"]
    if diagnosis["condition"] == "normal" and oil >= LIMIT_OIL_CRITICAL:
        notes.append("OVERRIDE: LLM said normal but oil is above the "
                     "critical limit; severity raised to critical.")
        diagnosis = dict(diagnosis, condition="overheating")
    if "switch" in diagnosis["recommended_action"].lower():
        notes.append("GUARDRAIL: any switching action needs human operator "
                     "approval first; the agent only proposes, never executes.")
    else:
        notes.append("GUARDRAIL: no switching action proposed; nothing "
                     "requires operator approval in this reply.")
    return diagnosis, notes


# ------------------------------------------------------------ orchestrator
class Agent:
    def __init__(self):
        self.llm = LocalLLMStub()
        self.memory = []  # short-term memory: past diagnoses

    def diagnose(self, snapshot, verbose=True):
        """One Thought-Action-Observation (TAO) loop. Returns the report."""
        log = []
        tid = snapshot["transformer_id"]
        # Thought 1
        log.append(("Thought", "A new one-minute reading has arrived for "
                    f"{tid}. I need its features, a fault prediction and "
                    "its maintenance history before I judge it."))
        # Action 1: tools
        history = get_maintenance_history(tid)
        log.append(("Action", f"get_maintenance_history({tid!r})"))
        log.append(("Observation", f"{len(history)} past jobs found; latest: "
                    f"{history[-1]['work'] if history else 'none'}"))
        # Thought 2
        log.append(("Thought", "Now I classify the readings with the "
                    "rule-based tool, then ask the language model to "
                    "explain the result in words."))
        fault, conf, reason = predict_fault(snapshot)
        log.append(("Action", "predict_fault(features)"))
        log.append(("Observation", f"classifier says {fault} "
                    f"(confidence {conf:.2f}): {reason}"))
        # Thought 3
        llm_reply = self.llm.reason(snapshot, fault, conf, reason, history)
        log.append(("Action", "LLM reasoning (stub) over features + history"))
        log.append(("Observation", "LLM reply: " +
                    json.dumps(llm_reply, indent=None)))
        # Guardrails
        final, notes = guardrail_check(llm_reply, snapshot)
        for n in notes:
            log.append(("Guardrail", n))
        self.memory.append({"timestamp": snapshot["timestamp"],
                            "transformer": tid,
                            "condition": final["condition"]})
        if verbose:
            for kind, text in log:
                print(f"{kind}: {text}")
        return {"transformer": tid, "timestamp": snapshot["timestamp"],
                "diagnosis": final, "tao_log": log}


def main():
    ap = argparse.ArgumentParser(description="Transformer health agent demo")
    ap.add_argument("--scenario", default="overheating",
                    choices=list(SCENARIOS))
    args = ap.parse_args()
    print(f"=== Scenario: {args.scenario} ===")
    agent = Agent()
    result = agent.diagnose(SCENARIOS[args.scenario])
    print("\nFinal diagnosis:")
    print(json.dumps(result["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
