#!/usr/bin/env python3
"""
Task 1 - Data generation (COMP232 Assignment 1).
Simulates one-minute telemetry for a 250 kVA, 11 kV / 0.4 kV distribution
transformer (unit T-114) over 7.5 days: 10,800 records.

Variables (7, i.e. more than the required 5):
    timestamp, voltage_kv, current_a, oil_temp_c, load_pct,
    power_factor, ambient_temp_c
Plus a ground-truth label column used only for evaluation:
    normal / overload / overheating / cooling_failure / sensor_failure

The data are synthetic, generated with the random module (seed 42, so every
run reproduces the same file). Fault regimes are inserted as blocks so the
labels are known exactly. Missing values (~2%), Gaussian noise and a small
number of spikes are injected deliberately, and are then handled in
preprocess_features.py - they are part of the assignment, not accidents.

Usage:  python3 generate_data.py            (writes transformer_data.csv)
"""
import csv
import math
import random
from datetime import datetime, timedelta

SEED = 42
N_RECORDS = 10800          # 7.5 days x 24 h x 60 min
START = datetime(2026, 11, 1, 0, 0)
OUTFILE = "transformer_data.csv"

# Fault-regime blocks: (start_minute, end_minute, label).
# They cover about 30% of the record; the rest is normal operation.
REGIMES = [
    (900, 1020, "overload"),        # day 1 morning peak overload
    (1500, 1620, "cooling_failure"),
    (2100, 2160, "sensor_failure"),
    (3000, 3180, "overheating"),
    (3900, 4020, "overload"),
    (4700, 4820, "cooling_failure"),
    (5400, 5460, "sensor_failure"),
    (6100, 6300, "overheating"),
    (7000, 7140, "overload"),
    (7900, 8020, "cooling_failure"),
    (8700, 8880, "overheating"),
    (9500, 9620, "overload"),
    (9900, 10000, "overheating"),
    (10200, 10290, "cooling_failure"),
    (10500, 10560, "sensor_failure"),
]


def label_at(minute):
    for start, end, label in REGIMES:
        if start <= minute < end:
            return label
    return "normal"


def main():
    rng = random.Random(SEED)
    oil = 52.0  # thermal state: oil temperature follows load with a lag
    rows = []
    for minute in range(N_RECORDS):
        ts = START + timedelta(minutes=minute)
        hour = ts.hour + ts.minute / 60.0
        label = label_at(minute)

        # Diurnal load: low at night, evening peak (18:00-22:00).
        base_load = 58 + 22 * math.sin((hour - 9) / 24 * 2 * math.pi)
        base_load += 8 * math.exp(-((hour - 19.5) ** 2) / 3.0)
        ambient = 24 + 7 * math.sin((hour - 14) / 24 * 2 * math.pi)

        if label == "overload":
            load = rng.uniform(104, 118)
        elif label in ("overheating",):
            load = rng.uniform(92, 106)
        elif label == "cooling_failure":
            load = rng.uniform(50, 66)          # load is ordinary: heat is the clue
        else:
            load = max(28.0, base_load + rng.gauss(0, 4))

        # Oil temperature: first-order lag towards a load-driven target.
        target = 30 + 0.42 * load + 0.55 * (ambient - 24)
        if label == "overheating":
            target += rng.uniform(16, 24)
        elif label == "cooling_failure":
            target += rng.uniform(20, 30)       # cooling cannot shed the heat
        elif label == "overload":
            target += rng.uniform(4, 9)
        oil += 0.06 * (target - oil) + rng.gauss(0, 0.25)
        oil = max(25.0, oil)

        voltage = 11.0 + rng.gauss(0, 0.08)
        if label == "normal" and rng.random() < 0.01:
            voltage -= rng.uniform(0.3, 0.7)    # occasional small network dip
        current = load / 100.0 * 13.1 * rng.uniform(0.98, 1.02)
        pf = max(0.80, min(0.99, 0.94 + rng.gauss(0, 0.015)
                            - (0.03 if load > 100 else 0.0)))

        temp_reading = oil
        if label == "sensor_failure":
            # Stuck, spiking or wildly wrong probe readings.
            kind = minute % 2
            if kind == 0:
                temp_reading = 150.0 + rng.gauss(0, 3)   # impossible spike
            else:
                temp_reading = 25.0                       # frozen low value

        row = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            "voltage_kv": round(voltage, 3),
            "current_a": round(current, 2),
            "oil_temp_c": round(temp_reading, 2),
            "load_pct": round(load, 2),
            "power_factor": round(pf, 3),
            "ambient_temp_c": round(ambient + rng.gauss(0, 0.4), 2),
            "label": label,
        }
        rows.append(row)

    # Inject missing values (~2% of cells) and isolated outliers (~0.3%).
    for row in rows:
        for col in ("voltage_kv", "current_a", "oil_temp_c", "load_pct",
                    "power_factor", "ambient_temp_c"):
            r = rng.random()
            if r < 0.02:
                row[col] = ""
            elif r < 0.023:
                row[col] = round(row[col] * rng.choice([1.6, 0.4]), 2)

    with open(OUTFILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    missing = sum(1 for r in rows for c in
                  ("voltage_kv", "current_a", "oil_temp_c", "load_pct",
                   "power_factor", "ambient_temp_c") if r[c] == "")
    print(f"Wrote {OUTFILE}: {len(rows)} records, 7 variables + label")
    print("Label counts:", dict(counts))
    print(f"Missing cells injected: {missing}")


if __name__ == "__main__":
    main()
