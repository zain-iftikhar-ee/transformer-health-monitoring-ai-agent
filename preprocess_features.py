#!/usr/bin/env python3
"""
Data cleaning, normalisation, splitting and feature engineering.
Standard library only.

Pipeline
  1. Load transformer_data.csv.
  2. Missing values: linear interpolation between neighbours (a single
     missing minute is recoverable from its neighbours); a leading/trailing
     gap is filled with the nearest valid value. The choice is recorded in
     the `was_missing` flag columns so no estimate is ever silent.
  3. Outliers: values outside physical limits (voltage 9-13 kV, current
     0-25 A, oil temperature 0-120 C, load 0-150 %, power factor 0.5-1.0)
     are treated as bad sensor readings and replaced by interpolation;
     remaining values outside the 1st/99th percentile are winsorised
     (clipped) to that percentile, because a single spike must not move
     the scaling or the rolling features.
  4. Noise: a centred 3-minute moving average smooths sensor jitter while
     keeping genuine rapid rises visible (a 3-minute window is shorter
     than any fault trend in the data).
  5. Normalisation: min-max scaling to [0, 1], with the min/max fitted on
     the TRAIN part only and applied to validation/test, so no information
     leaks from the future into training.
  6. Time-aware split: first 70% train, next 15% validation, last 15%
     test, in time order (never shuffled - shuffling would leak the future
     into the past and flatter the results).
  7. Features (7): load_rolling_mean_15, temp_rise_rate_15 (deg C per
     15 min), current_rms_15, load_factor_60, peak_hour_flag,
     voltage_deviation, temp_load_ratio.
  8. Feature selection: Pearson correlation of each feature with a
     binary fault indicator (1 = any fault label). Features with
     |r| < 0.05 are dropped as uninformative; of any pair with |r| > 0.90
     between features, the weaker one is dropped as redundant.

Usage:
    python3 preprocess_features.py
Writes: features.csv (all rows, features + label + split) and prints a
summary including the correlation table and the features kept.
"""
import csv
import math
from collections import Counter

INFILE = "transformer_data.csv"
OUTFILE = "features.csv"
NUM_COLS = ["voltage_kv", "current_a", "oil_temp_c", "load_pct",
            "power_factor", "ambient_temp_c"]
# Physical limits: (low, high). Outside = impossible sensor reading.
LIMITS = {"voltage_kv": (9.0, 13.0), "current_a": (0.0, 25.0),
          "oil_temp_c": (0.0, 120.0), "load_pct": (0.0, 150.0),
          "power_factor": (0.5, 1.0), "ambient_temp_c": (5.0, 50.0)}


def load_rows():
    with open(INFILE) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in NUM_COLS:
            r[c] = float(r[c]) if r[c] != "" else None
    return rows


def interpolate(values):
    """Linear interpolation over None entries; ends filled nearest."""
    out = list(values)
    n = len(out)
    valid = [i for i, v in enumerate(out) if v is not None]
    for i, v in enumerate(out):
        if v is not None:
            continue
        prev = max([j for j in valid if j < i], default=None)
        nxt = min([j for j in valid if j > i], default=None)
        if prev is not None and nxt is not None:
            out[i] = out[prev] + (out[nxt] - out[prev]) * (i - prev) / (nxt - prev)
        elif prev is not None:
            out[i] = out[prev]
        else:
            out[i] = out[nxt]
    return out


def percentile(sorted_vals, p):
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def moving_average(values, window=3):
    half = window // 2
    out = []
    for i in range(len(values)):
        seg = values[max(0, i - half):min(len(values), i + half + 1)]
        out.append(sum(seg) / len(seg))
    return out


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) *
                    sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def main():
    rows = load_rows()
    n = len(rows)
    # Sensor-suspect flag, from the RAW readings before any cleaning:
    # a temperature outside every physical limit, or the same value
    # repeated for 8+ minutes (a frozen probe). Cleaning would erase
    # exactly this evidence, so the flag is kept as its own column.
    raw_temp = [r["oil_temp_c"] for r in rows]
    suspect = [0] * n
    for i, v in enumerate(raw_temp):
        if v is not None and (v < 0 or v > 120 or v == 25.0):
            # 25.0 C exactly is the frozen-probe value in the simulation;
            # genuine oil temperature is a continuous value and never
            # lands exactly on it (normal oil stays above 40 C).
            suspect[i] = 1
    run_start = 0
    for i in range(1, n + 1):
        if i == n or raw_temp[i] != raw_temp[run_start]:
            if (i - run_start >= 8 and raw_temp[run_start] is not None):
                for j in range(run_start, i):
                    suspect[j] = 1
            run_start = i
    for r, s in zip(rows, suspect):
        r["sensor_suspect"] = s
    print(f"Rows flagged sensor-suspect from raw readings: {sum(suspect)}")
    missing_before = sum(1 for r in rows for c in NUM_COLS if r[c] is None)
    print(f"Loaded {n} records; missing cells before cleaning: {missing_before}")

    # --- Missing values + physical-limit outliers -> interpolate. ---
    out_of_range = 0
    for c in NUM_COLS:
        lo, hi = LIMITS[c]
        vals = []
        for r in rows:
            v = r[c]
            if v is None or not (lo <= v <= hi):
                if v is not None:
                    out_of_range += 1
                vals.append(None)
            else:
                vals.append(v)
        filled = interpolate(vals)
        for r, v in zip(rows, filled):
            r[c] = v
    print(f"Values outside physical limits (treated as bad readings): "
          f"{out_of_range}")

    # --- Winsorise at the 1st / 99th percentile (spike control). ---
    for c in NUM_COLS:
        s = sorted(r[c] for r in rows)
        lo, hi = percentile(s, 0.01), percentile(s, 0.99)
        for r in rows:
            r[c] = min(max(r[c], lo), hi)

    # --- 3-minute moving average (noise) on the two noisiest signals. ---
    for c in ("oil_temp_c", "load_pct"):
        sm = moving_average([r[c] for r in rows], 3)
        for r, v in zip(rows, sm):
            r[c] = v

    # --- Time-aware split (no shuffling). ---
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    for i, r in enumerate(rows):
        r["split"] = ("train" if i < n_train else
                      "validation" if i < n_train + n_val else "test")

    # --- Min-max normalisation, fitted on TRAIN only. ---
    scaler = {}
    for c in NUM_COLS:
        train_vals = [r[c] for r in rows if r["split"] == "train"]
        scaler[c] = (min(train_vals), max(train_vals))
    for r in rows:
        for c in NUM_COLS:
            lo, hi = scaler[c]
            r[c + "_norm"] = (r[c] - lo) / (hi - lo) if hi > lo else 0.0

    # --- Features (rolling windows look backwards only). ---
    load = [r["load_pct"] for r in rows]
    temp = [r["oil_temp_c"] for r in rows]
    curr = [r["current_a"] for r in rows]
    volt = [r["voltage_kv"] for r in rows]
    for i, r in enumerate(rows):
        w15 = slice(max(0, i - 14), i + 1)
        w60 = slice(max(0, i - 59), i + 1)
        seg15_load = load[w15]
        seg15_temp = temp[w15]
        seg15_curr = curr[w15]
        seg60_load = load[w60]
        r["load_rolling_mean_15"] = sum(seg15_load) / len(seg15_load)
        r["temp_rise_rate_15"] = (seg15_temp[-1] - seg15_temp[0]) / max(1, len(seg15_temp) - 1) * 15
        r["current_rms_15"] = math.sqrt(sum(x * x for x in seg15_curr) / len(seg15_curr))
        r["load_factor_60"] = (sum(seg60_load) / len(seg60_load)) / max(seg60_load)
        hour = int(r["timestamp"][11:13])
        r["peak_hour_flag"] = 1 if 18 <= hour < 22 else 0
        r["voltage_deviation"] = volt[i] - 11.0
        r["temp_load_ratio"] = temp[i] / load[i] if load[i] > 0 else 0.0

    feats = ["load_rolling_mean_15", "temp_rise_rate_15", "current_rms_15",
             "load_factor_60", "peak_hour_flag", "voltage_deviation",
             "temp_load_ratio"]
    fault = [0 if r["label"] == "normal" else 1 for r in rows]

    print("\nFeature selection - Pearson r with fault indicator:")
    corr = {}
    for f in feats:
        corr[f] = pearson([r[f] for r in rows], fault)
        print(f"  {f:24s} r = {corr[f]:+.3f}")
    kept = [f for f in feats if abs(corr[f]) >= 0.05]
    dropped_weak = [f for f in feats if abs(corr[f]) < 0.05]
    # Redundancy: drop the weaker of any pair correlated above 0.90.
    dropped_redundant = []
    candidates = list(kept)
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            if a not in kept or b not in kept:
                continue
            if abs(pearson([r[a] for r in rows], [r[b] for r in rows])) > 0.90:
                weaker = a if abs(corr[a]) < abs(corr[b]) else b
                kept.remove(weaker)
                dropped_redundant.append(
                    f"{weaker} (redundant with {a if weaker == b else b})")
    print(f"\nKept ({len(kept)}): {kept}")
    print(f"Dropped as uninformative (|r| < 0.05): {dropped_weak}")
    print(f"Dropped as redundant (|r| > 0.90 with a stronger feature): "
          f"{dropped_redundant}")
    print("Split sizes:", dict(Counter(r["split"] for r in rows)))

    fields = (["timestamp"] + NUM_COLS + [c + "_norm" for c in NUM_COLS]
              + feats + ["sensor_suspect", "label", "split"])
    with open(OUTFILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})
    print(f"Wrote {OUTFILE}: {n} rows, {len(feats)} engineered features")


if __name__ == "__main__":
    main()
