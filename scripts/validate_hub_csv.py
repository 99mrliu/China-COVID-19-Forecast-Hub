#!/usr/bin/env python3
"""Local Hub-format check for a forecast CSV (no R / hubValidations needed).

Checks, against hub-config/tasks.json for the weekly COVID target:
- UTF-8 without BOM, exactly the 8 Hub columns, no extra columns
- filename is <reference_date>-<team>-<model>.csv and matches the
  reference_date inside the file; reference_date is a Monday
- target is `wk inc covid prop ili`, location is CN,
  output_type is quantile
- every horizon present has exactly the 23 required quantiles,
  horizons are integers in [-5, 6]
- target_end_date == reference_date + horizon*7 days
- values are finite, >= 0 and <= 100, and non-decreasing across
  quantiles within each horizon (no quantile crossing)
- no duplicate rows, no missing values

Exit code 0 = pass, 1 = fail. This is a local pre-check only;
the authoritative check is the Hub's GitHub Actions validation
(hubValidations) on the pull request.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

HUB_QUANTILES = [
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99,
]
COLUMNS = [
    "reference_date", "target", "horizon", "target_end_date",
    "location", "output_type", "output_type_id", "value",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Forecast CSV to check")
    p.add_argument("--model-id", default="GZNL-Mantis")
    args = p.parse_args()

    path = Path(args.csv)
    errors: list[str] = []
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("file has a UTF-8 BOM (use encoding='utf-8', not 'utf-8-sig')")

    stem = path.stem
    parts = stem.split("-", 3)
    if len(parts) < 4:
        errors.append(f"filename {path.name} is not <date>-<team>-<model>.csv")
        ref_str, model_id = None, None
    else:
        ref_str = "-".join(parts[:3])
        model_id = parts[3]
        if model_id != args.model_id:
            errors.append(f"filename model_id {model_id} != expected {args.model_id}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot read CSV: {e}")
        return 1

    if list(df.columns) != COLUMNS:
        errors.append(f"columns are {list(df.columns)}, expected {COLUMNS}")
    if df.isna().any().any():
        errors.append("missing values present")
    if df.duplicated().any():
        errors.append("duplicate rows present")

    if ref_str:
        try:
            ref = date.fromisoformat(ref_str)
        except ValueError:
            errors.append(f"reference_date in filename is not a date: {ref_str}")
            ref = None
        else:
            if ref.weekday() != 0:
                errors.append(f"reference_date {ref} is not a Monday")
            if not (df["reference_date"] == ref_str).all():
                errors.append("reference_date column does not match filename")
    else:
        ref = None

    if not (df["target"] == "wk inc covid prop ili").all():
        errors.append("target is not 'wk inc covid prop ili' in every row")
    if not (df["location"] == "CN").all():
        errors.append("location is not CN in every row")
    if not (df["output_type"] == "quantile").all():
        errors.append("output_type is not quantile in every row")

    for horizon, g in df.groupby("horizon"):
        if not isinstance(horizon, (int, float)) or int(horizon) != horizon:
            errors.append(f"horizon {horizon} is not an integer")
            continue
        horizon = int(horizon)
        if horizon < -5 or horizon > 6:
            errors.append(f"horizon {horizon} outside Hub range -5..6")
        qs = sorted(g["output_type_id"].tolist())
        if qs != HUB_QUANTILES:
            errors.append(f"horizon {horizon}: quantiles {qs} != required 23")
        vals = g.sort_values("output_type_id")["value"].tolist()
        if any(v < 0 or v > 100 for v in vals):
            errors.append(f"horizon {horizon}: values outside [0, 100]: {vals}")
        if any(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
            errors.append(f"horizon {horizon}: quantiles cross (not monotonic)")
        if ref is not None:
            expected_end = (ref + timedelta(days=7 * horizon)).isoformat()
            if not (g["target_end_date"] == expected_end).all():
                errors.append(
                    f"horizon {horizon}: target_end_date != {expected_end}")

    if errors:
        print(f"FAIL: {path} ({len(errors)} problem(s))")
        for e in errors:
            print(f"  - {e}")
        return 1
    horizons = sorted(df["horizon"].unique().tolist())
    print(f"PASS: {path} ({len(df)} rows, horizons {horizons}, "
          f"23 quantiles each, values in [0, 100], monotonic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
