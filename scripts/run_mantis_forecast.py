#!/usr/bin/env python3
"""
Mantis -> China Infectious Disease Forecast Hub (weekly COVID positivity).

What this does
---------------
1. Read the Hub target data (target-data/time-series.csv), filter
   `wk inc covid prop ili`, sort by target_end_date, take the last 112
   weekly observations (Mantis input window = 112 weeks).
2. Run Mantis v1 (carsondudley1/Mantis), 8-week horizon, no covariate,
   target_type=0 (cases -- the closest of Mantis's 3 types to a rate).
3. Map Mantis's 8 forecast steps to Hub horizons and expand its
   9 quantiles to the Hub's required 23 quantiles.
4. Write <reference_date>-GZNL-Mantis.csv in Hub format.

Mapping (computed, not hard-coded)
----------------------------------
- target_end_date values in the Hub are Mondays.
- Mantis step i (0-based) forecasts the week ending
  last_observed_date + 7*(i+1) days.
- For a round with reference_date R (a Monday),
  horizon h = (target_end_date - R) / 7 days.
- Only horizons in [-5, 6] are emitted. With the data as of
  2026-11-13 (last observed 2026-10-26) and R = 2026-11-16,
  that gives horizons -2..5. Once the Hub target data catches up
  to the week before R, the same code gives -1..6.

Known limitations (do NOT hide these)
--------------------------------------
- Mantis was trained on simulated counts (cases/hosp/deaths), NOT on
  positivity rates. Feeding it a percentage (0-100) as if it were a
  case count is off-label: its log1p normalisation uses the
  "cases" mean/std (5.55 / 3.63). Treat the first runs as an
  experiment, check the medians against the recent trend, and do
  NOT submit until the user has reviewed them.
- Mantis outputs 9 quantiles
  [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95].
  The Hub requires 23 quantiles. The 23 are obtained by linear
  interpolation in quantile space, clamped at the Mantis extremes,
  so the 0.01 / 0.99 tails are UNDERSTATED (they equal the
  0.05 / 0.95 values). This is an approximation, not a Mantis
  native output.
- No covariate is used (flu positivity is available in the Hub
  target data and could be a covariate in a later version).
- Population is left at Mantis's default (None -> 0.0 after
  log1p), matching the upstream demo notebook.

Usage
-----
  python3 run_mantis_forecast.py \
      --target-data target-data/time-series.csv \
      --reference-date 2026-11-16 \
      --model-dir models \
      --mantis-dir mantis \
      --output model-output/GZNL-Mantis/2026-11-16-GZNL-Mantis.csv

  --mode dummy   # no torch / no Mantis: use the last observed value
                 # as a flat median with a widening spread, ONLY to
                 # test the Hub-format conversion + validation end
                 # to end. Never submit dummy output.

Requires: numpy, pandas, torch, and the Mantis package on
PYTHONPATH (the cloned carsondudley1/Mantis repo), plus
models/mantis_8w_nocov.pt .
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TEAM_ABBR = "GZNL"
MODEL_ABBR = "Mantis"
MODEL_ID = f"{TEAM_ABBR}-{MODEL_ABBR}"
TARGET = "wk inc covid prop ili"
LOCATION = "CN"

MANTIS_QUANTILES = np.array(
    [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95], dtype=float
)
HUB_QUANTILES = np.array(
    [
        0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99,
    ],
    dtype=float,
)
INPUT_WINDOW = 112
FORECAST_HORIZON = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-data", required=True, help="Hub target-data/time-series.csv")
    p.add_argument("--reference-date", required=True, help="Monday, YYYY-MM-DD")
    p.add_argument("--model-dir", default="models", help="Dir with mantis_8w_nocov.pt")
    p.add_argument("--mantis-dir", default="mantis", help="Cloned Mantis repo dir")
    p.add_argument("--output", required=True, help="Output Hub CSV path")
    p.add_argument("--summary", default=None, help="Optional run-summary JSON path")
    p.add_argument("--mode", choices=["mantis", "dummy"], default="mantis")
    return p.parse_args()


def load_covid_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["target_end_date"])
    g = df[df["target"] == TARGET].copy()
    g = g.sort_values("target_end_date").drop_duplicates("target_end_date")
    if len(g) < INPUT_WINDOW:
        raise ValueError(
            f"COVID series has only {len(g)} weeks, Mantis needs {INPUT_WINDOW}"
        )
    # Weekly continuity check (Mondays, 7 days apart).
    dates = g["target_end_date"].dt.date.tolist()
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    bad = [d for d in gaps if d != 7]
    if bad:
        print(f"WARNING: {len(bad)} non-7-day gaps in COVID series: {sorted(set(bad))}",
              file=sys.stderr)
    return g


def expand_quantiles(mantis_9: np.ndarray) -> np.ndarray:
    """[H, 9] -> [H, 23]: linear interpolation, clamped at the extremes,
    then enforce monotonicity and clip to a valid positivity range."""
    out = np.empty((mantis_9.shape[0], len(HUB_QUANTILES)), dtype=float)
    for i, row in enumerate(mantis_9):
        row = np.maximum.accumulate(np.asarray(row, dtype=float))
        out[i] = np.interp(HUB_QUANTILES, MANTIS_QUANTILES, row)
    out = np.maximum.accumulate(out, axis=1)
    return np.clip(out, 0.0, 100.0)


def run_mantis(series: np.ndarray, model_dir: str, mantis_dir: str) -> np.ndarray:
    sys.path.insert(0, mantis_dir)
    from mantis import Mantis  # noqa: E402  (needs torch, heavy import)

    model = Mantis(forecast_horizon=FORECAST_HORIZON, use_covariate=False,
                   model_dir=model_dir)
    pred = model.predict(time_series=series, target_type=0)
    pred = np.asarray(pred, dtype=float)
    if pred.shape != (FORECAST_HORIZON, len(MANTIS_QUANTILES)):
        raise ValueError(f"Unexpected Mantis output shape {pred.shape}")
    return pred


def run_dummy(series: np.ndarray) -> np.ndarray:
    """Flat-median dummy, for format testing only. Never submit this."""
    last = float(series[-1])
    med = np.full(FORECAST_HORIZON, last)
    spread = np.array([0.4, 0.7, 1.2, 1.8, 2.4, 3.0, 3.8, 4.6])
    cols = []
    for q in MANTIS_QUANTILES:
        # crude symmetric spread around the median, widening with horizon
        z = {0.05: -1.64, 0.1: -1.28, 0.25: -0.67, 0.4: -0.25, 0.5: 0.0,
             0.6: 0.25, 0.75: 0.67, 0.9: 1.28, 0.95: 1.64}[q]
        cols.append(med + z * spread)
    return np.clip(np.stack(cols, axis=1), 0.0, 100.0)


def main() -> int:
    args = parse_args()
    ref = date.fromisoformat(args.reference_date)
    if ref.weekday() != 0:
        raise ValueError(f"reference_date {ref} is not a Monday")

    series_df = load_covid_series(args.target_data)
    last112 = series_df.tail(INPUT_WINDOW)
    values = last112["observation"].to_numpy(dtype=float)
    last_date: date = last112["target_end_date"].dt.date.iloc[-1]

    print(f"COVID series: {len(series_df)} weeks, "
          f"{series_df['target_end_date'].min().date()} -> {last_date}")
    print(f"Input window: last {INPUT_WINDOW} weeks ending {last_date}, "
          f"last values {values[-5:].tolist()}")
    print("WARNING: Mantis is being used off-label on a positivity "
          "rate (see script header). Review medians before submitting.",
          file=sys.stderr)

    if args.mode == "mantis":
        mantis_9 = run_mantis(values, args.model_dir, args.mantis_dir)
    else:
        print("DUMMY MODE: output is a flat test forecast, DO NOT SUBMIT.",
              file=sys.stderr)
        mantis_9 = run_dummy(values)

    hub_23 = expand_quantiles(np.clip(mantis_9, 0.0, 100.0))

    rows = []
    emitted = []
    for i in range(FORECAST_HORIZON):
        target_end = last_date + timedelta(days=7 * (i + 1))
        horizon = (target_end - ref).days // 7
        if horizon < -5 or horizon > 6:
            print(f"Skipping Mantis step {i} (target_end {target_end}, "
                  f"horizon {horizon} outside Hub range -5..6)")
            continue
        emitted.append(horizon)
        for qi, q in enumerate(HUB_QUANTILES):
            rows.append({
                "reference_date": ref.isoformat(),
                "target": TARGET,
                "horizon": int(horizon),
                "target_end_date": target_end.isoformat(),
                "location": LOCATION,
                "output_type": "quantile",
                "output_type_id": float(q),
                "value": round(float(hub_23[i, qi]), 2),
            })
    if not rows:
        raise ValueError("No Mantis step mapped into Hub horizons -5..6; "
                         "check reference_date vs last observed date.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=[
        "reference_date", "target", "horizon", "target_end_date",
        "location", "output_type", "output_type_id", "value",
    ]).to_csv(out, index=False, encoding="utf-8")

    medians = {
        int(h): round(float(hub_23[i, list(HUB_QUANTILES).index(0.5)]), 2)
        for i, h in enumerate(emitted)
    }
    print(f"Wrote {out} ({len(rows)} rows, horizons {emitted})")
    print(f"Median forecast by horizon: {medians}")

    if args.summary:
        summary = {
            "model_id": MODEL_ID,
            "mode": args.mode,
            "reference_date": ref.isoformat(),
            "last_observed_date": last_date.isoformat(),
            "last_observed_value": float(values[-1]),
            "input_weeks": INPUT_WINDOW,
            "horizons_emitted": emitted,
            "median_by_horizon": medians,
            "mantis_quantiles": MANTIS_QUANTILES.tolist(),
            "hub_quantiles": HUB_QUANTILES.tolist(),
            "output": str(out),
            "warnings": [
                "Mantis used off-label on a positivity rate (trained on counts).",
                "Hub 0.01/0.99 tails are clamped to Mantis 0.05/0.95 (understated).",
            ],
        }
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
        print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
