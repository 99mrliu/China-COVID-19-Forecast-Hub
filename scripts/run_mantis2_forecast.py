#!/usr/bin/env python3
"""
Mantis 2 -> China Infectious Disease Forecast Hub (weekly COVID positivity).

What this does
---------------
1. Read the Hub target data (target-data/time-series.csv), filter
   `wk inc covid prop ili`, sort by target_end_date, take the last
   512 weekly observations (Mantis 2's context length; the series
   only has ~205 weeks, so in practice all of it is used).
2. Run Mantis v2 (carsondudley1/Mantis, tag mantis-v2.0.0),
   Chronos-2 backbone, 16-week horizon, single target, no covariate.
   Mantis 2 outputs 21 quantiles for 16 weeks.
3. Map the first forecast steps to Hub horizons and expand its
   21 quantiles to the Hub's required 23 quantiles.
4. Write <reference_date>-GZNL-Mantis2.csv in Hub format.

Mapping (computed, not hard-coded)
----------------------------------
- target_end_date values in the Hub are Mondays.
- Mantis 2 step i (0-based) forecasts the week ending
  last_observed_date + 7*(i+1) days.
- For a round with reference_date R (a Monday),
  horizon h = (target_end_date - R) / 7 days.
- Only horizons in [-5, 6] are emitted. With a 2-week data lag
  (last observed = R - 14 days) that gives horizons -1..6.
  The remaining Mantis 2 steps (weeks 9-16) are discarded: the
  Hub only accepts horizons up to +6.

Quantile expansion (much cleaner than v1)
------------------------------------------
- Mantis 2's 21 native quantiles:
  [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
  (from the v2.0.0 config.json, chronos_config.quantiles).
- The Hub requires 23 quantiles. Only TWO are missing (0.025 and
  0.975), both interior, obtained by linear interpolation between
  their neighbours (0.01<->0.05 and 0.95<->0.99). The 0.01/0.99
  tails are NATIVE in v2, unlike v1 where they were clamped to
  0.05/0.95 and understated.

Known limitations (do NOT hide these)
--------------------------------------
- Mantis 2 was trained on simulated + real infectious-disease
  counts, NOT on positivity rates. Feeding it a percentage (0-100)
  as if it were a count is off-label. Its arcsinh transform is
  generic (no per-target mean/std like v1's log1p z-score), but
  the first runs should still be treated as an experiment: check
  the medians against the recent trend, and do NOT submit until
  the user has reviewed them and the August-2026 backtest has
  been compared with v1 on the same origins.
- Seasonal features are generated inside Mantis 2 from a fixed
  synthetic start date, anchored to the first supplied week.
  Truncating the series for a backtest therefore also shifts the
  seasonal phase anchor, by design of the upstream code.
- No covariate is used. Mantis 2 can forecast up to 6 aligned
  series jointly (e.g. COVID + flu positivity); this script
  uses a single series for now.

Usage
-----
  python3 run_mantis2_forecast.py \
      --target-data target-data/time-series.csv \
      --reference-date 2026-11-23 \
      --checkpoint-dir models/mantis2 \
      --mantis-dir mantis2 \
      --output model-output/GZNL-Mantis2/2026-11-23-GZNL-Mantis2.csv

  --mode dummy   # no torch / no Mantis 2: use the last observed
                 # value as a flat median with a widening spread,
                 # ONLY to test the Hub-format conversion +
                 # validation end to end. Never submit dummy output.

Requires: numpy, pandas, torch, chronos-forecasting==2.3.2
(from `pip install "mantis[v2] @ git+https://github.com/
carsondudley1/Mantis.git@mantis-v2.0.0"`), and the v2 checkpoint
(models/mantis2/config.json + model.safetensors, ~478 MB, from
the mantis-v2.0.0 release).
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
MODEL_ABBR = "Mantis2"
MODEL_ID = f"{TEAM_ABBR}-{MODEL_ABBR}"
TARGET = "wk inc covid prop ili"
LOCATION = "CN"

MANTIS2_QUANTILES = np.array(
    [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
    dtype=float,
)
HUB_QUANTILES = np.array(
    [
        0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99,
    ],
    dtype=float,
)
# Mantis 2 quantiles missing from the Hub set: only the two interior
# ones, interpolated between neighbours. Everything else is native.
INTERPOLATED_QUANTILES = [0.025, 0.975]
INPUT_WINDOW = 512  # Mantis 2 context length; series has ~205 weeks
PREDICTION_LENGTH = 16  # Mantis 2 always forecasts 16 weeks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-data", required=True, help="Hub target-data/time-series.csv")
    p.add_argument("--reference-date", required=True, help="Monday, YYYY-MM-DD")
    p.add_argument("--checkpoint-dir", default="models/mantis2",
                   help="Dir with config.json + model.safetensors")
    p.add_argument("--mantis-dir", default="mantis2", help="Cloned Mantis repo dir (v2.0.0)")
    p.add_argument("--output", required=True, help="Output Hub CSV path")
    p.add_argument("--summary", default=None, help="Optional run-summary JSON path")
    p.add_argument("--mode", choices=["mantis2", "dummy"], default="mantis2")
    p.add_argument("--as-of-date", default=None,
                   help="Backtest only: pretend data ends on this date "
                        "(YYYY-MM-DD, a Monday in the series). The input "
                        "window is then the last weeks up to that date, "
                        "so no future information leaks in. Omit for a live run.")
    return p.parse_args()


def load_covid_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["target_end_date"])
    g = df[df["target"] == TARGET].copy()
    g = g.sort_values("target_end_date").drop_duplicates("target_end_date")
    if len(g) < 8:
        raise ValueError(
            f"COVID series has only {len(g)} weeks, Mantis 2 needs at least 8"
        )
    # Weekly continuity check (Mondays, 7 days apart).
    dates = g["target_end_date"].dt.date.tolist()
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    bad = [d for d in gaps if d != 7]
    if bad:
        print(f"WARNING: {len(bad)} non-7-day gaps in COVID series: {sorted(set(bad))}",
              file=sys.stderr)
    return g


def expand_quantiles(mantis2_21: np.ndarray) -> np.ndarray:
    """[H, 21] -> [H, 23]: linear interpolation for the two missing
    interior quantiles, then enforce monotonicity and clip to a
    valid positivity range."""
    out = np.empty((mantis2_21.shape[0], len(HUB_QUANTILES)), dtype=float)
    for i, row in enumerate(mantis2_21):
        row = np.maximum.accumulate(np.asarray(row, dtype=float))
        out[i] = np.interp(HUB_QUANTILES, MANTIS2_QUANTILES, row)
    out = np.maximum.accumulate(out, axis=1)
    return np.clip(out, 0.0, 100.0)


def run_mantis2(series: np.ndarray, checkpoint_dir: str, mantis_dir: str) -> np.ndarray:
    sys.path.insert(0, mantis_dir)
    from mantis.v2 import Mantis2  # noqa: E402  (needs torch, heavy import)

    model = Mantis2(checkpoint_dir=checkpoint_dir, device="cpu")
    pred = model.predict(time_series=series)
    pred = np.asarray(pred, dtype=float)
    if pred.shape != (1, len(MANTIS2_QUANTILES), PREDICTION_LENGTH):
        raise ValueError(f"Unexpected Mantis 2 output shape {pred.shape}")
    return pred[0].T  # [16, 21]: step-major


def run_dummy(series: np.ndarray) -> np.ndarray:
    """Flat-median dummy, for format testing only. Never submit this."""
    last = float(series[-1])
    med = np.full(PREDICTION_LENGTH, last)
    spread = np.linspace(0.4, 6.0, PREDICTION_LENGTH)
    z = {0.01: -2.33, 0.05: -1.64, 0.1: -1.28, 0.15: -1.04, 0.2: -0.84,
         0.25: -0.67, 0.3: -0.52, 0.35: -0.39, 0.4: -0.25, 0.45: -0.13,
         0.5: 0.0, 0.55: 0.13, 0.6: 0.25, 0.65: 0.39, 0.7: 0.52,
         0.75: 0.67, 0.8: 0.84, 0.85: 1.04, 0.9: 1.28, 0.95: 1.64,
         0.99: 2.33}
    cols = [med + z[q] * spread for q in MANTIS2_QUANTILES]
    return np.clip(np.stack(cols, axis=1), 0.0, 100.0)


def main() -> int:
    args = parse_args()
    ref = date.fromisoformat(args.reference_date)
    if ref.weekday() != 0:
        raise ValueError(f"reference_date {ref} is not a Monday")

    series_df = load_covid_series(args.target_data)
    if args.as_of_date:
        cutoff = date.fromisoformat(args.as_of_date)
        series_df = series_df[series_df["target_end_date"].dt.date <= cutoff]
        if len(series_df) < 8:
            raise ValueError(
                f"Only {len(series_df)} weeks up to {cutoff}, "
                f"Mantis 2 needs at least 8")
        print(f"Backtest as-of {cutoff}: series truncated to "
              f"{len(series_df)} weeks (no future leakage)")
    window = series_df.tail(INPUT_WINDOW)
    values = window["observation"].to_numpy(dtype=float)
    last_date: date = window["target_end_date"].dt.date.iloc[-1]

    print(f"COVID series: {len(series_df)} weeks, "
          f"{series_df['target_end_date'].min().date()} -> {last_date}")
    print(f"Input window: last {len(values)} weeks ending {last_date}, "
          f"last values {values[-5:].tolist()}")
    print("WARNING: Mantis 2 is being used off-label on a positivity "
          "rate (see script header). Review medians before submitting.",
          file=sys.stderr)

    if args.mode == "mantis2":
        mantis2_21 = run_mantis2(values, args.checkpoint_dir, args.mantis_dir)
    else:
        print("DUMMY MODE: output is a flat test forecast, DO NOT SUBMIT.",
              file=sys.stderr)
        mantis2_21 = run_dummy(values)

    hub_23 = expand_quantiles(np.clip(mantis2_21, 0.0, 100.0))

    rows = []
    emitted = []
    for i in range(PREDICTION_LENGTH):
        target_end = last_date + timedelta(days=7 * (i + 1))
        horizon = (target_end - ref).days // 7
        if horizon < -5 or horizon > 6:
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
        raise ValueError("No Mantis 2 step mapped into Hub horizons -5..6; "
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
    print(f"Mantis 2 native steps used: {len(emitted)} of "
          f"{PREDICTION_LENGTH} (rest beyond Hub horizon +6)")

    if args.summary:
        summary = {
            "model_id": MODEL_ID,
            "mode": args.mode,
            "reference_date": ref.isoformat(),
            "last_observed_date": last_date.isoformat(),
            "last_observed_value": float(values[-1]),
            "input_weeks": len(values),
            "horizons_emitted": emitted,
            "median_by_horizon": medians,
            "mantis_quantiles": MANTIS2_QUANTILES.tolist(),
            "hub_quantiles": HUB_QUANTILES.tolist(),
            "interpolated_quantiles": INTERPOLATED_QUANTILES,
            "output": str(out),
            "warnings": [
                "Mantis 2 used off-label on a positivity rate (trained on counts).",
                "Hub quantiles 0.025/0.975 interpolated between neighbours; "
                "0.01/0.99 tails are native Mantis 2 outputs.",
            ],
        }
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
        print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
