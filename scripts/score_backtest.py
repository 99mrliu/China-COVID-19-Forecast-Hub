#!/usr/bin/env python3
"""Score one Hub-format forecast CSV against the Hub target data.

Metrics per horizon (and pooled):
  - actual value, median, absolute error of the median
  - WIS (weighted interval score, Hubverse definition, 23 quantiles ->
    11 symmetric intervals + median term, lower is better)
  - 50% interval (q25-q75) and 90% interval (q05-q95) coverage flags

Usage:
  python3 score_backtest.py --forecast <csv> --target-data <csv> [--output <json>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "wk inc covid prop ili"


def wis_for_row(quantiles: dict[float, float], y: float) -> float:
    """Hubverse WIS from a {tau: value} dict with a 0.5 median."""
    m = quantiles[0.5]
    total = 0.5 * abs(y - m)
    k = 0
    for lo_tau, hi_tau in [(0.01, 0.99), (0.025, 0.975), (0.05, 0.95),
                           (0.1, 0.9), (0.15, 0.85), (0.2, 0.8),
                           (0.25, 0.75), (0.3, 0.7), (0.35, 0.65),
                           (0.4, 0.6), (0.45, 0.55)]:
        lo, hi = quantiles[lo_tau], quantiles[hi_tau]
        alpha = 1.0 - (hi_tau - lo_tau)  # e.g. 0.02 for the 98% interval
        score = (hi - lo)
        if y < lo:
            score += (2.0 / alpha) * (lo - y)
        elif y > hi:
            score += (2.0 / alpha) * (y - hi)
        total += score
        k += 1
    return total / (k + 0.5)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecast", required=True)
    p.add_argument("--target-data", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    fc = pd.read_csv(args.forecast, parse_dates=["target_end_date"])
    td = pd.read_csv(args.target_data, parse_dates=["target_end_date"])
    actual = (td[td["target"] == TARGET]
              .set_index("target_end_date")["observation"])

    rows = []
    for (horizon, target_end), g in fc.groupby(["horizon", "target_end_date"]):
        q = dict(zip(g["output_type_id"], g["value"]))
        if target_end not in actual.index:
            print(f"horizon {horizon} ({target_end.date()}): actual not yet known, skipped")
            continue
        y = float(actual[target_end])
        med = q[0.5]
        rows.append({
            "horizon": int(horizon),
            "target_end_date": target_end.date().isoformat(),
            "actual": round(y, 2),
            "median": round(med, 2),
            "abs_error": round(abs(med - y), 2),
            "wis": round(wis_for_row(q, y), 2),
            "in_50pct": bool(q[0.25] <= y <= q[0.75]),
            "in_90pct": bool(q[0.05] <= y <= q[0.95]),
            "q05": round(q[0.05], 2),
            "q95": round(q[0.95], 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False) if not df.empty else "No scored horizons (actuals not yet known).")
    if not df.empty:
        print(f"\nPooled over {len(df)} horizons: "
              f"MAE(median)={df['abs_error'].mean():.2f}, "
              f"mean WIS={df['wis'].mean():.2f}, "
              f"50% coverage={df['in_50pct'].mean():.0%}, "
              f"90% coverage={df['in_90pct'].mean():.0%}")
    if args.output:
        out = {"rows": rows,
               "pooled": ({"n": len(df),
                           "mae_median": round(float(df['abs_error'].mean()), 3),
                           "mean_wis": round(float(df['wis'].mean()), 3),
                           "coverage_50": round(float(df['in_50pct'].mean()), 3),
                           "coverage_90": round(float(df['in_90pct'].mean()), 3)}
                          if not df.empty else None)}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
