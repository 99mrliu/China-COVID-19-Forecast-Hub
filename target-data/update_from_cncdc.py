#!/usr/bin/env python3
"""Update Hub target data from cn_cdc_dashboard surveillance CSVs.

Targets:
- wk inc covid prop ili  (weekly, from cncdc_surveillance_covid19.csv)
- wk inc flu prop ili    (weekly, from cncdc_surveillance_all.csv)
- mo inc dengue cases    (monthly, from notifiable_all.csv)

Dengue history before the official China CDC crawl (2004-01 through the
month before official coverage) is kept by merging
data/backfill/notifiable-flu-dengue.csv the same way
cn_cdc_dashboard scripts/sync-notifiable.mjs does: an existing
(month, disease) row wins, and the backfill only fills gaps. That
extract also contains monthly notifiable influenza case counts. Those
are not `wk inc flu prop ili` (weekly sentinel ILI positivity) and are
not imported, and this hub has no monthly flu target.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

CNCDC_BASE = (
    "https://raw.githubusercontent.com/dailypartita/cn_cdc_dashboard/main/data/"
)
COVID_URL = CNCDC_BASE + "cncdc_surveillance_covid19.csv"
SURV_ALL_URL = CNCDC_BASE + "cncdc_surveillance_all.csv"
NOTIFIABLE_URL = CNCDC_BASE + "notifiable_all.csv"
# Compiled national monthly history. Official syncs merge this file in
# writeOutputs and never overwrite a (month, disease) the crawl returned.
BACKFILL_URL = CNCDC_BASE + "backfill/notifiable-flu-dengue.csv"
LOCAL_DATA_DIR = Path("/data/ykx/hubverse_proj/00-cn_cdc_dashboard/data")
DENGUE_DISEASE = "登革热"
# Monthly notifiable influenza counts in the backfill extract. This hub's
# flu target is weekly sentinel positivity, so those rows are ignored.
MONTHLY_FLU_DISEASE = "流行性感冒"

TARGET_COVID = "wk inc covid prop ili"
TARGET_FLU = "wk inc flu prop ili"
TARGET_DENGUE = "mo inc dengue cases"
LOCATION = "CN"
MIN_DATE_COVID = "2022-12-05"
MAX_ORACLE_LOOKBACK = 6
QUANTILE_LEVELS = [
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99,
]

DENGUE_HORIZONS = [-1, 0, 1, 2, 3]
DENGUE_FIRST_REFERENCE = "2024-01-01"
DENGUE_LAST_REFERENCE = "2027-10-01"

ROOT = Path(__file__).resolve().parent
TIME_SERIES_PATH = ROOT / "time-series.csv"
ORACLE_PATH = ROOT / "oracle-output.csv"
TASKS_PATH = ROOT.parent / "hub-config" / "tasks.json"


def load_csv_rows(url: str, csv_path: Path | None) -> list[dict[str, str]]:
    """Load rows from a local CSV, or fall back to GitHub raw."""
    if csv_path is not None:
        path = csv_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        text = path.read_text(encoding="utf-8-sig")
        return list(csv.DictReader(text.splitlines()))

    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


def local_or_github(filename: str, url: str, explicit: Path | None,
                    from_github: bool) -> tuple[list[dict[str, str]], str]:
    if not from_github:
        candidate = explicit or (LOCAL_DATA_DIR / filename)
        if candidate.exists():
            return load_csv_rows(url, candidate), str(candidate)
    return load_csv_rows(url, None), url


def build_weekly_series(rows: list[dict[str, str]], pathogen: str,
                        target: str, min_date: str | None) -> list[dict[str, str]]:
    series: dict[str, float] = {}
    for row in rows:
        if row.get("pathogen") != pathogen:
            continue
        date = row["reference_date"]
        if min_date and date < min_date:
            continue
        value = row.get("ili_percent", "").strip()
        if not value:
            continue
        series[date] = float(value)

    return [
        {
            "target_end_date": date,
            "target": target,
            "location": LOCATION,
            "observation": f"{series[date]:g}",
        }
        for date in sorted(series.keys(), reverse=True)
    ]


def is_dengue_row(row: dict[str, str]) -> bool:
    return row.get("disease") == DENGUE_DISEASE and row.get("row_kind") == "disease"


def month_key(row: dict[str, str]) -> str:
    month = (row.get("month") or "").strip()
    if month:
        return month
    return row["reference_date"][:7]


def merge_dengue_backfill(notifiable_rows: list[dict[str, str]],
                          backfill_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Append compiled dengue months the official table does not already have.

    Mirrors ``mergeNotifiableBackfill`` in cn_cdc_dashboard: existing
    ``(month, disease)`` keys stay authoritative (2024+ China CDC rows
    are left untouched, including their case counts). Monthly influenza
    rows in the same extract are skipped.
    """
    present = {month_key(row) for row in notifiable_rows if is_dengue_row(row)}
    merged = list(notifiable_rows)
    added = 0
    for row in backfill_rows:
        if not is_dengue_row(row):
            continue
        key = month_key(row)
        if key in present:
            continue
        if not row.get("cases", "").strip():
            continue
        merged.append(row)
        present.add(key)
        added += 1
    return merged, added


def build_dengue_series(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    series: dict[str, int] = {}
    for row in rows:
        if not is_dengue_row(row):
            continue
        cases = row.get("cases", "").strip()
        if not cases:
            continue
        # reference_date is the first day of the reported month.
        # Later duplicate months do not override an earlier official row:
        # merge_dengue_backfill only appends months that were absent.
        series.setdefault(row["reference_date"], int(float(cases)))

    return [
        {
            "target_end_date": date,
            "target": TARGET_DENGUE,
            "location": LOCATION,
            "observation": str(series[date]),
        }
        for date in sorted(series.keys(), reverse=True)
    ]


def assert_dengue_history(series: list[dict[str, str]],
                          backfill_rows: list[dict[str, str]],
                          previous: list[dict[str, str]]) -> None:
    """Refuse a sync that would drop compiled history or open a gap."""
    dates = sorted(row["target_end_date"] for row in series)
    if not dates:
        raise RuntimeError("dengue series is empty")
    expected = month_starts(dates[0], dates[-1])
    if dates != expected:
        missing = [date for date in expected if date not in set(dates)]
        raise RuntimeError(
            f"dengue series has {len(missing)} gaps, first {missing[:3]}"
        )

    required = sorted({
        row["reference_date"]
        for row in backfill_rows
        if is_dengue_row(row) and row.get("cases", "").strip()
    })
    if not required:
        raise RuntimeError(
            "dengue backfill extract has no disease rows; refusing to "
            "rewrite target data without the historical series"
        )
    have = set(dates)
    missing_backfill = [date for date in required if date not in have]
    if missing_backfill:
        raise RuntimeError(
            "dengue backfill months missing after merge: "
            f"{missing_backfill[:3]}"
        )

    previous_dates = {
        row["target_end_date"]
        for row in previous
        if row.get("target") == TARGET_DENGUE
    }
    dropped = sorted(previous_dates - have)
    if dropped:
        raise RuntimeError(
            f"refusing to drop {len(dropped)} existing dengue months, "
            f"first {dropped[:3]}"
        )


def month_diff(later: str, earlier: str) -> int:
    a = datetime.strptime(later, "%Y-%m-%d")
    b = datetime.strptime(earlier, "%Y-%m-%d")
    return (a.year - b.year) * 12 + (a.month - b.month)


def add_months(date: str, n: int) -> str:
    dt = datetime.strptime(date, "%Y-%m-%d")
    month = dt.month - 1 + n
    year = dt.year + month // 12
    month = month % 12 + 1
    return f"{year:04d}-{month:02d}-01"


def month_starts(first: str, last: str) -> list[str]:
    dates = []
    current = first
    while current <= last:
        dates.append(current)
        current = add_months(current, 1)
    return dates


def build_oracle(time_series: list[dict[str, str]], target: str,
                 monthly: bool) -> list[dict[str, str]]:
    rows = [row for row in time_series if row["target"] == target]
    dates = sorted({row["target_end_date"] for row in rows})
    values = {row["target_end_date"]: row["observation"] for row in rows}
    oracle_rows: list[dict[str, str]] = []

    for ref in dates:
        available = [date for date in dates if date <= ref][-MAX_ORACLE_LOOKBACK:]
        for target_date in available:
            if monthly:
                horizon = month_diff(target_date, ref)
            else:
                ref_dt = datetime.strptime(ref, "%Y-%m-%d")
                target_dt = datetime.strptime(target_date, "%Y-%m-%d")
                horizon = (target_dt - ref_dt).days // 7
            oracle_rows.append(
                {
                    "reference_date": ref,
                    "target": target,
                    "horizon": str(horizon),
                    "target_end_date": target_date,
                    "location": LOCATION,
                    "output_type": "quantile",
                    "output_type_id": "0.5",
                    "oracle_value": values[target_date],
                }
            )

    return oracle_rows


def flu_target_metadata() -> dict[str, object]:
    return {
        "target_id": TARGET_FLU,
        "target_name": "Influenza virus positivity rate among department "
                       "influenza-like illness (ILI) cases",
        "target_units": "proportion",
        "target_keys": {"target": TARGET_FLU},
        "description": "This target represents the influenza virus positivity "
                       "rate among department influenza-like illness (ILI) "
                       "cases, as reported in China CDCs weekly National "
                       "Sentinel Surveillance of Acute Respiratory Infectious "
                       "Diseases, in the week ending on the target_end_date "
                       "(i.e., [horizon] weeks after the reference_date).",
        "target_type": "continuous",
        "is_step_ahead": True,
        "time_unit": "week",
    }


def dengue_target_metadata() -> dict[str, object]:
    return {
        "target_id": TARGET_DENGUE,
        "target_name": "Monthly reported dengue cases",
        "target_units": "count",
        "target_keys": {"target": TARGET_DENGUE},
        "description": "This target represents the number of reported dengue "
                       "cases in China, as published in China CDCs monthly "
                       "notifiable infectious disease surveillance report, in "
                       "the calendar month starting on the target_end_date "
                       "(i.e., [horizon] months after the reference_date).",
        "target_type": "discrete",
        "is_step_ahead": True,
        "time_unit": "month",
    }


def make_quantile_output_type(value_type: str) -> dict[str, object]:
    return {
        "quantile": {
            "output_type_id": {"required": QUANTILE_LEVELS},
            "is_required": False,
            "value": {"type": value_type, "minimum": 0},
        },
        "sample": {
            "output_type_id_params": {
                "type": "character",
                "max_length": 15,
                "min_samples_per_task": 200,
                "max_samples_per_task": 200,
                "compound_taskid_set": ["location"],
            },
            "is_required": False,
            "value": {"type": value_type, "minimum": 0},
        },
    }


def make_dengue_model_task() -> dict[str, object]:
    """Dengue model_tasks entry (monthly cadence) for the shared round."""
    reference_dates = month_starts(DENGUE_FIRST_REFERENCE, DENGUE_LAST_REFERENCE)
    first_end = add_months(DENGUE_FIRST_REFERENCE, min(DENGUE_HORIZONS))
    last_end = add_months(DENGUE_LAST_REFERENCE, max(DENGUE_HORIZONS))
    return {
        "task_ids": {
            "reference_date": {"required": None, "optional": reference_dates},
            "location": {"required": None, "optional": [LOCATION]},
            "horizon": {"required": None, "optional": DENGUE_HORIZONS},
            "target_end_date": {
                "required": None,
                "optional": month_starts(first_end, last_end),
            },
            "target": {"required": None, "optional": [TARGET_DENGUE]},
        },
        "output_type": make_quantile_output_type("integer"),
        "target_metadata": [dengue_target_metadata()],
    }


def merge_dates(task_ids: dict[str, object], dates: list[str]) -> bool:
    changed = False
    for key in ("reference_date", "target_end_date"):
        if merge_optional_dates(task_ids, key, dates):
            changed = True
    return changed


def merge_optional_dates(task_ids: dict[str, object], key: str,
                         dates: list[str]) -> bool:
    current = task_ids[key]["optional"]
    merged = sorted(set(current) | set(dates))
    if merged != current:
        task_ids[key]["optional"] = merged
        return True
    return False


def update_tasks_json(weekly_dates: list[str],
                      monthly_dates: list[str],
                      monthly_target_end_dates: list[str] | None = None) -> bool:
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    rounds = tasks["rounds"]
    changed = False

    # Round 1 (weekly): covid + flu share task ids.
    weekly_task = rounds[0]["model_tasks"][0]
    if merge_dates(weekly_task["task_ids"], weekly_dates):
        changed = True
    targets = weekly_task["task_ids"]["target"]["optional"]
    if TARGET_FLU not in targets:
        targets.append(TARGET_FLU)
        changed = True
    metadata_ids = {m["target_id"] for m in weekly_task["target_metadata"]}
    if TARGET_FLU not in metadata_ids:
        weekly_task["target_metadata"].append(flu_target_metadata())
        changed = True

    # Single round holds all targets: dengue (monthly) is a separate
    # model_tasks block inside round 0 so dashboard tools keyed to
    # rounds_idx 0 (e.g. predtimechart) can see every target.
    if len(rounds) > 1:
        del rounds[1:]
        changed = True
    model_tasks = rounds[0]["model_tasks"]
    dengue_task = None
    for task in model_tasks:
        if TARGET_DENGUE in (task["task_ids"]["target"]["optional"] or []):
            dengue_task = task
            break
    if dengue_task is None:
        dengue_task = make_dengue_model_task()
        model_tasks.append(dengue_task)
        changed = True
    if merge_dates(dengue_task["task_ids"], monthly_dates):
        changed = True
    # hubValidations target_tbl_values requires every observed dengue
    # target_end_date, including 2004–2023 history, to be listed here.
    # Those months are not submission rounds: leave reference_date alone
    # so they are not unioned into the weekly round ids.
    if monthly_target_end_dates and merge_optional_dates(
        dengue_task["task_ids"], "target_end_date", monthly_target_end_dates
    ):
        changed = True

    # hubAdmin's validate_round_ids_consistent requires the round_id variable
    # task_id to be identical across model_tasks when round_id_from_variable
    # is true, so keep the reference_date optional values unified.
    union = sorted(
        set(weekly_task["task_ids"]["reference_date"]["optional"])
        | set(dengue_task["task_ids"]["reference_date"]["optional"])
    )
    for task in (weekly_task, dengue_task):
        if task["task_ids"]["reference_date"]["optional"] != union:
            task["task_ids"]["reference_date"]["optional"] = union
            changed = True

    if changed:
        TASKS_PATH.write_text(
            json.dumps(tasks, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return changed


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_hub_target_data(args: argparse.Namespace) -> dict[str, object]:
    """Regenerate Hub target-data (+ tasks.json dates) from cncdc CSVs."""
    covid_rows, covid_src = local_or_github(
        "cncdc_surveillance_covid19.csv", COVID_URL, args.csv, args.from_github)
    surv_rows, surv_src = local_or_github(
        "cncdc_surveillance_all.csv", SURV_ALL_URL, args.surv_csv, args.from_github)
    notifiable_rows, notifiable_src = local_or_github(
        "notifiable_all.csv", NOTIFIABLE_URL, args.notifiable_csv, args.from_github)
    backfill_rows, backfill_src = local_or_github(
        "backfill/notifiable-flu-dengue.csv", BACKFILL_URL,
        args.backfill_csv, args.from_github)

    covid_series = build_weekly_series(
        covid_rows, "新型冠状病毒", TARGET_COVID, MIN_DATE_COVID)
    flu_series = build_weekly_series(
        surv_rows, "流感病毒", TARGET_FLU, None)
    notifiable_rows, dengue_backfill_added = merge_dengue_backfill(
        notifiable_rows, backfill_rows)
    dengue_series = build_dengue_series(notifiable_rows)
    if not covid_series or not flu_series or not dengue_series:
        raise RuntimeError("No time-series rows generated from cncdc CSVs")
    previous = []
    if TIME_SERIES_PATH.exists():
        with TIME_SERIES_PATH.open(newline="", encoding="utf-8") as handle:
            previous = list(csv.DictReader(handle))
    assert_dengue_history(dengue_series, backfill_rows, previous)
    ignored_monthly_flu = sum(
        1 for row in backfill_rows
        if row.get("disease") == MONTHLY_FLU_DISEASE and row.get("row_kind") == "disease"
    )

    time_series = covid_series + flu_series + dengue_series
    oracle = (
        build_oracle(time_series, TARGET_COVID, monthly=False)
        + build_oracle(time_series, TARGET_FLU, monthly=False)
        + build_oracle(time_series, TARGET_DENGUE, monthly=True)
    )
    write_csv(
        TIME_SERIES_PATH,
        time_series,
        ["target_end_date", "target", "location", "observation"],
    )
    write_csv(
        ORACLE_PATH,
        oracle,
        [
            "reference_date",
            "target",
            "horizon",
            "target_end_date",
            "location",
            "output_type",
            "output_type_id",
            "oracle_value",
        ],
    )
    weekly_dates = sorted(
        {row["target_end_date"] for row in covid_series + flu_series})
    monthly_dates = sorted({row["target_end_date"] for row in dengue_series})
    # Months before the configured dengue submission window are observed
    # history, not new forecast rounds. They are valid target_end_dates
    # (see monthly_target_end_dates) but must not be merged into
    # reference_date, which is unioned across the weekly task as well.
    task_monthly_dates = [
        date for date in monthly_dates if date >= DENGUE_FIRST_REFERENCE
    ]
    tasks_changed = update_tasks_json(
        weekly_dates,
        task_monthly_dates,
        monthly_target_end_dates=monthly_dates,
    )

    def span(series: list[dict[str, str]]) -> str:
        return (f"{series[-1]['target_end_date']} .. "
                f"{series[0]['target_end_date']} ({len(series)} rows)")

    return {
        "covid": span(covid_series),
        "flu": span(flu_series),
        "dengue": span(dengue_series),
        "dengue_backfill_added": dengue_backfill_added,
        "ignored_monthly_flu": ignored_monthly_flu,
        "oracle_rows": len(oracle),
        "tasks_changed": tasks_changed,
        "sources": [covid_src, surv_src, notifiable_src, backfill_src],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Local cncdc_surveillance_covid19.csv (default: local path if present, else GitHub)",
    )
    parser.add_argument(
        "--surv-csv",
        type=Path,
        default=None,
        help="Local cncdc_surveillance_all.csv (default: local path if present, else GitHub)",
    )
    parser.add_argument(
        "--notifiable-csv",
        type=Path,
        default=None,
        help="Local notifiable_all.csv (default: local path if present, else GitHub)",
    )
    parser.add_argument(
        "--backfill-csv",
        type=Path,
        default=None,
        help="Local notifiable-flu-dengue.csv backfill "
             "(default: local path if present, else GitHub)",
    )
    parser.add_argument(
        "--from-github",
        action="store_true",
        help="Force fetch from GitHub raw instead of local CSVs",
    )
    args = parser.parse_args()

    stats = update_hub_target_data(args)
    print(f"Updated {TIME_SERIES_PATH.name}:")
    print(f"  covid:  {stats['covid']}")
    print(f"  flu:    {stats['flu']}")
    print(f"  dengue: {stats['dengue']}")
    print(f"  dengue backfill months added: {stats['dengue_backfill_added']}")
    print(f"  monthly flu backfill rows ignored: {stats['ignored_monthly_flu']}")
    print(f"Updated {ORACLE_PATH.name}: {stats['oracle_rows']} rows")
    print(f"Updated tasks.json: {'yes' if stats['tasks_changed'] else 'no'}")
    for src in stats["sources"]:
        print(f"Source: {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
