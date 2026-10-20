"""
Forecast output CONTRACT v0 (Component 1 -> Component 2/3/4) and a first producer for it.

What this is, in plain words
  Component 1 promises to hand the other components ONE file with a fixed shape. This module
  (a) writes that file, (b) checks it against the contract (the validator), and (c) reads it back
  the way a consuming component would. Today the numbers come from a simple baseline, so every row
  says status = placeholder_baseline. The SHAPE is what Components 2-4 can build against now; the
  numbers improve later without the shape changing (bump contract_version if the shape changes).

Contract v0 (one CSV, one row per level x district x month)
  contract_version   '0.1'
  run_id             e.g. 'naive_ratio_v0@2026-08'  (method @ last observed month)
  level              'national' | 'district'
  district           'ALL' for national, otherwise the DMT district name (25 districts, UPPER CASE)
  vehicle_class      'all'  (v0: all EV classes together)
  period             'YYYY-MM' month being forecast
  horizon_months     1, 2, 3 ... months after the last observed month
  p10, p50, p90      forecast of NEW EV registrations in that month (80% interval; p10 <= p50 <= p90)
  method             'naive_ratio_v0'
  status             'placeholder_baseline'  (v0: not the adaptive framework yet)
  Rules the validator enforces: all columns present; no blanks; counts >= 0; p10 <= p50 <= p90;
  periods are consecutive months; 'national' has exactly one district 'ALL'; 'district' rows cover
  the same 25 districts in every month; district p50 values sum to the national p50.

How the v0 numbers are made (honest and simple)
  * National: last observed month x the empirical ratio (actual / last value) seen at that horizon
    in the walk-forward history (targets from 2023-01, when monthly counts became large enough for
    ratios to mean something). p10 / p50 / p90 are the 10th / 50th / 90th percentiles of that ratio.
    Only data up to the last observed month is used. Interval coverage has NOT been validated yet.
  * District: the national numbers split by each district's share of the last FULL year (2025) of EV
    registrations. This is a static placeholder for the spatial disaggregation, not the real method.
    District bounds are proportional shares of the national bounds (they do not combine statistically).

Run from the repo root (private DMT CSVs in data/raw):
  python -m src.models.forecast_v0 --raw-dir data/raw
"""
import argparse
import csv
import glob
import math
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION = "0.1"
METHOD = "naive_ratio_v0"
STATUS = "placeholder_baseline"
COLUMNS = ["contract_version", "run_id", "level", "district", "vehicle_class", "period",
           "horizon_months", "p10", "p50", "p90", "method", "status"]
RATIO_FROM = "2023-01"          # ratios are only learned from targets in the high-volume era
STEPS = 4                       # forecast the next 4 months
Q_LOW, Q_MID, Q_HIGH = 0.10, 0.50, 0.90

Row = dict[str, Any]

# ---------------------------------------------------------------- helpers on plain month strings
def month_add(period: str, k: int) -> str:
    y, m = int(period[:4]), int(period[5:7])
    t = y * 12 + (m - 1) + k
    return f"{t // 12:04d}-{t % 12 + 1:02d}"

# ---------------------------------------------------------------- the national forecast
def empirical_ratio_quantiles(values: list[float], periods: list[str], observed: list[bool],
                              steps: int = STEPS, ratio_from: str = RATIO_FROM) -> dict[int, tuple[float, float, float]]:
    """For each horizon h: quantiles (p10, p50, p90) of actual[t+h] / actual[t], using only pairs fully inside the history."""
    out: dict[int, tuple[float, float, float]] = {}
    n = len(values)
    for h in range(1, steps + 1):
        ratios: list[float] = []
        for o in range(0, n - h):
            t = o + h
            if periods[t] >= ratio_from and observed[t] and observed[o] and values[o] > 0 and values[t] > 0:
                ratios.append(math.log(values[t] / values[o]))
        if len(ratios) < 8:
            raise ValueError(f"Only {len(ratios)} usable ratios at horizon {h}; need at least 8.")
        q = np.quantile(ratios, [Q_LOW, Q_MID, Q_HIGH])
        out[h] = (float(np.exp(q[0])), float(np.exp(q[1])), float(np.exp(q[2])))
    return out

def national_forecast(values: list[float], periods: list[str], observed: list[bool], steps: int = STEPS) -> list[Row]:
    """Rows for the next `steps` months after the last period. Uses only the data passed in."""
    last = values[-1]
    qs = empirical_ratio_quantiles(values, periods, observed, steps)
    rows: list[Row] = []
    for h in range(1, steps + 1):
        lo, mid, hi = qs[h]
        rows.append({"horizon_months": h, "period": month_add(periods[-1], h),
                     "p10": last * lo, "p50": last * mid, "p90": last * hi})
    return rows

# ---------------------------------------------------------------- district split + contract rows
def district_shares(totals: dict[str, float]) -> dict[str, float]:
    s = sum(totals.values())
    if s <= 0:
        raise ValueError("District totals must sum to a positive number.")
    return {d: v / s for d, v in totals.items()}

def build_contract_rows(national: list[dict[str, Any]], shares: dict[str, float], last_period: str) -> list[Row]:
    run_id = f"{METHOD}@{last_period}"
    base = {"contract_version": CONTRACT_VERSION, "run_id": run_id, "vehicle_class": "all", "method": METHOD, "status": STATUS}
    rows: list[Row] = []
    for r in national:
        rows.append({**base, "level": "national", "district": "ALL", "period": r["period"],
                     "horizon_months": r["horizon_months"], "p10": r["p10"], "p50": r["p50"], "p90": r["p90"]})
        for d in sorted(shares):
            rows.append({**base, "level": "district", "district": d, "period": r["period"],
                         "horizon_months": r["horizon_months"],
                         "p10": r["p10"] * shares[d], "p50": r["p50"] * shares[d], "p90": r["p90"] * shares[d]})
    return rows

# ---------------------------------------------------------------- the contract validator (what a consumer calls)
def validate_contract(rows: list[Row]) -> list[str]:
    """Return a list of problems (empty list = the file honours contract v0)."""
    problems: list[str] = []
    if not rows:
        return ["file has no rows"]
    for c in COLUMNS:
        if any(c not in r for r in rows):
            problems.append(f"missing column: {c}")
    if problems:
        return problems
    if {r["contract_version"] for r in rows} != {CONTRACT_VERSION}:
        problems.append(f"contract_version must be {CONTRACT_VERSION}")
    for i, r in enumerate(rows):
        for c in COLUMNS:
            v = r[c]
            if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
                problems.append(f"row {i}: blank value in {c}")
        try:
            p10, p50, p90 = float(r["p10"]), float(r["p50"]), float(r["p90"])
        except (TypeError, ValueError):
            problems.append(f"row {i}: p10/p50/p90 must be numbers")
            continue
        if min(p10, p50, p90) < 0:
            problems.append(f"row {i}: negative count")
        if not (p10 <= p50 + 1e-9 and p50 <= p90 + 1e-9):
            problems.append(f"row {i}: need p10 <= p50 <= p90")
        if r["level"] not in ("national", "district"):
            problems.append(f"row {i}: level must be national or district")
    if problems:
        return problems
    nat = [r for r in rows if r["level"] == "national"]
    dis = [r for r in rows if r["level"] == "district"]
    if any(r["district"] != "ALL" for r in nat):
        problems.append("national rows must have district = ALL")
    periods = sorted({r["period"] for r in nat})
    if not periods or periods != [month_add(periods[0], k) for k in range(len(periods))]:
        problems.append("national periods must be consecutive months")
    if len(nat) != len(periods):
        problems.append("national must have exactly one row per period")
    names = {r["district"] for r in dis}
    if dis and len(names) != 25:
        problems.append(f"district rows must cover 25 districts, found {len(names)}")
    for p in periods:
        in_p = [r for r in dis if r["period"] == p]
        if in_p:
            if {r["district"] for r in in_p} != names:
                problems.append(f"{p}: district list differs from other months")
            nat_p50 = float([r for r in nat if r["period"] == p][0]["p50"])
            if abs(sum(float(r["p50"]) for r in in_p) - nat_p50) > 1e-6 * max(1.0, nat_p50):
                problems.append(f"{p}: district p50 values do not sum to the national p50")
    return problems

def write_contract(rows: list[Row], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.3f}" if c in ("p10", "p50", "p90") else r[c]) for c in COLUMNS})

def read_contract(path: str) -> list[Row]:
    """What a CONSUMING component calls: reads the file and refuses it if it breaks the contract."""
    with open(path, newline="", encoding="utf-8") as f:
        rows: list[Row] = [dict(r) for r in csv.DictReader(f)]
    for r in rows:
        for c in ("p10", "p50", "p90"):
            r[c] = float(r[c])
        r["horizon_months"] = int(r["horizon_months"])
    problems = validate_contract(rows)
    if problems:
        raise ValueError("Forecast file breaks contract v0:\n  " + "\n  ".join(problems[:10]))
    return rows

# ---------------------------------------------------------------- loading DMT inputs (plain csv, no pandas)
def _one(raw_dir: str, pattern: str) -> list[dict[str, str]]:
    hits = glob.glob(str(Path(raw_dir) / pattern))
    if len(hits) != 1:
        raise SystemExit(f"Expected one {pattern} in {raw_dir}, found {len(hits)}")
    with open(hits[0], newline="", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]

def _num(text: str | None) -> float:
    try:
        return float(str(text).strip())
    except ValueError:
        return float("nan")                              # blank, '.', or anything non-numeric = not observed

def load_national_monthly(raw_dir: str, start: str = "2019-01") -> tuple[list[float], list[str], list[bool]]:
    rows = [r for r in _one(raw_dir, "*02_*.csv") if str(r["Period"]).strip() >= start]
    periods = [str(r["Period"]).strip() for r in rows]
    raw = [_num(r["New_EV_registrations"]) for r in rows]
    observed = [not math.isnan(v) for v in raw]
    values = list(raw)
    for i, v in enumerate(values):                       # blank months: carry the previous value (INPUT only, never scored)
        if math.isnan(v):
            values[i] = values[i - 1] if i > 0 else next(x for x in raw if not math.isnan(x))
    return values, periods, observed

def load_district_totals(raw_dir: str, year: int = 2025) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in _one(raw_dir, "*01_*.csv"):
        if int(float(r["Year"])) == year:
            out[str(r["District"]).strip().upper()] = _num(r["Total"])
    return out

# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/forecast_v0.csv")
    a = ap.parse_args()
    values, periods, observed = load_national_monthly(a.raw_dir)
    nat = national_forecast(values, periods, observed)
    rows = build_contract_rows(nat, district_shares(load_district_totals(a.raw_dir)), periods[-1])
    problems = validate_contract(rows)
    if problems:
        raise SystemExit("Producer broke its own contract:\n  " + "\n  ".join(problems))
    write_contract(rows, a.out)
    back = read_contract(a.out)                          # round trip: read it the way Component 2 would
    print(f"Wrote {a.out}: {len(back)} rows, contract v{CONTRACT_VERSION}, last observed month {periods[-1]} ({values[-1]:,.0f} EVs).")
    print("National forecast (new EV registrations per month; 80% interval):")
    for r in back:
        if r["level"] == "national":
            print(f"  {r['period']}  h={r['horizon_months']}  p10 {r['p10']:>8,.0f}   p50 {r['p50']:>8,.0f}   p90 {r['p90']:>8,.0f}")
    top = sorted((r for r in back if r["level"] == "district" and r["horizon_months"] == 1), key=lambda r: -r["p50"])[:3]
    print("Top 3 districts, next month (p50): " + ", ".join(f"{r['district']} {r['p50']:,.0f}" for r in top))
    print("Status:", STATUS, "- numbers are a baseline placeholder; interval coverage not yet validated.")

if __name__ == "__main__":
    main()