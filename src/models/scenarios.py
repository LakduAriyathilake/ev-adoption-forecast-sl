"""
Scenario engine v0 (Scenario agent): annual EV scenarios for 2027-2037, national + 25 districts.

Plain-language idea
  The question is not "what WILL happen in 2037" (nobody can know that from ~2 years of high-volume data).
  The question is "IF Sri Lanka's EV share of new registrations climbs at the pace other countries' shares
  climbed from a similar starting point, what would that mean for new EV registrations each year?"
  Three paces are shown: slow, baseline, accelerated.

How it is calculated (all numbers come from data, nothing typed in by hand)
  1. Start: Sri Lanka's EV share of ALL new registrations in the latest year (2026, Jan-Aug, DMT file 03).
  2. Pace: look at every country-year in the OWID file (aggregates such as 'Europe' removed) where the EV share of
     car sales was at a similar stage (logit within 0.75 of Sri Lanka's current logit; widened if fewer than 20
     cases). Take their next-year logit step: slow = 25th percentile, baseline = median, accelerated = 75th.
     The same quantile is used every year (a persistent pace), which is a strong assumption. The step is recomputed
     each year from the new share, so the pace slows as the share rises, the way it did in the data.
     (Finding: the 9 regional peers alone cannot do this. They have too few cases above ~20% share, and a straight-line
     rule fitted to them never slows down: it reached ~99% EV share by 2037. Hence the wider analogue set.)
  3. Volume: new EVs = share x total new registrations that year. Total new registrations are held flat at the
     2026 pace (Jan-Aug x 12/8). This is an explicit assumption; --new-vehicles overrides it.
  4. Districts: national numbers split by each district's share of 2025 EV registrations (static placeholder).

Known limits (say these out loud in the viva)
  * Peer data are CARS (BEV+PHEV share of car sales); Sri Lanka's EVs are about 80% motorcycles. The peer pace is a
    stand-in for the SHAPE of an S-curve, not proof for two-wheelers.
  * Total new registrations after the import ban are unsettled, and 2026 may include pent-up demand. If the market
    stays at the 2025 size (362,100) instead, every EV count scales by about 0.54 (shares do not change).
  * Seasonality of 2026 is unknown, so Jan-Aug x 12/8 is an approximation.
  * These are 'if' paths, not predictions. status = placeholder_scenario.

Run from the repo root:
  python -m src.models.scenarios --raw-dir data/raw
"""
import argparse
import csv
import glob
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.models.forecast_v0 import district_shares, load_district_totals
from src.models.transfer import load_peer_transitions, logit_pct, inv_logit_pct

SCENARIO_VERSION = "0.1"
METHOD = "stage_pace_logit_v0"
STATUS = "placeholder_scenario"
SCENARIOS = ["slow", "baseline", "accelerated"]
Q = {"slow": 0.25, "baseline": None, "accelerated": 0.75}
FIRST_YEAR, LAST_YEAR = 2027, 2037
COLUMNS = ["scenario_version", "run_id", "level", "district", "scenario", "year",
           "national_ev_share_pct", "new_evs", "cum_new_evs_from_2027", "method", "status"]

Row = dict[str, Any]

AGGREGATES = {"Europe", "European Union (27)", "Rest of World", "World"}

def all_countries(owid_share_csv: str) -> list[str]:
    """Every country in the OWID file, with regional aggregates removed (they would double-count their members)."""
    with open(owid_share_csv, newline="", encoding="utf-8-sig") as f:
        return sorted({r["entity"] for r in csv.DictReader(f)} - AGGREGATES)

# ---------------------------------------------------------------- anchor from DMT (plain csv)
def load_anchor(raw_dir: str) -> tuple[int, int, float, float]:
    """Latest year in DMT file 03: (year, months_present, total_new_registrations, electric_new_registrations)."""
    hits = glob.glob(str(Path(raw_dir) / "*03_*.csv"))
    if len(hits) != 1:
        raise SystemExit(f"Expected one *03_*.csv in {raw_dir}, found {len(hits)}")
    with open(hits[0], newline="", encoding="utf-8-sig") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    year = max(int(r["Year"]) for r in rows)
    months: set[str] = set()
    total = ev = 0.0
    for r in rows:
        if int(r["Year"]) != year:
            continue
        try:
            c = float(r["Count"])
        except ValueError:
            continue
        if r["Vehicle_class"] == "TOTAL (all classes)":
            if r["Fuel"] == "All":
                total += c
                months.add(r["Month"])
        elif r["Fuel"] == "Electric":
            ev += c
    if total <= 0 or not months:
        raise SystemExit("Could not read the anchor year from file 03.")
    return year, len(months), total, ev

# ---------------------------------------------------------------- the pace (stage-matched)
WINDOW, MIN_N, WIDEN = 0.75, 20, 0.25

def stage_pace(trans: list[Row], z: float) -> dict[str, float]:
    """Slow / baseline / accelerated logit step for a share whose logit is z, from analogue country-years at a similar stage."""
    w = WINDOW
    while True:
        dz = [r["dz"] for r in trans if abs(r["z_now"] - z) <= w]
        if len(dz) >= MIN_N or w > 6:
            break
        w += WIDEN
    if len(dz) < MIN_N:
        raise ValueError("Not enough analogue country-years to set a pace.")
    q = np.quantile(dz, [0.25, 0.5, 0.75])
    return {"slow": float(q[0]), "baseline": float(q[1]), "accelerated": float(q[2]), "n": float(len(dz))}

def share_path(share_pct_now: float, trans: list[Row], scenario: str, steps: int) -> list[float]:
    """EV share (%) for each of the next `steps` years. Uses only the analogue transitions passed in."""
    z = float(logit_pct(share_pct_now))
    out: list[float] = []
    for _ in range(steps):
        z = z + stage_pace(trans, z)[scenario]
        out.append(float(inv_logit_pct(z)))
    return out

# ---------------------------------------------------------------- build rows
def build_rows(anchor_share_pct: float, anchor_year: int, new_vehicles: float, trans: list[Row],
               shares: dict[str, float]) -> list[Row]:
    steps = LAST_YEAR - anchor_year
    run_id = f"{METHOD}@{anchor_year}"
    base = {"scenario_version": SCENARIO_VERSION, "run_id": run_id, "method": METHOD, "status": STATUS}
    rows: list[Row] = []
    for sc in SCENARIOS:
        path = share_path(anchor_share_pct, trans, sc, steps)
        years = list(range(anchor_year + 1, LAST_YEAR + 1))
        cum = {"ALL": 0.0, **{d: 0.0 for d in shares}}
        for y, s in zip(years, path):
            if y < FIRST_YEAR:
                continue
            nat = new_vehicles * s / 100.0
            cum["ALL"] += nat
            rows.append({**base, "level": "national", "district": "ALL", "scenario": sc, "year": y,
                         "national_ev_share_pct": s, "new_evs": nat, "cum_new_evs_from_2027": cum["ALL"]})
            for d in sorted(shares):
                v = nat * shares[d]
                cum[d] += v
                rows.append({**base, "level": "district", "district": d, "scenario": sc, "year": y,
                             "national_ev_share_pct": s, "new_evs": v, "cum_new_evs_from_2027": cum[d]})
    return rows

# ---------------------------------------------------------------- validator
def validate_scenarios(rows: list[Row]) -> list[str]:
    problems: list[str] = []
    if not rows:
        return ["file has no rows"]
    for c in COLUMNS:
        if any(c not in r for r in rows):
            problems.append(f"missing column: {c}")
    if problems:
        return problems
    for i, r in enumerate(rows):
        for c in COLUMNS:
            v = r[c]
            if v is None or str(v).strip() == "" or (isinstance(v, float) and math.isnan(v)):
                problems.append(f"row {i}: blank value in {c}")
        try:
            if float(r["new_evs"]) < 0 or float(r["cum_new_evs_from_2027"]) < 0:
                problems.append(f"row {i}: negative count")
            if not (0 < float(r["national_ev_share_pct"]) < 100):
                problems.append(f"row {i}: share must be between 0 and 100")
        except (TypeError, ValueError):
            problems.append(f"row {i}: numeric column is not a number")
    if problems:
        return problems
    if {r["scenario"] for r in rows} != set(SCENARIOS):
        problems.append(f"scenarios must be exactly {SCENARIOS}")
    nat = [r for r in rows if r["level"] == "national"]
    dis = [r for r in rows if r["level"] == "district"]
    years = sorted({int(r["year"]) for r in nat})
    if years != list(range(FIRST_YEAR, LAST_YEAR + 1)):
        problems.append(f"years must run {FIRST_YEAR}-{LAST_YEAR} without gaps")
    if len(dis) and len({r["district"] for r in dis}) != 25:
        problems.append("district rows must cover 25 districts")
    for y in years:
        by = {r["scenario"]: float(r["new_evs"]) for r in nat if int(r["year"]) == y}
        if len(by) == 3 and not (by["slow"] <= by["baseline"] + 1e-9 <= by["accelerated"] + 2e-9):
            problems.append(f"{y}: need slow <= baseline <= accelerated")
        for sc in SCENARIOS:
            n = [float(r["new_evs"]) for r in nat if int(r["year"]) == y and r["scenario"] == sc]
            d = [float(r["new_evs"]) for r in dis if int(r["year"]) == y and r["scenario"] == sc]
            if n and d and abs(sum(d) - n[0]) > max(0.05, 1e-6 * n[0]):
                problems.append(f"{y} {sc}: district values do not sum to the national value")
    return problems

def write_scenarios(rows: list[Row], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fmt = {"national_ev_share_pct": "{:.3f}", "new_evs": "{:.3f}", "cum_new_evs_from_2027": "{:.3f}"}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: (fmt[c].format(r[c]) if c in fmt else r[c]) for c in COLUMNS})

def read_scenarios(path: str) -> list[Row]:
    with open(path, newline="", encoding="utf-8") as f:
        rows: list[Row] = [dict(r) for r in csv.DictReader(f)]
    for r in rows:
        r["year"] = int(r["year"])
        for c in ("national_ev_share_pct", "new_evs", "cum_new_evs_from_2027"):
            r[c] = float(r[c])
    problems = validate_scenarios(rows)
    if problems:
        raise ValueError("Scenario file breaks the contract:\n  " + "\n  ".join(problems[:10]))
    return rows

# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--peers-csv", default="data/external/owid_electric_car_sales_share.csv")
    ap.add_argument("--new-vehicles", type=float, default=None, help="assumed total new registrations per year (default: 2026 pace)")
    ap.add_argument("--out", default="data/processed/scenarios_v0.csv")
    a_ = ap.parse_args()

    year, months, total, ev = load_anchor(a_.raw_dir)
    share = 100.0 * ev / total
    n_year = a_.new_vehicles if a_.new_vehicles else total * 12.0 / months
    trans = load_peer_transitions(a_.peers_csv, all_countries(a_.peers_csv))
    rows = build_rows(share, year, n_year, trans, district_shares(load_district_totals(a_.raw_dir)))
    problems = validate_scenarios(rows)
    if problems:
        raise SystemExit("Producer broke its own contract:\n  " + "\n  ".join(problems))
    write_scenarios(rows, a_.out)
    back = read_scenarios(a_.out)

    print(f"Anchor: {year} ({months} months) EV share of all new registrations = {share:.1f}% ({ev:,.0f} of {total:,.0f}).")
    print(f"Assumed total new registrations per year: {n_year:,.0f}" + ("" if a_.new_vehicles else f" (= {year} Jan-Aug x 12/{months}; 2025 full year was 362,100)."))
    p0 = stage_pace(trans, float(logit_pct(share)))
    print(f"Analogues: all countries ({len(trans)} country-years). At today's share, {p0['n']:.0f} similar-stage cases give a yearly logit step of "
          f"slow {p0['slow']:+.2f} / baseline {p0['baseline']:+.2f} / accelerated {p0['accelerated']:+.2f}.")
    print(f"Wrote {a_.out}: {len(back)} rows ({LAST_YEAR - FIRST_YEAR + 1} years x 3 scenarios x (1 national + 25 districts)).")
    print("National new EVs per year (EV share in brackets):")
    print("  year   slow                 baseline             accelerated")
    for y in (2027, 2030, 2033, 2035, 2037):
        cells = []
        for sc in SCENARIOS:
            r = [x for x in back if x["level"] == "national" and x["scenario"] == sc and x["year"] == y][0]
            cells.append(f"{r['new_evs']:>9,.0f} ({r['national_ev_share_pct']:>4.1f}%)")
        print(f"  {y}  " + "   ".join(cells))
    print("Status:", STATUS, "- if-paths, not predictions.")

if __name__ == "__main__":
    main()