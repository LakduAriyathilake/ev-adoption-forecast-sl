"""
Spatial step v0 (Spatial agent): district scenario -> DS division ("town") x vehicle class.

Plain-language idea
  A district forecast says 'Colombo district: 31,086 new EVs in 2030'. This step answers 'how many of those are
  cars, how many are bikes, and how many are in Kaduwela?' using only things we can point to:
    EVs in (DS, class, year, scenario) = district scenario total
                                         x district's share of EVs that are this class   (DMT, 2025 + 2026 Jan-Aug pooled)
                                         x the DS division's share of the district's population (Census 2024)

  Why the class mix is per district: Colombo's EVs are about half cars, Kurunegala's about 4% cars. One national
  mix would be badly wrong. A district with fewer than MIN_POOL EVs in the pool (too noisy) uses the national mix and
  is flagged mix_source = national_fallback.

Honest limits (say them in the viva)
  * Inside a district the weight is POPULATION ONLY. Wealthier, more urban DS divisions probably hold more cars than
    their population share; this version cannot see that. There is no EV data below district to test it against.
  * DMT records where a vehicle was REGISTERED, not where the owner lives.
  * The class mix is held at its 2025-26 level for all years (the real mix will drift).
  * The district-level check printed by this script asks: if national EVs were shared out by population alone, how far
    off would each district be? It is evidence about the weighting idea, not a validation of town numbers.
  status = placeholder_estimate.

Run from the repo root:
  python -m src.models.town_split --raw-dir data/raw
"""
import argparse
import csv
import glob
import math
from pathlib import Path
from typing import Any

from src.data.census_ds import read_ds
from src.models.scenarios import read_scenarios

ESTIMATE_VERSION = "0.1"
METHOD = "district_x_mix_x_pop_share_v0"
STATUS = "placeholder_estimate"
CLASSES = ["car", "motorcycle", "three_wheeler", "other"]
POOL_YEARS = ("2025", "2026")
MIN_POOL = 300
COLUMNS = ["estimate_version", "run_id", "district", "ds_name", "ds_code", "scenario", "year", "vehicle_class",
           "new_evs", "mix_source", "method", "status"]

Row = dict[str, Any]

# ---------------------------------------------------------------- DMT class mix
def _f(x: str | None) -> float:
    try:
        return float(str(x).strip())
    except ValueError:
        return 0.0

def load_pooled_counts(raw_dir: str) -> dict[str, dict[str, float]]:
    """district -> {car, motorcycle, three_wheeler, other}: EV registrations pooled over POOL_YEARS (DMT file 01)."""
    hits = glob.glob(str(Path(raw_dir) / "*01_*.csv"))
    if len(hits) != 1:
        raise SystemExit(f"Expected one *01_*.csv in {raw_dir}, found {len(hits)}")
    out: dict[str, dict[str, float]] = {}
    with open(hits[0], newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["Year"] not in POOL_YEARS:
                continue
            car, bike = _f(r["Car"]), _f(r["Motor Cycle"])
            three = _f(r["Three Wheeler"]) + _f(r["Three Wheeler Car"])
            other = _f(r["Total"]) - car - bike - three
            d = out.setdefault(str(r["District"]).strip().upper(), {c: 0.0 for c in CLASSES})
            d["car"] += car; d["motorcycle"] += bike; d["three_wheeler"] += three; d["other"] += other
    return out

def class_mix(counts: dict[str, dict[str, float]], min_pool: int = MIN_POOL) -> dict[str, tuple[dict[str, float], str]]:
    """district -> (mix summing to 1, source). Small districts take the national mix."""
    nat = {c: sum(d[c] for d in counts.values()) for c in CLASSES}
    nat_total = sum(nat.values())
    if nat_total <= 0:
        raise ValueError("No EV registrations in the pooled years.")
    nat_mix = {c: nat[c] / nat_total for c in CLASSES}
    out: dict[str, tuple[dict[str, float], str]] = {}
    for dist, d in counts.items():
        t = sum(d.values())
        if t >= min_pool and all(v >= 0 for v in d.values()):
            out[dist] = ({c: d[c] / t for c in CLASSES}, "district")
        else:
            out[dist] = (dict(nat_mix), "national_fallback")
    return out

# ---------------------------------------------------------------- the split
def split_rows(scen_rows: list[Row], ds: list[Row], mix: dict[str, tuple[dict[str, float], str]]) -> list[Row]:
    """Town rows for every district scenario row. Totals are preserved exactly (mix and population shares each sum to 1)."""
    by_dist: dict[str, list[Row]] = {}
    for r in ds:
        by_dist.setdefault(r["district_key"], []).append(r)
    pop_total = {d: float(sum(t["population"] for t in rows)) for d, rows in by_dist.items()}   # exact counts, not the rounded stored share
    out: list[Row] = []
    for s in scen_rows:
        if s["level"] != "district":
            continue
        d = s["district"]
        if d not in by_dist or d not in mix:
            raise ValueError(f"District {d} is missing from the census or the class mix.")
        shares, source = mix[d]
        for t in by_dist[d]:
            for c in CLASSES:
                out.append({"estimate_version": ESTIMATE_VERSION, "run_id": f"{METHOD}@{s['run_id']}", "district": d,
                            "ds_name": t["ds_name"], "ds_code": t["ds_code"], "scenario": s["scenario"], "year": s["year"],
                            "vehicle_class": c, "new_evs": s["new_evs"] * shares[c] * (t["population"] / pop_total[d]),
                            "mix_source": source, "method": METHOD, "status": STATUS})
    return out

def validate_split(rows: list[Row], scen_rows: list[Row], n_ds: int) -> list[str]:
    problems: list[str] = []
    if not rows:
        return ["file has no rows"]
    for c in COLUMNS:
        if any(c not in r for r in rows):
            return [f"missing column: {c}"]
    if any(float(r["new_evs"]) < 0 or math.isnan(float(r["new_evs"])) for r in rows):
        problems.append("negative or missing new_evs")
    if {r["vehicle_class"] for r in rows} != set(CLASSES):
        problems.append(f"vehicle_class must be exactly {CLASSES}")
    if len({(r["district"], r["ds_code"]) for r in rows}) != n_ds:
        problems.append(f"expected {n_ds} DS divisions")
    tot: dict[tuple[str, str, int], float] = {}
    for r in rows:
        k = (r["district"], r["scenario"], int(r["year"]))
        tot[k] = tot.get(k, 0.0) + float(r["new_evs"])
    for s in scen_rows:
        if s["level"] == "district":
            k = (s["district"], s["scenario"], int(s["year"]))
            if k not in tot or abs(tot[k] - s["new_evs"]) > max(0.05, 1e-6 * s["new_evs"]):
                problems.append(f"{k}: DS and class values do not add up to the district scenario value")
                break
    return problems

def write_split(rows: list[Row], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.3f}" if c == "new_evs" else r[c]) for c in COLUMNS})

def read_split(path: str) -> list[Row]:
    with open(path, newline="", encoding="utf-8") as f:
        rows: list[Row] = [dict(r) for r in csv.DictReader(f)]
    for r in rows:
        r["new_evs"] = float(r["new_evs"]); r["year"] = int(r["year"]); r["ds_code"] = int(r["ds_code"])
    return rows

# ---------------------------------------------------------------- district-level check of the population-only idea
def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r

def rank_corr(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (average ranks for ties), without scipy."""
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")

def population_only_check(counts: dict[str, dict[str, float]], ds: list[Row], cls: str) -> tuple[float, float, list[tuple[str, float, float]]]:
    """(Spearman, mean abs gap in share points, rows of (district, actual share %, population share %)) for one class."""
    pop: dict[str, float] = {}
    for r in ds:
        pop[r["district_key"]] = pop.get(r["district_key"], 0.0) + r["population"]
    names = sorted(set(pop) & set(counts))
    tot_pop = sum(pop[n] for n in names)
    tot_ev = sum(counts[n][cls] for n in names)
    act = [100.0 * counts[n][cls] / tot_ev for n in names]
    ps = [100.0 * pop[n] / tot_pop for n in names]
    gap = sum(abs(a - p) for a, p in zip(act, ps)) / len(names)
    return rank_corr(act, ps), gap, list(zip(names, act, ps))

# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--scenarios", default="data/processed/scenarios_v0.csv")
    ap.add_argument("--census", default="data/external/census2024_ds_population.csv")
    ap.add_argument("--out", default="data/processed/town_estimates_v0.csv")
    a = ap.parse_args()
    scen = read_scenarios(a.scenarios)
    ds = read_ds(a.census)
    counts = load_pooled_counts(a.raw_dir)
    mix = class_mix(counts)
    rows = split_rows(scen, ds, mix)
    problems = validate_split(rows, scen, len(ds))
    if problems:
        raise SystemExit("Producer broke its own contract:\n  " + "\n  ".join(problems))
    write_split(rows, a.out)
    back = read_split(a.out)
    fb = sorted(d for d, (_, src) in mix.items() if src != "district")
    print(f"Wrote {a.out}: {len(back):,} rows ({len(ds)} DS divisions x 11 years x 3 scenarios x {len(CLASSES)} classes).")
    print("Districts using the national mix (too few EVs):", ", ".join(fb) if fb else "none")
    print("\nExample, Kaduwela (Colombo district), baseline scenario, new EVs per year:")
    print("  year      cars      bikes   three-wheelers   other")
    for y in (2027, 2030, 2035, 2037):
        v = {c: sum(r["new_evs"] for r in back if r["ds_name"] == "Kaduwela" and r["district"] == "COLOMBO" and r["scenario"] == "baseline"
                    and r["year"] == y and r["vehicle_class"] == c) for c in CLASSES}
        print(f"  {y}  {v['car']:>8,.0f}  {v['motorcycle']:>9,.0f}  {v['three_wheeler']:>14,.0f}  {v['other']:>7,.0f}")
    print("\nCheck: share EVs out to districts by POPULATION ALONE, how far off is each class? (2025 + 2026 Jan-Aug)")
    for cls in ("car", "motorcycle"):
        rho, gap, tab = population_only_check(counts, ds, cls)
        worst = sorted(tab, key=lambda t: -abs(t[1] - t[2]))[:3]
        print(f"  {cls:<10} rank correlation {rho:+.2f}, average gap {gap:.2f} share points; biggest gaps: " +
              "; ".join(f"{n.title()} actual {x:.1f}% vs population {p:.1f}%" for n, x, p in worst))
    print("Status:", STATUS, "- population-only weights inside a district; not validated below district level.")

if __name__ == "__main__":
    main()
