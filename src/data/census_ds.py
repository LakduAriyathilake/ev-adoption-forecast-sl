"""
Census 2024 (provisional) -> population per DS division (the 'town level' used by the spatial step).

Why this exists
  The Department of Census and Statistics publishes population by sex and age for every Grama Niladhari (GN)
  division, about 14,000 rows. The spatial step needs one row per DS division (340 of them, e.g. Kaduwela,
  Kolonnawa, Homagama), so this script (1) reads the GN Excel file, (2) checks every row adds up, (3) sums GN rows
  into DS divisions and (4) writes a small public CSV the rest of the project can use.

Input : the GN Excel file 'GN_population_excel.xlsx' (Census of Population and Housing 2024, provisional), kept in
        data/raw/census2024/ (not committed; the derived CSV below is).
Output: data/external/census2024_ds_population.csv  (one row per DS division)
Source: Department of Census and Statistics, Sri Lanka, CPH 2024 (provisional). Cite it on slides.

Run from the repo root:
  python -m src.data.census_ds --xlsx data/raw/census2024/GN_population_excel.xlsx
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import openpyxl

Row = dict[str, Any]
AGE_KEYS = ["age_0_14", "age_15_59", "age_60_64", "age_65_plus"]
OUT_COLUMNS = ["district", "district_key", "ds_code", "ds_name", "population", "male", "female",
               *AGE_KEYS, "gn_count", "pop_share_in_district"]
EXPECTED_DISTRICTS = 25

def _num(v: object) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    raise ValueError(f"Expected a number in the census file, found {v!r}")

def read_gn(xlsx: str) -> list[Row]:
    """One dict per GN division. Columns by position: 3 district, 4/5 DS code/name, 9 total, 10 male, 11 female, 13-16 age groups."""
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb.active
    if ws is None:
        raise ValueError("The workbook has no active sheet.")
    out: list[Row] = []
    for r in ws.iter_rows(values_only=True):
        if not isinstance(r[2], (int, float)) or not isinstance(r[9], (int, float)):
            continue                                              # title, header and blank rows
        out.append({"district_code": int(_num(r[2])), "district": str(r[3]).strip(), "ds_code": int(_num(r[4])), "ds_name": str(r[5]).strip(),
                    "total": _num(r[9]), "male": _num(r[10]), "female": _num(r[11]),
                    "age_0_14": _num(r[13]), "age_15_59": _num(r[14]), "age_60_64": _num(r[15]), "age_65_plus": _num(r[16])})
    return out

def check_rows(rows: list[Row]) -> list[str]:
    """Rows whose parts do not add up to the total (male+female, or the four age groups)."""
    bad: list[str] = []
    for i, r in enumerate(rows):
        if abs(r["male"] + r["female"] - r["total"]) > 0.5:
            bad.append(f"row {i} {r['district']}/{r['ds_name']}: male+female != total")
        if abs(sum(r[k] for k in AGE_KEYS) - r["total"]) > 0.5:
            bad.append(f"row {i} {r['district']}/{r['ds_name']}: age groups != total")
    return bad

def aggregate_ds(rows: list[Row]) -> list[Row]:
    """Sum GN rows into DS divisions (key = district code + DS code) and add each DS's share of its district."""
    acc: dict[tuple[int, int], Row] = {}
    for r in rows:
        k = (r["district_code"], r["ds_code"])
        if k not in acc:
            acc[k] = {"district": r["district"], "district_key": r["district"].upper(), "ds_code": r["ds_code"],
                      "ds_name": r["ds_name"], "population": 0.0, "male": 0.0, "female": 0.0,
                      **{a: 0.0 for a in AGE_KEYS}, "gn_count": 0}
        a_ = acc[k]
        a_["population"] += r["total"]; a_["male"] += r["male"]; a_["female"] += r["female"]
        for key in AGE_KEYS:
            a_[key] += r[key]
        a_["gn_count"] += 1
    dist_total: dict[str, float] = defaultdict(float)
    for a_ in acc.values():
        dist_total[a_["district_key"]] += a_["population"]
    out = sorted(acc.values(), key=lambda x: (x["district_key"], x["ds_code"]))
    for a_ in out:
        a_["pop_share_in_district"] = a_["population"] / dist_total[a_["district_key"]]
    return out

def write_ds(rows: list[Row], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.6f}" if c == "pop_share_in_district" else (int(r[c]) if c in ("population", "male", "female", "gn_count", *AGE_KEYS) else r[c])) for c in OUT_COLUMNS})

def read_ds(path: str) -> list[Row]:
    with open(path, newline="", encoding="utf-8") as f:
        rows: list[Row] = [dict(r) for r in csv.DictReader(f)]
    for r in rows:
        for c in ("population", "male", "female", "gn_count", "ds_code", *AGE_KEYS):
            r[c] = int(r[c])
        r["pop_share_in_district"] = float(r["pop_share_in_district"])
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/raw/census2024/GN_population_excel.xlsx")
    ap.add_argument("--out", default="data/external/census2024_ds_population.csv")
    a = ap.parse_args()
    gn = read_gn(a.xlsx)
    bad = check_rows(gn)
    ds = aggregate_ds(gn)
    n_dist = len({r["district_key"] for r in ds})
    if n_dist != EXPECTED_DISTRICTS:
        raise SystemExit(f"Expected {EXPECTED_DISTRICTS} districts, found {n_dist}.")
    write_ds(ds, a.out)
    total = sum(r["population"] for r in ds)
    print(f"Read {len(gn):,} GN divisions -> {len(ds)} DS divisions in {n_dist} districts. National population {total:,.0f}.")
    print(f"Rows where parts do not add up to the total: {len(bad)}" + ("" if not bad else f" (first: {bad[0]})"))
    print(f"Wrote {a.out}")
    print("Colombo district DS divisions (population, share of district):")
    for r in sorted((r for r in ds if r["district_key"] == "COLOMBO"), key=lambda r: -r["population"]):
        print(f"  {r['ds_name']:<28}{r['population']:>9,.0f}  {r['pop_share_in_district']:>6.1%}")

if __name__ == "__main__":
    main()
