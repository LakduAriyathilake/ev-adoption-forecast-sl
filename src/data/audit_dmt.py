"""
DMT data audit from the 5 prepared CSV files (PRIVATE data - never commit).

Expected files in --raw-dir (matched by their number, so any prefix is fine):
  01_ev_registrations_by_district_year_type.csv
  02_ev_registrations_national_monthly.csv
  03_new_registrations_by_fuel_monthly_long.csv
  04_registrations_and_vehicle_population_by_class_year_long.csv
  05_vehicle_population_by_district_class.csv

Writes (all git-ignored):
  data/processed/*.csv           tidy tables for modelling
  outputs/reports/dmt_audit.md   audit report
  outputs/figures/*.png          three evidence charts

Run from repo root with the venv active:
  python -m src.data.audit_dmt --raw-dir data/raw
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # draw to files, no window needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
# EV column in file 01  ->  class name in file 04  (for EV share of new registrations)
TYPE_MAP = {"Car": "MOTOR CARS", "Motor Cycle": "MOTOR CYCLES", "Three Wheeler": "MOTOR TRICYCLES"}
INK, MUTED, ACCENT, ACCENT2 = "#1f2937", "#6b7280", "#1d6fa5", "#c2410c"
report = []

def say(line=""):
    report.append(line); print(line)

def find(raw_dir, n):
    hits = sorted(Path(raw_dir).glob(f"*{n:02d}_*.csv"))
    if len(hits) != 1:
        raise SystemExit(f"Expected exactly one file matching *{n:02d}_*.csv in {raw_dir}, found {len(hits)}")
    return hits[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--reports", default="outputs/reports")
    ap.add_argument("--figures", default="outputs/figures")
    a = ap.parse_args()
    for p in (a.processed, a.reports, a.figures):
        Path(p).mkdir(parents=True, exist_ok=True)

    f1, f2, f3, f4, f5 = (find(a.raw_dir, i) for i in range(1, 6))
    dist = pd.read_csv(f1); mon = pd.read_csv(f2); fuel = pd.read_csv(f3)
    cls = pd.read_csv(f4); stock = pd.read_csv(f5)
    vtypes = [c for c in dist.columns if c not in ("Year", "Period", "District", "Total")]

    say("# DMT data audit (PRIVATE - real DMT data, do not publish)\n")
    say("Files: " + ", ".join(p.name for p in (f1, f2, f3, f4, f5)) + "\n")

    # ---- 1 structure
    say("## 1. Structure and integrity")
    years = sorted(dist.Year.unique()); nd = dist.District.nunique()
    grid_ok = len(dist) == nd * len(years) and not dist.duplicated(["Year", "District"]).any()
    say(f"- 01 district x year: {len(dist)} rows, {nd} districts x {len(years)} years ({years[0]}-{years[-1]}) -> {'PASS complete grid, no duplicates' if grid_ok else 'FAIL grid incomplete or duplicated'}")
    say(f"- 01 missing values: {int(dist.isna().sum().sum())}")
    rowsum_bad = int((dist[vtypes].sum(axis=1) != dist.Total).sum())
    say(f"- 01 'Total' equals sum of vehicle-type columns: {'PASS' if rowsum_bad == 0 else f'FAIL on {rowsum_bad} rows'}")
    partial = dist.loc[dist.Period != "Jan-Dec", ["Year", "Period"]].drop_duplicates()
    for _, r in partial.iterrows():
        say(f"- WARN {r.Year} is partial ({r.Period}) -> exclude from annual comparisons / treat as incomplete")
    say(f"- {'PASS' if nd == 25 else 'FAIL'}: {nd} districts (expected 25). Note: the earlier raw workbook had 2 'invalid ...' labels; they are already merged in these CSVs.")
    same = set(dist.District) == set(stock.District)
    say(f"- {'PASS' if same else 'FAIL'}: district names in 01 match 05")

    # ---- 2 reconciliation
    say("\n## 2. Reconciliation")
    a1 = dist.groupby("Year").Total.sum()
    mon["New_EV_registrations"] = pd.to_numeric(mon.New_EV_registrations, errors="coerce")
    a2 = mon.groupby("Year").New_EV_registrations.sum()
    diff = (a1 - a2).abs().max()
    say(f"- {'PASS' if diff < 0.5 else 'FAIL'}: 01 annual totals vs 02 monthly sums, max difference = {diff:.1f} vehicles over {len(a1)} years")

    ele = fuel[fuel.Fuel == "Electric"]
    mi = {m: i + 1 for i, m in enumerate(MONTHS)}
    fm = ele.assign(m=ele.Month.map(mi)).groupby(["Year", "m"]).Count.sum()
    mm = mon.set_index(["Year", "Month_number"]).New_EV_registrations
    for y in sorted(ele.Year.unique()):
        v = fm.loc[y]; ref = mm.loc[y].reindex(v.index)
        gap = 100 * (v - ref).abs().sum() / ref.sum()
        say(f"- 03 fuel sheet, electric total vs 02, {y}: {int(v.sum())} vs {int(ref.sum())} (mean abs monthly gap {gap:.1f}%) -> {'PASS' if gap < 5 else 'WARN'}")
    tot = fuel[fuel.Fuel == "All"].groupby("Year").Count.sum()
    chk = cls[(cls.Table == "new_registrations") & (cls.Vehicle_class == "TOTAL")].set_index("Year").Count
    for y in (2024, 2025):
        say(f"- 03 'All fuels' total vs 04 new registrations total, {y}: {int(tot[y])} vs {int(chk[y])} -> {'PASS' if tot[y] == chk[y] else 'WARN'}")

    # ---- 3 monthly series
    say("\n## 3. National monthly series")
    dots = mon[mon.Source_was_dot.astype(str).str.lower() == "true"]
    say(f"- {len(dots)} months are '.' in the source (kept as missing, NOT zero; confirm meaning with DMT): " + ", ".join(dots.Period))
    s = mon.dropna(subset=["New_EV_registrations"]).copy()
    s["prev"] = s.New_EV_registrations.shift(1)
    s["ratio"] = s.New_EV_registrations / s.prev
    jumps = s[(s.prev >= 1000) & ((s.ratio > 1.8) | (s.ratio < 0.55))]
    for r in jumps.to_dict("records"):          # plain dicts: no pandas typing surprises
        say(f"- FLAG month-on-month jump {r['Period']}: {int(r['prev'])} -> {int(r['New_EV_registrations'])} "
            f"(x{float(r['ratio']):.2f}); ask DMT whether real or a process/data change")

    # ---- 4 EV share
    say("\n## 4. EV share of new registrations (national, by vehicle class)")
    nr = cls[cls.Table == "new_registrations"].pivot_table(index="Year", columns="Vehicle_class", values="Count", aggfunc="sum")
    e = dist.groupby("Year")[list(TYPE_MAP) + ["Total"]].sum()
    sh = pd.DataFrame({k: 100 * e.loc[2020:2025, k] / nr.loc[2020:2025, v] for k, v in TYPE_MAP.items()})
    sh["All classes"] = 100 * e.loc[2020:2025, "Total"] / nr.loc[2020:2025, "TOTAL"]
    sh = sh.round(1)
    say("```\n" + sh.to_string() + "\n```")
    say("- Small denominators (e.g. under 100 new tricycles in 2022-2024) make three-wheeler shares unstable; always show counts beside shares.")

    # ---- 5 districts
    say("\n## 5. Districts")
    wide = dist.pivot_table(index="District", columns="Year", values="Total", aggfunc="sum")
    p25 = wide[2025].sort_values(ascending=False)
    say(f"- Top 3 districts hold {100 * p25.head(3).sum() / p25.sum():.1f}% of 2025 EV registrations: " + ", ".join(p25.head(3).index))
    zeros = int((wide.loc[:, 2020:2025] == 0).sum().sum())
    say(f"- {zeros} zero cells of {wide.shape[0] * 6} in the district x year grid for 2020-2025 -> sparse early years: pool districts instead of one model each")
    frac = [c for c in stock.columns[1:] if (stock[c].dropna() % 1 != 0).any()]
    say(f"- 05 columns with non-integer 'counts' (estimated/allocated, not raw DMT counts): {frac}")
    inten = (1000 * wide[2025] / stock.set_index("District").Grand_total).sort_values(ascending=False).round(2)
    say("- 2025 EV registrations per 1,000 vehicles on the road (top 5): " + ", ".join(f"{k} {v}" for k, v in inten.head(5).items()))
    say("- CAVEAT: DMT district = registration district; it may not equal where the owner lives or charges.")

    # ---- 6 limits
    say("\n## 6. Limits that shape the method")
    say("- No monthly district data -> model the national monthly series, then allocate to districts with a pooled / share model.")
    say("- Cumulative registrations are not fleet size: there is no deregistration data.")
    say("- 2015 spike and the March 2026 monthly jump are unexplained in the files; open questions for DMT.")

    # ---- write tidy tables
    P = a.processed
    dist.melt(id_vars=["Year", "Period", "District", "Total"], value_vars=vtypes,
              var_name="vehicle_type", value_name="n").to_csv(f"{P}/ev_district_year_type_long.csv", index=False)
    wide.reset_index().to_csv(f"{P}/ev_district_year_wide.csv", index=False)
    mon.to_csv(f"{P}/ev_national_month.csv", index=False)
    nr.to_csv(f"{P}/all_new_registrations_year.csv")
    sh.to_csv(f"{P}/ev_share_new_registrations.csv")
    ele.to_csv(f"{P}/electric_by_class_month.csv", index=False)
    stock.to_csv(f"{P}/vehicle_population_district_2025.csv", index=False)

    # ---- figures (single series -> one colour, direct labels)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e7eb"})
    t = pd.PeriodIndex(s.Period, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, s.New_EV_registrations, color=ACCENT, lw=2)
    ax.axvline(float(mdates.date2num(pd.Timestamp("2025-02-01"))), color=MUTED, ls="--", lw=1)
    ax.text(float(mdates.date2num(pd.Timestamp("2025-01-10"))), float(s.New_EV_registrations.max()) * 0.97,
            "Feb 2025: import reopening", color=MUTED, fontsize=9, va="top", ha="right")
    ax.set_xlim(float(mdates.date2num(pd.Timestamp("2019-01-01"))),
                float(mdates.date2num(t.max() + pd.Timedelta(days=45))))
    ax.set_title("New EV registrations per month, Sri Lanka (DMT, private)", loc="left", color=INK)
    ax.set_ylabel("registrations"); fig.tight_layout(); fig.savefig(f"{a.figures}/f1_national_monthly.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for col, lab, colr in [("Car", "Cars", ACCENT), ("Motor Cycle", "Motorcycles", ACCENT2)]:
        ax.plot(sh.index, sh[col], marker="o", lw=2, color=colr)
        ax.annotate(lab, (sh.index[-1], sh[col].iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_title("EV share of new registrations by class", loc="left", color=INK)
    ax.set_ylabel("% of new registrations"); fig.tight_layout(); fig.savefig(f"{a.figures}/f2_ev_share.png", dpi=160); plt.close(fig)

    top = p25.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh([str(d) for d in top.index], top.to_numpy(), color=ACCENT, height=0.6); ax.grid(axis="y", visible=False)
    for i, v in enumerate(top.to_numpy()):
        ax.text(v, i, f" {int(v):,}", va="center", fontsize=9, color=INK)
    ax.set_title("EV registrations in 2025 by district (top 10)", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(f"{a.figures}/f3_districts_2025.png", dpi=160); plt.close(fig)

    Path(f"{a.reports}/dmt_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote tables to {P}, report to {a.reports}/dmt_audit.md, figures to {a.figures}")

if __name__ == "__main__":
    main()
