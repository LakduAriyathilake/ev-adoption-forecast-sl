"""
External EV data audit (Our World in Data, built on IEA data; CC BY).

Checks the two downloaded files, builds a peer-country table, and aligns countries by
ADOPTION STAGE (years since EV share of new car sales sustainably passed 1%) instead of
calendar year. If the private DMT CSVs are present, Sri Lanka's EV-CAR share is added.

Scope warning: the OWID series counts electric CARS only (battery + plug-in hybrid).
Sri Lanka's EV fleet is mostly motorcycles, so only the CAR series is comparable.

Run from repo root with the venv active:
  python -m src.data.audit_external
Optional (adds Sri Lanka cars from your private DMT files):
  python -m src.data.audit_external --raw-dir data/raw
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Candidate peers: a JUDGEMENT (emerging economies with an EV take-off), not yet justified
# with GDP-per-capita data. Treat as a shortlist to defend or change with the panel.
PEERS = ["Nepal", "Indonesia", "India", "Philippines", "Vietnam", "Thailand", "Malaysia", "Cambodia", "Laos"]
INK, MUTED, ACCENT = "#1f2937", "#6b7280", "#c2410c"
report = []

def say(line=""):
    report.append(line); print(line)

def sig_figs_le2(v):
    """True if a value looks rounded to <= 2 significant figures (OWID rounds later years)."""
    if v == 0 or not np.isfinite(v):
        return False
    return float(f"{v:.2g}") == v

def sustained_crossing(s, thr=1.0):
    """First year from which the share stays >= thr in every later observed year."""
    s = s.dropna().sort_index()
    for y in s.index:
        if (s.loc[y:] >= thr).all():
            return int(y)
    return None

def sl_car_share(raw_dir):
    """Sri Lanka battery-electric CAR share of new car registrations, from the private DMT CSVs."""
    f1 = sorted(Path(raw_dir).glob("*01_*.csv")); f4 = sorted(Path(raw_dir).glob("*04_*.csv"))
    if len(f1) != 1 or len(f4) != 1:
        return None
    d = pd.read_csv(f1[0]); c = pd.read_csv(f4[0])
    ev = d.groupby("Year")["Car"].sum()
    nr = c[(c.Table == "new_registrations") & (c.Vehicle_class == "MOTOR CARS")].set_index("Year").Count
    yrs = [y for y in range(2013, 2026)]
    return (100 * ev.reindex(yrs) / nr.reindex(yrs)).rename("Sri Lanka")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-dir", default="data/external")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--reports", default="outputs/reports")
    ap.add_argument("--figures", default="outputs/figures")
    a = ap.parse_args()
    for p in (a.processed, a.reports, a.figures):
        Path(p).mkdir(parents=True, exist_ok=True)

    sales = pd.read_csv(Path(a.external_dir) / "owid_electric_car_sales.csv")
    share = pd.read_csv(Path(a.external_dir) / "owid_electric_car_sales_share.csv")
    m = sales.merge(share, on=["entity", "code", "year"], how="outer", indicator=True)

    say("# External EV data audit (OWID / IEA, CC BY)\n")
    say("## 1. Structure")
    say(f"- sales file: {len(sales)} rows; share file: {len(share)} rows; {m.entity.nunique()} entities; years {int(m.year.min())}-{int(m.year.max())}")
    both = int((m._merge == "both").sum())
    say(f"- rows present in both files: {both}; only in sales file: {int((m._merge == 'left_only').sum())}; only in share file: {int((m._merge == 'right_only').sum())}")
    dup = int(m.duplicated(["entity", "year"]).sum())
    say(f"- {'PASS' if dup == 0 else 'FAIL'}: duplicate (entity, year) rows = {dup}")
    bad = int(((m.ev_sales_share < 0) | (m.ev_sales_share > 100)).sum())
    say(f"- {'PASS' if bad == 0 else 'FAIL'}: share values outside 0-100 = {bad}")
    for c in ("Sri Lanka", "Bangladesh", "Pakistan"):
        say(f"- {'PRESENT' if (m.entity == c).any() else 'ABSENT'}: {c}")

    say("\n## 2. Precision and meaning")
    sf = share.ev_sales_share.map(sig_figs_le2).mean()
    say(f"- {100 * sf:.0f}% of share values look rounded to <= 2 significant figures (OWID rounds later years); small differences between countries are not meaningful.")
    say("- Definition: electric CARS = battery-electric + plug-in hybrid. DMT's electric series is battery-electric only. Hybrids (HEV) are not counted by IEA.")
    say("- Annual, national only. No months, no districts, no two-wheelers.")

    say("\n## 3. Candidate peers: coverage and adoption stage")
    wide = m.pivot(index="year", columns="entity", values="ev_sales_share")
    rows = []
    for p in PEERS:
        s = wide[p].dropna()
        t0 = sustained_crossing(s)
        cens = bool(t0 is not None and t0 == int(s.index.min()))
        rows.append(dict(country=p, first_year=int(s.index.min()), last_year=int(s.index.max()), n_years=len(s),
                         share_2025=float(s.get(2025, np.nan)), year_passed_1pct=t0, first_obs_already_above_1pct=cens))
    cov = pd.DataFrame(rows)
    say("```\n" + cov.to_string(index=False) + "\n```")
    for r in cov[cov.first_obs_already_above_1pct].itertuples():
        say(f"- CAUTION {r.country}: first observed year ({r.first_year}) is already >= 1%, so its take-off is not observed (left-censored).")
    say("- Peer list is a judgement; justify it with GDP per capita and import-policy context before using it in the thesis.")

    sl = sl_car_share(a.raw_dir) if a.raw_dir else None
    series = {p: wide[p].dropna() for p in PEERS}
    if sl is not None:
        series["Sri Lanka"] = sl.dropna()
        t0 = sustained_crossing(sl)
        say(f"\n- Sri Lanka EV-car share of new car registrations (DMT, private): " + ", ".join(f"{y}: {v:.1f}%" for y, v in sl.loc[2020:2025].items())
            + f"; sustained >=1% from {t0}")
        say("- NOTE: Sri Lanka's denominator (new cars) collapsed during the import suspension (1,489 in 2022, 1,644 in 2024), which inflates shares in those years.")

    # aligned table
    al = []
    for k, s in series.items():
        t0 = sustained_crossing(s)
        if t0 is None:
            continue
        for y, v in zip(s.index.to_list(), s.to_list()):      # plain lists: avoids pandas 'Hashable' typing errors
            al.append(dict(country=k, year=int(y), years_since_1pct=int(y) - t0, share=float(v)))
    al = pd.DataFrame(al)
    al.to_csv(f"{a.processed}/external_peers_aligned.csv", index=False)
    m.drop(columns="_merge").to_csv(f"{a.processed}/external_owid_merged.csv", index=False)

    # figure
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k in series:
        d = al[al.country == k]
        d = d[d.years_since_1pct >= -3]
        if d.empty:
            continue
        isl = k == "Sri Lanka"
        ax.plot(d.years_since_1pct, d.share, lw=2.6 if isl else 1.2, color=ACCENT if isl else MUTED,
                alpha=1 if isl else 0.7, marker="o" if isl else None, ms=4, zorder=3 if isl else 2)
        ax.annotate(k, (d.years_since_1pct.iloc[-1], d.share.iloc[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=ACCENT if isl else INK, va="center")
    ax.set_yscale("log"); ax.set_xlabel("years since EV share of new car sales sustainably passed 1%")
    ax.set_ylabel("EV share of new car sales (%), log scale")
    ax.set_title("EV take-off by adoption stage: peers vs Sri Lanka (cars only)", loc="left", color=INK)
    ax.grid(color="#e5e7eb"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{a.figures}/f4_peers_aligned.png", dpi=160); plt.close(fig)

    say("\n## 4. What this means for experiment E2 (train external, test Sri Lanka)")
    say("- Train on SHARE or GROWTH, aligned by adoption stage, from a PANEL of peers; do not train on one country.")
    say("- Test only on Sri Lankan EV CARS (comparable scope). Two-wheelers need a different source.")
    say("- Sri Lanka has at most ~6 annual points for 2020-2025: report uncertainty, avoid strong claims.")
    Path(f"{a.reports}/external_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote {a.reports}/external_audit.md, {a.figures}/f4_peers_aligned.png, tables in {a.processed}")

if __name__ == "__main__":
    main()