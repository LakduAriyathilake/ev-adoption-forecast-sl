"""
E2: train on PEER countries, test on Sri Lanka (the research panel's request).

Idea in plain words
  Every country's EV share of new CAR sales climbs in an S-shape. How fast it climbs next year depends
  on where it is now (early stage: fast relative growth; late stage: it slows). We learn that
  "growth vs stage" rule ONLY from peer countries (OWID: Nepal, Indonesia, India, Philippines,
  Vietnam, Thailand, Malaysia, Cambodia, Laos), then ask it to forecast Sri Lanka's EV-car share one year ahead.
  No Sri Lankan number is used to fit the rule.

  Work is done on logit(share) = log(s/(1-s)). It is scale-free, so a country's size does not matter,
  and it keeps forecasts between 0% and 100%. Stage = the current level of logit(share), so countries
  are compared by how far along they are, not by calendar year.

  Rule:  logit(s[t+1]) = logit(s[t]) + a + b * logit(s[t])      (a, b fitted by pooled least squares on peers)

No-leakage rules
  * Calendar-strict: to forecast year t+1 from origin t, the rule may only use peer transitions whose
    TARGET year is <= t (peers' future is not known at origin t).
  * Peer rows below FLOOR_PCT share carry no signal (rounded to 2 significant figures) and are dropped.
  * An origin needs at least MIN_ROWS training transitions, otherwise it is skipped and reported as skipped.
  * Sri Lanka's 2026 value (Jan-Aug share) is a protected holdout, reported separately.

Honest caveats written into the report: OWID counts BEV+PHEV, DMT counts battery-electric only; Sri Lanka's
denominator (new cars) collapsed during the import suspension, which inflates 2021-2024 shares; Sri Lanka
gives only a handful of yearly points, so errors are shown year by year, not as a significance test.

Run from the repo root (private DMT CSVs in data/raw, OWID CSV in data/external):
  python -m src.models.transfer --raw-dir data/raw
"""
import argparse
import glob
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PEERS = ["Nepal", "Indonesia", "India", "Philippines", "Vietnam", "Thailand", "Malaysia", "Cambodia", "Laos"]
FLOOR_PCT = 0.1          # ignore peer observations below 0.1% share
MIN_ROWS = 8             # fewest peer transitions needed to fit the rule
HOLDOUT_YEAR = 2026
FIRST_ORIGIN_YEAR = 2019   # Sri Lanka window of interest (panel: 2020-2025); targets therefore start in 2020
INK, BLUE, ORANGE, GREY, LIGHT, TEAL = "#1f2937", "#1d6fa5", "#c2410c", "#6b7280", "#e5e7eb", "#0f766e"

def logit_pct(p):
    p = np.clip(np.asarray(p, dtype=float) / 100.0, 1e-4, 0.995)
    return np.log(p / (1 - p))

def inv_logit_pct(z):
    return 100.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))

# ---------------------------------------------------------------- data
# Plain Python lists/dicts are used for the calculations below (pandas only reads the CSVs and writes the outputs),
# so the code is type-checker-friendly whether or not pandas type stubs are installed.
Row = dict[str, Any]

def load_peer_transitions(owid_share_csv: str, peers: list[str] = PEERS) -> list[Row]:
    """One record per peer-year pair (s -> s+1) where s is a real observation above the floor."""
    d = pd.read_csv(owid_share_csv)
    series: dict[str, dict[int, float]] = {}
    for ent, yr, sh in zip(d["entity"].tolist(), d["year"].tolist(), d["ev_sales_share"].tolist()):
        if ent in peers and not math.isnan(float(sh)):
            series.setdefault(str(ent), {})[int(yr)] = float(sh)
    out: list[Row] = []
    for peer in sorted(series):
        s = series[peer]
        for yr in sorted(s):
            if yr + 1 in s and s[yr] >= FLOOR_PCT:
                z_now = float(logit_pct(s[yr])); z_next = float(logit_pct(s[yr + 1]))
                out.append({"peer": peer, "origin": yr, "target": yr + 1, "share_now": s[yr], "share_next": s[yr + 1],
                            "z_now": z_now, "z_next": z_next, "dz": z_next - z_now})
    return out

def _read_one(raw_dir: str, pattern: str) -> pd.DataFrame:
    hits = glob.glob(str(Path(raw_dir) / pattern))
    if len(hits) != 1:
        raise SystemExit(f"Expected one {pattern} in {raw_dir}, found {len(hits)}")
    return pd.read_csv(hits[0])

def load_sl_car_share(raw_dir: str) -> dict[int, float]:
    """Sri Lanka EV-car share (%) of new cars, by year. <=2025: DMT file 01 (EV cars) / file 04 (new MOTOR CARS).
    2026 (Jan-Aug): DMT file 03 electric cars / all new cars in the same months."""
    ev = _read_one(raw_dir, "*01_*.csv"); c4 = _read_one(raw_dir, "*04_*.csv"); f3 = _read_one(raw_dir, "*03_*.csv")
    num: dict[int, float] = {}
    for yr, car in zip(ev["Year"].tolist(), ev["Car"].tolist()):
        num[int(yr)] = num.get(int(yr), 0.0) + float(car)
    den: dict[int, float] = {}
    for tab, cls, yr, cnt in zip(c4["Table"].tolist(), c4["Vehicle_class"].tolist(), c4["Year"].tolist(), c4["Count"].tolist()):
        if tab == "new_registrations" and cls == "MOTOR CARS":
            den[int(yr)] = float(cnt)
    share = {y: 100.0 * num[y] / den[y] for y in sorted(num) if y in den and den[y] > 0 and y <= 2025}
    e = t = 0.0
    for cls, fuel, yr, cnt in zip(f3["Vehicle_class"].tolist(), f3["Fuel"].tolist(), f3["Year"].tolist(), f3["Count"].tolist()):
        if cls == "Motor cars" and fuel != "All" and int(yr) == HOLDOUT_YEAR:
            t += float(cnt)
            if fuel == "Electric":
                e += float(cnt)
    if t > 0:
        share[HOLDOUT_YEAR] = 100.0 * e / t
    return share

# ---------------------------------------------------------------- the peer-trained rule
def fit_rule(trans: list[Row], origin_year: int, exclude_peer: str | None = None) -> tuple[float, float, int] | None:
    """Fit dz = a + b*z_now on peer transitions whose TARGET year <= origin_year. Returns (a, b, n_rows) or None."""
    rows = [r for r in trans if r["target"] <= origin_year and (exclude_peer is None or r["peer"] != exclude_peer)]
    if len(rows) < MIN_ROWS or len({round(r["z_now"], 9) for r in rows}) < 3:
        return None
    b, a = np.polyfit([r["z_now"] for r in rows], [r["dz"] for r in rows], 1)
    return float(a), float(b), len(rows)

def predict_next(share_now: float, rule: tuple[float, float, int]) -> float:
    a, b, _ = rule
    z = float(logit_pct(share_now))
    return float(inv_logit_pct(z + a + b * z))

# ---------------------------------------------------------------- walk-forward on Sri Lanka
def sl_walk_forward(sl: dict[int, float], trans: list[Row]) -> list[Row]:
    """Origin year t -> forecast t+1. Methods: e2_peer_rule (peers only), naive (same share), local_trend (repeat last logit change)."""
    rows: list[Row] = []
    for t in sorted(sl):
        if t < FIRST_ORIGIN_YEAR or t + 1 not in sl:
            continue
        rule = fit_rule(trans, t)
        preds: dict[str, float] = {"naive": float(sl[t])}
        if t - 1 in sl:
            dz_last = float(logit_pct(sl[t]) - logit_pct(sl[t - 1]))
            preds["local_trend"] = float(inv_logit_pct(float(logit_pct(sl[t])) + dz_last))
        preds["e2_peer_rule"] = predict_next(sl[t], rule) if rule else float("nan")
        for m, p in preds.items():
            rows.append({"origin": t, "target": t + 1, "method": m, "actual_pct": float(sl[t + 1]), "pred_pct": p,
                         "n_train": rule[2] if rule else 0,
                         "period": "holdout" if t + 1 == HOLDOUT_YEAR else "development"})
    return rows

# ---------------------------------------------------------------- how well does the rule transfer BETWEEN peers? (benchmark)
def leave_one_peer_out(trans: list[Row]) -> list[Row]:
    rows: list[Row] = []
    for r in trans:
        rule = fit_rule(trans, int(r["origin"]), exclude_peer=r["peer"])
        if rule is None:
            continue
        rows.append({"peer": r["peer"], "origin": int(r["origin"]), "actual_pct": r["share_next"],
                     "e2": predict_next(r["share_now"], rule), "naive": r["share_now"]})
    return rows

def mae(rows: list[Row], method: str, origins: set[int] | None = None) -> float:
    errs = [abs(r["pred_pct"] - r["actual_pct"]) for r in rows
            if r["method"] == method and not math.isnan(r["pred_pct"]) and (origins is None or r["origin"] in origins)]
    return float(np.mean(errs)) if errs else float("nan")

# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--owid-share", default="data/external/owid_electric_car_sales_share.csv")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--reports", default="outputs/reports")
    ap.add_argument("--figures", default="outputs/figures")
    a = ap.parse_args()
    for p in (a.processed, a.reports, a.figures):
        Path(p).mkdir(parents=True, exist_ok=True)

    trans = load_peer_transitions(a.owid_share)
    sl = load_sl_car_share(a.raw_dir)
    wf = sl_walk_forward(sl, trans)
    pd.DataFrame(wf).to_csv(f"{a.processed}/e2_transfer_forecasts.csv", index=False)
    lopo = leave_one_peer_out(trans)

    L = ["# E2 report: train on peer countries, test on Sri Lanka (PRIVATE: uses DMT data)\n",
         f"Peers: {', '.join(PEERS)}. Peer transitions used (share >= {FLOOR_PCT}%): {len(trans)}.",
         "Sri Lanka EV-car share of new cars (%): " + ", ".join(f"{y}: {v:.2f}" for y, v in sorted(sl.items())),
         "(2026 = Jan-Aug.)\n",
         "## Sri Lanka walk-forward, one year ahead (predicted share %, by origin year)", "```",
         f"{'origin':>6} {'target':>6} {'actual':>8} {'n_train':>7} {'e2_peer_rule':>13} {'local_trend':>12} {'naive':>8}"]
    for o in sorted({r["origin"] for r in wf}):
        rs = {r["method"]: r for r in wf if r["origin"] == o}
        first = rs["naive"]
        cell = lambda m: f"{rs[m]['pred_pct']:.2f}" if m in rs and not math.isnan(rs[m]["pred_pct"]) else "n/a"
        L.append(f"{o:>6} {first['target']:>6} {first['actual_pct']:>8.2f} {first['n_train']:>7} {cell('e2_peer_rule'):>13} {cell('local_trend'):>12} {cell('naive'):>8}")
    L += ["```", ""]
    for per in ("development", "holdout"):
        sub = [r for r in wf if r["period"] == per]
        origins = {r["origin"] for r in sub if r["method"] == "e2_peer_rule" and not math.isnan(r["pred_pct"])}
        res = {m: mae(sub, m, origins) for m in ("e2_peer_rule", "naive", "local_trend")}
        L += [f"MAE in percentage points, {per}, on the {len(origins)} origin(s) where E2 could be fitted {sorted(origins)}:",
              "  " + "   ".join(f"{m}: {v:.2f}" for m, v in res.items()), ""]
    if lopo:
        e2_err = float(np.mean([abs(r["e2"] - r["actual_pct"]) for r in lopo]))
        nv_err = float(np.mean([abs(r["naive"] - r["actual_pct"]) for r in lopo]))
        L += ["## Benchmark: how well does the peer rule transfer BETWEEN peers? (leave-one-peer-out, calendar-strict)",
              f"transitions scored: {len(lopo)}   MAE pp:  e2 {e2_err:.2f}   naive {nv_err:.2f}", ""]
    L += ["## How to read this",
          "- n_train = number of peer transitions the rule was fitted on at that origin (calendar-strict). Too few = skipped (n/a).",
          "- Sri Lanka's share FELL from 2024 to 2025 because new-car registrations exploded after the reopening (the denominator), "
          "while EV cars still grew. A peer rule that expects a smooth S-curve cannot anticipate that. This is a regime effect, "
          "which is why the framework treats the peer model as ONE candidate that the agent can down-weight after a backtest.",
          "- OWID = battery + plug-in hybrid cars; DMT = battery-electric only. Levels are not exactly comparable.",
          "- Only a handful of Sri Lankan yearly points exist: read the year-by-year table, do not claim significance."]
    Path(f"{a.reports}/e2_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    # figure
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    years = [y for y in sorted(sl) if y >= FIRST_ORIGIN_YEAR]
    ax.plot(years, [sl[y] for y in years], color=INK, lw=2.2, marker="o", ms=5)
    ax.annotate("actual (Sri Lanka)", (years[-1], sl[years[-1]]), xytext=(6, 4), textcoords="offset points", fontsize=9, color=INK, va="center")
    for m, colr, lab, dy in (("e2_peer_rule", ORANGE, "E2 peer rule", 6), ("naive", BLUE, "naive", -10)):
        pts = sorted((r["target"], r["pred_pct"]) for r in wf if r["method"] == m and not math.isnan(r["pred_pct"]))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colr, lw=1.6, marker="s", ms=4)
            ax.annotate(lab, (pts[-1][0], pts[-1][1]), xytext=(6, dy), textcoords="offset points", fontsize=9, color=colr, va="center")
    ax.set_xlim(min(years) - 0.3, max(years) + 1.8)
    ax.set_ylabel("EV share of new cars (%)")
    ax.set_title("E2: peer-trained one-year-ahead forecasts vs Sri Lanka actual (2026 = Jan-Aug holdout)", loc="left", color=INK, fontsize=10.5)
    ax.grid(axis="y", color=LIGHT); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{a.figures}/f7_e2_transfer.png", dpi=160); plt.close(fig)
    print(f"\nWrote {a.processed}/e2_transfer_forecasts.csv, {a.reports}/e2_report.md, {a.figures}/f7_e2_transfer.png")

if __name__ == "__main__":
    main()