"""
Regime (structural-break) detection on the NATIONAL MONTHLY EV series, compared with the policy events table.

What it does, in plain words
  1. Takes the monthly EV registrations (DMT, private) and works on log(1 + count), because growth is
     multiplicative: a jump from 100 to 200 matters as much as 5,000 to 10,000.
  2. Runs PELT (a change-point finder from the 'ruptures' library) several times with different
     penalty strengths. A stronger penalty = fewer, bigger breaks. Penalty = multiplier x BIC-style
     base, where the noise level is estimated robustly from month-to-month changes.
  3. Keeps a break as ROBUST only if it appears (within +-2 months) at ALL penalty settings tried.
  4. Compares robust breaks with the dated policy events (nearest event within +-3 months).
  5. Re-runs everything using only data up to 2025-12 to check the robust breaks do not depend on
     the 2026 months (a hindsight check; PELT is an offline method, so this is for labelling and
     understanding regimes, NOT a live alarm).

Run from the repo root (private DMT CSVs needed in data/raw):
  python -m src.analysis.regimes --raw-dir data/raw
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.evaluation.baselines import load_series

MULTS = (0.5, 1.0, 2.0, 4.0)       # penalty strengths tried (x BIC-style base)
MIN_SIZE = 3                        # a regime must last at least 3 months
CLUSTER_TOL = 2                     # breaks within +-2 months count as "the same break"
EVENT_WINDOW = 3                    # event is "near" a break if within +-3 months
TRUNCATE_AT = "2025-12"
INK, BLUE, GREY, LIGHT, ORANGE = "#1f2937", "#1d6fa5", "#6b7280", "#e5e7eb", "#c2410c"

# ---------------------------------------------------------------- detection
def robust_sigma(x):
    """Noise level from month-to-month changes (median absolute deviation, so big jumps do not inflate it)."""
    d = np.diff(x)
    return 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2)

def pelt_breaks(x, mult):
    """Indices where a new regime STARTS (0-based, never 0)."""
    n = len(x)
    pen = mult * 2 * robust_sigma(x) ** 2 * np.log(n)
    bk = rpt.Pelt(model="l2", min_size=MIN_SIZE, jump=1).fit(x.reshape(-1, 1)).predict(pen=pen)
    return [int(b) for b in bk[:-1]]

def consensus(found, tol=CLUSTER_TOL):
    """found: {multiplier: [break indices]}. Returns list of (index, support) for each cluster of nearby breaks."""
    allb = sorted(b for v in found.values() for b in v)
    clusters = []
    for b in allb:
        if clusters and b - clusters[-1][-1] <= tol:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    out = []
    for c in clusters:
        centre = int(np.median(c))
        support = sum(any(abs(b - centre) <= tol for b in v) for v in found.values())
        out.append((centre, support))
    return out

def detect(y):
    x = np.log1p(y.to_numpy(dtype=float))
    found = {m: pelt_breaks(x, m) for m in MULTS}
    return found, consensus(found)

# ---------------------------------------------------------------- policy events
def load_events(path):
    ev = pd.read_csv(path)
    ev = ev[ev["date_precision"] != "year"].copy()                       # year-only rows cannot be placed in a month
    ev["month"] = pd.PeriodIndex(ev["date"].astype(str).str[:7], freq="M")
    return ev[["event_id", "month", "category", "event"]]

def match_events(break_month, events, window=EVENT_WINDOW):
    """Nearest event within +-window months of a break. Returns (event_id, months_from_event) or (None, None).
    months_from_event > 0 means the BREAK comes AFTER the event."""
    best = None
    for _, r in events.iterrows():
        gap = (break_month - r["month"]).n
        if abs(gap) <= window and (best is None or abs(gap) < abs(best[1])):
            best = (r["event_id"], gap)
    return best if best else (None, None)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--events", default="data/external/policy_events_lk.csv")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--reports", default="outputs/reports")
    ap.add_argument("--figures", default="outputs/figures")
    a = ap.parse_args()
    for p in (a.processed, a.reports, a.figures):
        Path(p).mkdir(parents=True, exist_ok=True)

    y, obs = load_series(a.raw_dir)
    events = load_events(a.events)
    found, cons = detect(y)
    n_set = len(MULTS)

    rows = []
    for idx, support in cons:
        month = y.index[idx]
        ev_id, gap = match_events(month, events)
        rows.append({"break_month": str(month), "support": f"{support}/{n_set}", "robust": support == n_set,
                     "nearest_event": ev_id, "months_after_event": gap})
    tab = pd.DataFrame(rows)
    tab["months_after_event"] = tab["months_after_event"].astype("Int64")      # whole months, blank when no event is near
    tab.to_csv(f"{a.processed}/regime_breaks.csv", index=False)

    # hindsight check: only data up to TRUNCATE_AT
    y_cut = y.loc[:TRUNCATE_AT]
    _, cons_cut = detect(y_cut)
    robust_full = [str(y.index[i]) for i, s in cons if s == n_set]
    robust_cut = [str(y_cut.index[i]) for i, s in cons_cut if s == n_set]

    lines = ["# Regime detection report (PRIVATE: built from DMT data)\n",
             f"Series: national monthly EV registrations, {y.index[0]} to {y.index[-1]} ({len(y)} months), log(1+count).",
             f"PELT with penalty multipliers {list(MULTS)} (x BIC-style base), minimum regime length {MIN_SIZE} months.",
             f"ROBUST = detected at all {n_set} settings (within +-{CLUSTER_TOL} months). Event match window = +-{EVENT_WINDOW} months.\n",
             "## Breaks found (a break month = first month of a new regime)", "```", tab.to_string(index=False), "```", "",
             f"## Hindsight check: same method using data only up to {TRUNCATE_AT}",
             f"Robust breaks, full series : {robust_full}",
             f"Robust breaks, truncated   : {robust_cut}",
             "If a robust break survives truncation it does not depend on the 2026 months.\n",
             "## How to read this",
             "- 'months_after_event' > 0: the break comes after the event. < 0: the break comes BEFORE the event (the change started first).",
             "- A break near an event is a coincidence in time, NOT proof the event caused it. Many events are close together.",
             "- PELT sees the whole series at once, so this labels regimes after the fact. A live alarm needs an online method (CUSUM) tested the same way as the baselines.",
             "- Counts before 2023 are very small (often under 100 a month), so early breaks are the least reliable; robust breaks matter most."]
    Path(f"{a.reports}/regimes_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # figure
    ts = pd.PeriodIndex(y.index).to_timestamp()
    x_all = np.asarray(mdates.date2num(ts), dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_all, np.maximum(y.to_numpy(dtype=float), 1.0), color=BLUE, lw=2)
    ax.set_yscale("log")
    ax.xaxis_date()
    ymax = float(ax.get_ylim()[1])
    for idx, support in cons:
        xb = float(x_all[idx])
        if support == n_set:
            ax.axvline(xb, color=ORANGE, lw=1.8)
            ax.text(xb, ymax * 0.9, f" {y.index[idx]}", color=ORANGE, fontsize=8.5, va="top", rotation=90, fontweight="bold")
        else:
            ax.axvline(xb, color=GREY, lw=0.7, ls=":")
    ax.set_ylabel("new EV registrations per month (log scale)")
    ax.set_title("Detected regime breaks (orange = found at every penalty setting; dotted = found at some)", loc="left", color=INK, fontsize=10.5)
    ax.grid(axis="y", color=LIGHT); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{a.figures}/f6_regimes.png", dpi=160); plt.close(fig)
    print(f"\nWrote {a.processed}/regime_breaks.csv, {a.reports}/regimes_report.md, {a.figures}/f6_regimes.png")

if __name__ == "__main__":
    main()