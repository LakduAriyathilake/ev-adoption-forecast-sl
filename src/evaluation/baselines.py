"""
Baseline forecasts + walk-forward (rolling-origin) evaluation on the NATIONAL MONTHLY EV series.

Why baselines: every fancier model (regime-aware, Bass, XGBoost, the adaptive framework) must
beat these simple rules, or it has not earned its complexity.

Methods (each uses ONLY data up to the forecast origin):
  naive      next months = last observed month
  naive3     next months = mean of the last 3 months
  snaive     each month = same month one year earlier
  cagr       smoothed growth: level = mean of last 3 months, monthly growth g from the same
             3-month mean one year earlier; forecast = level * g**(h+1)

Evaluation: at every origin from 2021-12 onward, forecast 1-6 months ahead and score against
what actually happened. Targets up to 2025-12 are 'development'; targets from 2026-01 are the
PROTECTED HOLDOUT (nothing is tuned on them; they are reported separately).

Run from the repo root (private DMT CSVs needed in data/raw):
  python -m src.evaluation.baselines --raw-dir data/raw
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

H = 6
START, FIRST_ORIGIN, HOLDOUT_START = "2019-01", "2021-12", "2026-01"
INK, BLUE, ORANGE, GRID = "#1f2937", "#1d6fa5", "#c2410c", "#e5e7eb"

# ---------------------------------------------------------------- forecasting rules
def naive(hist, h_max):
    return np.repeat(hist[-1], h_max)

def naive3(hist, h_max):
    return np.repeat(hist[-3:].mean(), h_max)

def snaive(hist, h_max):
    return np.array([hist[-12 + (h - 1)] for h in range(1, h_max + 1)])

def cagr(hist, h_max):
    level, base = hist[-3:].mean(), hist[-15:-12].mean()
    g = (level / base) ** (1 / 12) if base > 0 else 1.0
    return np.array([level * g ** (h + 1) for h in range(1, h_max + 1)])

METHODS = {"naive": naive, "naive3": naive3, "snaive": snaive, "cagr": cagr}

# ---------------------------------------------------------------- walk-forward
def walk_forward(y, observed, first_origin, h_max=H, methods=METHODS):
    """y: Series with PeriodIndex (gaps filled for INPUT only); observed: bool Series (True = real DMT value).
    Returns long table: origin, target, h, method, y_true, y_pred, observed."""
    rows, idx = [], y.index
    start = idx.get_loc(pd.Period(first_origin, "M"))
    for o in range(start, len(y) - 1):
        hist = y.iloc[: o + 1].to_numpy(float)          # data up to and including the origin ONLY
        for name, f in methods.items():
            pred = f(hist, h_max)
            for h in range(1, h_max + 1):
                t = o + h
                if t >= len(y):
                    break
                rows.append((idx[o], idx[t], h, name, float(y.iloc[t]), float(pred[h - 1]), bool(observed.iloc[t])))
    return pd.DataFrame(rows, columns=["origin", "target", "h", "method", "y_true", "y_pred", "observed"])

def metrics(df):
    e = df.y_pred - df.y_true
    return pd.Series({"n": len(df), "MAE": e.abs().mean(),
                      "sMAPE_%": (200 * e.abs() / (df.y_true.abs() + df.y_pred.abs())).mean(),
                      "bias": e.mean()})

def load_series(raw_dir):
    f = sorted(Path(raw_dir).glob("*02_*.csv"))
    if len(f) != 1:
        raise SystemExit(f"Expected one *02_*.csv in {raw_dir}, found {len(f)}")
    d = pd.read_csv(f[0])
    s = pd.Series(pd.to_numeric(d.New_EV_registrations, errors="coerce").to_numpy(float),
                  index=pd.PeriodIndex(d.Period, freq="M"))
    s = s.loc[START:]
    observed = s.notna()
    return s.interpolate(limit_direction="both"), observed     # interpolate only for model INPUT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--reports", default="outputs/reports")
    ap.add_argument("--figures", default="outputs/figures")
    a = ap.parse_args()
    for p in (a.processed, a.reports, a.figures):
        Path(p).mkdir(parents=True, exist_ok=True)

    y, obs = load_series(a.raw_dir)
    wf = walk_forward(y, obs, FIRST_ORIGIN)
    wf = wf[wf.observed].copy()                                  # never score a month DMT left blank
    wf["period"] = np.where(wf.target >= pd.Period(HOLDOUT_START, "M"), "holdout", "development")
    wf.to_csv(f"{a.processed}/baseline_forecasts.csv", index=False)

    lines = ["# Baseline walk-forward results (PRIVATE: built from DMT data)\n",
             f"Series: national monthly EV registrations, {y.index[0]} to {y.index[-1]} ({len(y)} months). "
             f"{int((~obs).sum())} blank month(s) interpolated for model input only and never scored.",
             f"Origins {FIRST_ORIGIN} onward, horizons 1-{H} months. Development = targets before {HOLDOUT_START}; "
             "holdout = targets from then on (not used to choose anything).\n"]
    for per in ("development", "holdout"):
        sub = wf[(wf.period == per) & wf.h.isin([1, 3, 6])]
        tab = pd.DataFrame({k: metrics(g) for k, g in sub.groupby(["h", "method"])}).T.round(1)
        tab.index = pd.MultiIndex.from_tuples(list(tab.index), names=["h", "method"])
        lines += [f"## {per.capitalize()} period", "```", tab.to_string(), "```", ""]
        # relative MAE vs naive on identical pairs
        piv = sub.assign(ae=(sub.y_pred - sub.y_true).abs()).pivot_table(index=["h", "target", "origin"], columns="method", values="ae")
        rel_mae = piv.drop(columns="naive").groupby(level="h").mean().div(piv["naive"].groupby(level="h").mean(), axis=0)
        lines += [f"MAE relative to naive ({per}; below 1.00 beats naive):", "```", rel_mae.round(2).to_string(), "```", ""]
    lines += ["## How to read this",
              "- bias > 0 means the method over-forecasts, < 0 under-forecasts. During a rapid take-off a 'last value' rule under-forecasts.",
              "- Regime breaks (Feb 2025 reopening; the Mar 2026 jump) are where simple rules fail; those are the cases the adaptive framework must handle.",
              "- Counts span three orders of magnitude, so sMAPE is shown beside MAE."]
    Path(f"{a.reports}/baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # figure: actual vs 3-month-ahead forecasts
    h3 = wf[wf.h == 3]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ts = pd.PeriodIndex(y.index).to_timestamp()
    x_all = np.asarray(mdates.date2num(ts), dtype=float)          # matplotlib date numbers (plain floats)
    obs_mask = obs.to_numpy(dtype=bool)
    ax.plot(x_all[obs_mask], y.to_numpy(dtype=float)[obs_mask], color=INK, lw=2.2)
    ax.annotate("actual", (float(x_all[-1]), float(y.iloc[-1])), xytext=(5, 0), textcoords="offset points",
                fontsize=9, color=INK, va="center")
    for name, colr in (("naive", BLUE), ("cagr", ORANGE)):
        d = h3[h3.method == name].sort_values("target")
        xs = np.asarray(mdates.date2num(pd.PeriodIndex(d["target"]).to_timestamp()), dtype=float)
        ys = d["y_pred"].to_numpy(dtype=float)
        ax.plot(xs, ys, color=colr, lw=1.4)
        ax.annotate(name, (float(xs[-1]), float(ys[-1])), xytext=(5, 0),
                    textcoords="offset points", fontsize=9, color=colr, va="center")
    ax.xaxis_date()                                               # tell matplotlib the x numbers are dates
    hold_x = float(mdates.date2num(pd.Period(HOLDOUT_START, "M").to_timestamp()))   # matplotlib date number
    end_x = float(mdates.date2num(ts[-1] + pd.Timedelta(days=20)))
    ax.axvspan(hold_x, end_x, color=GRID, alpha=0.6, lw=0)
    ax.text(hold_x + 8, ax.get_ylim()[1] * 0.9, "holdout", fontsize=8, color="#6b7280", va="top")
    ax.set_yscale("log"); ax.set_ylabel("new EV registrations per month (log scale)")
    ax.set_title("Baselines: 3-month-ahead forecasts vs actual (national, DMT, private)", loc="left", color=INK)
    ax.grid(color=GRID); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(float(mdates.date2num(ts[0])), float(mdates.date2num(ts[-1] + pd.Timedelta(days=130))))
    fig.tight_layout(); fig.savefig(f"{a.figures}/f5_baselines_h3.png", dpi=160); plt.close(fig)
    print(f"\nWrote {a.processed}/baseline_forecasts.csv, {a.reports}/baseline_report.md, {a.figures}/f5_baselines_h3.png")

if __name__ == "__main__":
    main()