"""
Trained models v0: do small LEARNED models beat the simple baselines on the national monthly EV series?

Why this exists: the baselines (naive, naive3, snaive, cagr) are fixed rules. Here two small models
LEARN their numbers from the data, and they are scored with exactly the same walk-forward test.

The two models (both are refitted at EVERY forecast origin, using only data up to that origin):
  ridge_lag  Ridge regression. Predicts next month's log(1 + EVs) from the last 3 months.
             Ridge = ordinary linear regression plus a small penalty that stops the weights
             from swinging wildly when there are only ~30 training rows.
  gbm_growth Gradient boosting (small trees). Predicts next month's GROWTH (log change) from the last
             3 months' growth and the current level. Trees cannot go above the values they have seen,
             so we ask them for growth, not level.

Why log(1 + EVs): counts go from ~0 to ~9,000, so we model percentage moves, not raw counts.
Training window: the last 36 months before the origin (older months are the pre-take-off era).

Run from the repo root (private DMT CSVs needed in data/raw):
  python -m src.models.trained_v0 --raw-dir data/raw
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from src.evaluation.baselines import METHODS, FIRST_ORIGIN, HOLDOUT_START, load_series, metrics, walk_forward

LAGS, WINDOW = 3, 36

def _logs(hist):
    return np.log1p(np.asarray(hist, dtype=float))

def _ridge_fit(z):
    """Rows: [z(t-1), z(t-2), z(t-3)] -> z(t), using the last WINDOW months of z."""
    z = z[-(WINDOW + LAGS):]
    X = np.array([z[i - LAGS:i][::-1] for i in range(LAGS, len(z))])
    return Ridge(alpha=1.0).fit(X, z[LAGS:])

def ridge_lag(hist, h_max):
    z = list(_logs(hist))
    m = _ridge_fit(np.array(z))
    out = []
    for _ in range(h_max):                                   # feed each forecast back in as the next input
        nxt = float(m.predict(np.array([z[-LAGS:][::-1]]))[0])
        z.append(nxt)
        out.append(max(float(np.expm1(nxt)), 0.0))
    return np.array(out)

def _gbm_rows(z):
    """Features: last 3 monthly growth steps + current level. Target: next growth step."""
    z = z[-(WINDOW + LAGS + 1):]
    d = np.diff(z)
    X = np.array([np.r_[d[i - LAGS:i][::-1], z[i]] for i in range(LAGS, len(d))])
    return X, d[LAGS:]

def gbm_growth(hist, h_max):
    z = list(_logs(hist))
    X, y = _gbm_rows(np.array(z))
    m = GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.1, subsample=0.8, random_state=0).fit(X, y)
    out = []
    for _ in range(h_max):
        d = np.diff(z[-(LAGS + 1):])
        g = float(m.predict(np.array([np.r_[d[::-1], z[-1]]]))[0])
        z.append(z[-1] + g)
        out.append(max(float(np.expm1(z[-1])), 0.0))
    return np.array(out)

TRAINED = {"ridge_lag": ridge_lag, "gbm_growth": gbm_growth}

def compare(raw_dir):
    """Walk-forward table: baselines + trained models, same origins, same scoring."""
    y, obs = load_series(raw_dir)
    wf = walk_forward(y, obs, FIRST_ORIGIN, methods={**METHODS, **TRAINED})
    wf = wf[wf.observed].copy()
    wf["period"] = np.where(wf.target >= pd.Period(HOLDOUT_START, "M"), "holdout", "development")
    return wf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    a = ap.parse_args()
    Path(a.processed).mkdir(parents=True, exist_ok=True)
    wf = compare(a.raw_dir)
    wf.to_csv(f"{a.processed}/trained_vs_baselines.csv", index=False)
    for per in ("development", "holdout"):
        sub = pd.DataFrame(wf[(wf.period == per) & (wf.h == 1)])
        tab = pd.DataFrame({k: metrics(g) for k, g in sub.groupby("method")}).T.round(1)
        print(f"\n{per.upper()}, 1 month ahead (MAE = average miss in new EVs per month; lower is better)")
        print(tab.sort_values("MAE").to_string())

if __name__ == "__main__":
    main()
