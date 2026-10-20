import numpy as np
import pandas as pd
from src.evaluation.baselines import naive, naive3, snaive, cagr, walk_forward, METHODS

def _series(n=60, seed=0):
    idx = pd.period_range("2019-01", periods=n, freq="M")
    rng = np.random.default_rng(seed)
    return pd.Series(np.arange(1, n + 1) * 3.0 + rng.normal(0, 1, n) + 50, index=idx)

def test_rules_basic():
    h = np.arange(1.0, 25.0)
    assert (naive(h, 3) == 24).all()
    assert np.isclose(naive3(h, 2)[0], 23.0)
    assert snaive(h, 12)[0] == h[-12] and snaive(h, 12)[11] == h[-1]

def test_cagr_exact_on_pure_exponential():
    g = 1.05
    h = 10 * g ** np.arange(36)
    pred = cagr(h, 4)
    # level is the mean of the last 3 points, centred 1 step before the end -> (h+1) steps ahead
    truth = [10 * g ** (35 + k) for k in range(1, 5)]
    assert np.allclose(pred, truth, rtol=0.02)

def test_no_future_leakage():
    """Changing values AFTER an origin must not change forecasts made at that origin."""
    y = _series()
    obs = pd.Series(True, index=y.index)
    a = walk_forward(y, obs, "2021-12")
    y2 = y.copy()
    y2.iloc[45:] = y2.iloc[45:] * 100 + 1000          # rewrite the future
    b = walk_forward(y2, obs, "2021-12")
    cut = y.index[44]                                  # origins up to here only see data before index 45
    ka = a[a.origin <= cut].set_index(["origin", "target", "h", "method"]).y_pred
    kb = b[b.origin <= cut].set_index(["origin", "target", "h", "method"]).y_pred
    both = ka.index.intersection(kb.index)
    assert len(both) > 0 and np.allclose(ka.loc[both], kb.loc[both])

def test_all_methods_present_and_horizon_limit():
    y = _series(); obs = pd.Series(True, index=y.index)
    wf = walk_forward(y, obs, "2021-12", h_max=6)
    assert set(wf.method) == set(METHODS) and wf.h.max() == 6
    assert (wf.target > wf.origin).all()
