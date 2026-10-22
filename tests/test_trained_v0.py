import numpy as np
import pandas as pd
from src.evaluation.baselines import walk_forward
from src.models.trained_v0 import TRAINED, ridge_lag, gbm_growth

def _series(n=60, seed=1):
    idx = pd.period_range("2019-01", periods=n, freq="M")
    rng = np.random.default_rng(seed)
    v = 20 * 1.06 ** np.arange(n) * rng.lognormal(0, 0.05, n)
    return pd.Series(v, index=idx)

def test_shapes_and_no_negatives():
    h = _series().to_numpy()
    for f in (ridge_lag, gbm_growth):
        p = f(h, 6)
        assert p.shape == (6,) and np.isfinite(p).all() and (p >= 0).all()

def test_ridge_continues_a_smooth_exponential():
    h = 20 * 1.06 ** np.arange(60)
    p = ridge_lag(h, 3)
    truth = 20 * 1.06 ** np.arange(60, 63)
    assert np.allclose(p, truth, rtol=0.1)

def test_gbm_is_deterministic():
    h = _series().to_numpy()
    assert np.array_equal(gbm_growth(h, 4), gbm_growth(h, 4))

def test_no_future_leakage():
    """Changing values AFTER an origin must not change forecasts made at that origin."""
    y = _series(); obs = pd.Series(True, index=y.index)
    a = walk_forward(y, obs, "2022-12", h_max=3, methods=TRAINED)
    y2 = y.copy(); y2.iloc[50:] = y2.iloc[50:] * 5
    b = walk_forward(y2, obs, "2022-12", h_max=3, methods=TRAINED)
    key = ["origin", "target", "h", "method"]
    early = a[a.origin <= pd.Period("2023-01", "M")].merge(b[b.origin <= pd.Period("2023-01", "M")], on=key)
    assert len(early) > 0 and np.allclose(early.y_pred_x, early.y_pred_y)
