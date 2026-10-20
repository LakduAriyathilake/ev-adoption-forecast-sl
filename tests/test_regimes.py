import numpy as np
import pandas as pd
from src.analysis.regimes import pelt_breaks, consensus, match_events, detect, robust_sigma

def _step_series(seed=0):
    rng = np.random.default_rng(seed)
    level = np.r_[np.full(30, 2.0), np.full(30, 5.0), np.full(30, 8.0)]   # log-level steps at 30 and 60
    return level + rng.normal(0, 0.15, 90)

def test_pelt_finds_known_steps():
    x = _step_series()
    b = pelt_breaks(x, 2.0)
    assert any(abs(i - 30) <= 1 for i in b) and any(abs(i - 60) <= 1 for i in b)

def test_no_breaks_in_pure_noise():
    rng = np.random.default_rng(1)
    assert pelt_breaks(rng.normal(3, 0.2, 80), 2.0) == []

def test_consensus_support_counts():
    found = {0.5: [10, 40], 1.0: [10], 2.0: [11], 4.0: [10]}
    out = dict(consensus(found))
    assert out[10] == 4 and out[40] == 1

def test_detect_returns_robust_steps_on_clean_data():
    idx = pd.period_range("2019-01", periods=90, freq="M")
    y = pd.Series(np.expm1(_step_series()), index=idx)
    _, cons = detect(y)
    robust = [i for i, s in cons if s == 4]
    assert any(abs(i - 30) <= 2 for i in robust) and any(abs(i - 60) <= 2 for i in robust)

def test_match_events_sign_and_window():
    ev = pd.DataFrame({"event_id": ["E1", "E2"],
                       "month": pd.PeriodIndex(["2024-10", "2025-02"], freq="M")})
    assert match_events(pd.Period("2024-07", "M"), ev) == ("E1", -3)    # break BEFORE the event
    assert match_events(pd.Period("2025-04", "M"), ev) == ("E2", 2)     # break AFTER the event
    assert match_events(pd.Period("2020-01", "M"), ev) == (None, None)

def test_robust_sigma_ignores_big_jump():
    x = np.r_[np.zeros(40), np.full(40, 10.0)] + np.random.default_rng(2).normal(0, 0.1, 80)
    assert robust_sigma(x) < 0.3