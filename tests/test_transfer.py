import numpy as np
import pandas as pd
from src.models.transfer import (logit_pct, inv_logit_pct, fit_rule, predict_next, load_peer_transitions,
                                 sl_walk_forward, MIN_ROWS)

def _synthetic_transitions(a=0.9, b=-0.15, seed=0):
    """Peer transitions that follow dz = a + b*z exactly (plus tiny noise), spread over years."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, peer in enumerate(["P1", "P2", "P3", "P4"]):
        z = -4.0 + k * 0.3
        for yr in range(2015, 2025):
            dz = float(a + b * z + rng.normal(0, 0.01))
            rows.append({"peer": peer, "origin": yr, "target": yr + 1,
                         "share_now": float(inv_logit_pct(z)), "share_next": float(inv_logit_pct(z + dz)),
                         "z_now": z, "z_next": z + dz, "dz": dz})
            z = z + dz
    return rows

def test_logit_roundtrip():
    s = np.array([0.5, 5.0, 40.0, 90.0])
    assert np.allclose(inv_logit_pct(logit_pct(s)), s, rtol=1e-6)

def test_fit_rule_recovers_known_parameters():
    rule = fit_rule(_synthetic_transitions(), 2024)
    assert rule is not None
    a, b, n = rule
    assert abs(a - 0.9) < 0.05 and abs(b + 0.15) < 0.02 and n > MIN_ROWS

def test_fit_rule_is_calendar_strict():
    t = _synthetic_transitions()
    r1 = fit_rule(t, 2020)
    t2 = [dict(r, dz=99.0) if r["target"] > 2020 else dict(r) for r in t]      # rewrite everything AFTER the origin
    r2 = fit_rule(t2, 2020)
    assert r1 is not None and r2 is not None and np.allclose(r1, r2)

def test_fit_rule_skips_when_too_few_rows():
    assert fit_rule(_synthetic_transitions(), 2016) is None                    # only a handful of transitions exist by then

def test_predictions_stay_between_0_and_100():
    for share in (0.2, 5.0, 50.0, 95.0):
        p = predict_next(share, (1.0, -0.1, 20))
        assert 0.0 < p < 100.0

def test_peer_loader_applies_floor(tmp_path):
    f = tmp_path / "share.csv"
    pd.DataFrame({"entity": ["Nepal"] * 4, "code": ["NPL"] * 4, "year": [2020, 2021, 2022, 2023],
                  "ev_sales_share": [0.01, 0.5, 2.0, 8.0]}).to_csv(f, index=False)
    t = load_peer_transitions(str(f), peers=["Nepal"])
    origins = {r["origin"] for r in t}
    assert 2020 not in origins and origins == {2021, 2022}

def test_sl_forecast_ignores_future_sl_values():
    """The peer-rule forecast made at an origin must not change if later Sri Lankan values change."""
    t = _synthetic_transitions()
    sl1 = dict(zip(range(2019, 2027), [0.3, 0.5, 1.0, 2.0, 5.0, 12.0, 8.0, 9.0]))
    sl2 = dict(sl1); sl2.update({2024: 77.0, 2025: 55.0, 2026: 66.0})
    w1 = sl_walk_forward(sl1, t); w2 = sl_walk_forward(sl2, t)
    a = [r["pred_pct"] for r in w1 if r["origin"] <= 2022 and r["method"] == "e2_peer_rule"]
    b = [r["pred_pct"] for r in w2 if r["origin"] <= 2022 and r["method"] == "e2_peer_rule"]
    assert len(a) > 0 and np.allclose(np.nan_to_num(a, nan=-1.0), np.nan_to_num(b, nan=-1.0))