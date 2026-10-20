import math
import numpy as np
import pytest
from src.models.forecast_v0 import (month_add, empirical_ratio_quantiles, national_forecast, district_shares,
                                    build_contract_rows, validate_contract, write_contract, read_contract, COLUMNS)

def _series(n=60, growth=1.05, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    periods = [month_add("2021-01", k) for k in range(n)]
    values = [float(100 * growth ** k * math.exp(rng.normal(0, noise))) for k in range(n)]
    return values, periods, [True] * n

def _shares():
    return district_shares({f"D{i:02d}": float(i + 1) for i in range(25)})

def _good_rows():
    v, p, o = _series()
    return build_contract_rows(national_forecast(v, p, o), _shares(), p[-1])

def test_month_add_crosses_year():
    assert month_add("2026-11", 3) == "2027-02" and month_add("2026-08", 4) == "2026-12"

def test_ratio_quantiles_are_ordered_and_near_growth():
    v, p, o = _series(noise=0.02)
    q = empirical_ratio_quantiles(v, p, o)
    for h, (lo, mid, hi) in q.items():
        assert lo <= mid <= hi
        assert abs(mid - 1.05 ** h) < 0.05 * 1.05 ** h

def test_forecast_uses_only_the_data_it_is_given():
    v, p, o = _series()
    f1 = national_forecast(v[:50], p[:50], o[:50])
    v2 = v[:50] + [v_ * 1000 for v_ in v[50:]]               # rewrite everything after month 50
    f2 = national_forecast(v2[:50], p[:50], o[:50])
    assert f1 == f2

def test_good_file_passes_validation():
    assert validate_contract(_good_rows()) == []

def test_district_p50_sum_to_national():
    rows = _good_rows()
    first = rows[0]["period"]
    nat = [r for r in rows if r["level"] == "national" and r["period"] == first][0]["p50"]
    dis = sum(r["p50"] for r in rows if r["level"] == "district" and r["period"] == first)
    assert abs(nat - dis) < 1e-6 * nat

def test_validator_catches_bad_ordering_negative_and_blank():
    rows = _good_rows()
    rows[0]["p10"], rows[0]["p90"] = rows[0]["p90"], rows[0]["p10"]
    assert any("p10 <= p50 <= p90" in m for m in validate_contract(rows))
    rows = _good_rows(); rows[1]["p50"] = -1.0
    assert any("negative" in m for m in validate_contract(rows))
    rows = _good_rows(); rows[2]["district"] = ""
    assert any("blank" in m for m in validate_contract(rows))

def test_validator_catches_missing_column_and_wrong_district_count():
    rows = _good_rows()
    del rows[0]["status"]
    assert any("missing column" in m for m in validate_contract(rows))
    rows = [r for r in _good_rows() if r["district"] != "D03"]
    assert any("25 districts" in m for m in validate_contract(rows))

def test_roundtrip_file_and_consumer_refuses_broken_file(tmp_path):
    f = tmp_path / "forecast.csv"
    write_contract(_good_rows(), str(f))
    back = read_contract(str(f))
    assert len(back) == len(_good_rows()) and set(back[0]) == set(COLUMNS)
    bad = _good_rows(); bad[0]["p50"] = -5.0
    f2 = tmp_path / "bad.csv"
    write_contract(bad, str(f2))
    with pytest.raises(ValueError):
        read_contract(str(f2))