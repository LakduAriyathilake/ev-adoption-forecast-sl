import math
import pytest
from src.models.scenarios import (stage_pace, share_path, build_rows, validate_scenarios, write_scenarios, read_scenarios,
                                  district_shares, SCENARIOS, FIRST_YEAR, LAST_YEAR)

def _trans(n_per_stage=30):
    """Synthetic analogues: logit steps shrink as the logit rises, with a spread across countries."""
    rows = []
    for k in range(-60, 40):
        z = k / 10.0
        for j in range(n_per_stage // 10 + 1):
            rows.append({"z_now": z, "dz": 0.6 - 0.12 * z + 0.05 * (j - 1)})
    return rows

def _shares():
    return district_shares({f"D{i:02d}": float(i + 1) for i in range(25)})

def _rows():
    return build_rows(11.8, 2026, 600000.0, _trans(), _shares())

def test_pace_quantiles_are_ordered():
    p = stage_pace(_trans(), -2.0)
    assert p["slow"] <= p["baseline"] <= p["accelerated"]

def test_scenario_paths_are_ordered_and_between_0_and_100():
    paths = {sc: share_path(11.8, _trans(), sc, 11) for sc in SCENARIOS}
    for i in range(11):
        assert paths["slow"][i] <= paths["baseline"][i] <= paths["accelerated"][i]
    assert all(0 < v < 100 for p in paths.values() for v in p)

def test_pace_uses_only_the_analogues_passed_in():
    t = _trans()
    other = [dict(r, dz=r["dz"] + 5) for r in t]
    assert stage_pace(t, 0.0)["baseline"] != stage_pace(other, 0.0)["baseline"]

def test_too_few_analogues_is_refused():
    with pytest.raises(ValueError):
        stage_pace([{"z_now": 0.0, "dz": 0.3}] * 5, 0.0)

def test_good_file_validates_and_has_expected_shape():
    rows = _rows()
    assert validate_scenarios(rows) == []
    assert len(rows) == 11 * 3 * 26
    assert min(r["year"] for r in rows) == FIRST_YEAR and max(r["year"] for r in rows) == LAST_YEAR

def test_district_values_sum_to_national_and_cumulative_adds_up():
    rows = _rows()
    nat = [r for r in rows if r["level"] == "national" and r["scenario"] == "baseline"]
    assert abs(nat[-1]["cum_new_evs_from_2027"] - sum(r["new_evs"] for r in nat)) < 1e-6
    d = sum(r["new_evs"] for r in rows if r["level"] == "district" and r["scenario"] == "baseline" and r["year"] == 2030)
    n = [r["new_evs"] for r in nat if r["year"] == 2030][0]
    assert abs(d - n) < 1e-6 * n

def test_validator_catches_problems():
    rows = _rows(); rows[0]["new_evs"] = -1.0
    assert any("negative" in m for m in validate_scenarios(rows))
    rows = _rows(); rows[1]["national_ev_share_pct"] = 120.0
    assert any("between 0 and 100" in m for m in validate_scenarios(rows))
    rows = [r for r in _rows() if r["scenario"] != "slow"]
    assert any("scenarios must be" in m for m in validate_scenarios(rows))
    rows = [r for r in _rows() if r["year"] != 2030]
    assert any("without gaps" in m for m in validate_scenarios(rows))

def test_roundtrip_and_consumer_refuses_broken_file(tmp_path):
    f = tmp_path / "s.csv"
    write_scenarios(_rows(), str(f))
    assert len(read_scenarios(str(f))) == len(_rows())
    bad = _rows(); bad[0]["new_evs"] = float("nan")
    g = tmp_path / "bad.csv"
    write_scenarios(bad, str(g))
    with pytest.raises(ValueError):
        read_scenarios(str(g))