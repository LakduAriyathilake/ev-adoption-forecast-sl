import math
import pytest
from src.models.town_split import (class_mix, split_rows, validate_split, write_split, read_split, rank_corr, CLASSES,
                                   population_only_check)

def _counts():
    return {"A": {"car": 300.0, "motorcycle": 600.0, "three_wheeler": 50.0, "other": 50.0},
            "B": {"car": 5.0, "motorcycle": 90.0, "three_wheeler": 3.0, "other": 2.0}}          # B too small

def _ds():
    rows = []
    for dist, pops in (("A", [100, 300, 600]), ("B", [50, 50])):
        tot = sum(pops)
        for i, p in enumerate(pops):
            rows.append({"district_key": dist, "ds_name": f"{dist}{i}", "ds_code": i, "population": p, "pop_share_in_district": round(p / tot, 6)})
    return rows

def _scen():
    return [{"level": "district", "district": d, "scenario": sc, "year": y, "new_evs": v, "run_id": "r"}
            for d, v in (("A", 1000.0), ("B", 200.0)) for sc in ("slow", "baseline") for y in (2027, 2028)]

def test_mix_sums_to_one_and_small_district_falls_back():
    mix = class_mix(_counts(), min_pool=300)
    for _, (m, _) in mix.items():
        assert abs(sum(m.values()) - 1.0) < 1e-12
    assert mix["A"][1] == "district" and mix["B"][1] == "national_fallback"
    assert abs(mix["A"][0]["car"] - 0.3) < 1e-12
    nat_car = (300 + 5) / (1000 + 100)
    assert abs(mix["B"][0]["car"] - nat_car) < 1e-12

def test_split_preserves_district_totals_and_validates():
    rows = split_rows(_scen(), _ds(), class_mix(_counts(), 300))
    assert validate_split(rows, _scen(), 5) == []
    a = sum(r["new_evs"] for r in rows if r["district"] == "A" and r["scenario"] == "slow" and r["year"] == 2027)
    assert abs(a - 1000.0) < 1e-9

def test_split_follows_mix_and_population_share():
    rows = split_rows(_scen(), _ds(), class_mix(_counts(), 300))
    r = [x for x in rows if x["ds_name"] == "A2" and x["vehicle_class"] == "car" and x["scenario"] == "slow" and x["year"] == 2027][0]
    assert abs(r["new_evs"] - 1000.0 * 0.3 * 0.6) < 1e-9

def test_rounded_stored_shares_do_not_break_totals():
    ds = [{"district_key": "A", "ds_name": f"A{i}", "ds_code": i, "population": 1000 + 7 * i, "pop_share_in_district": round((1000 + 7 * i) / sum(1000 + 7 * k for k in range(13)), 6)} for i in range(13)]
    scen = [{"level": "district", "district": "A", "scenario": "baseline", "year": 2030, "new_evs": 50000.0, "run_id": "r"}]
    rows = split_rows(scen, ds, class_mix({"A": {c: 100.0 for c in CLASSES}}, 300))
    assert abs(sum(r["new_evs"] for r in rows) - 50000.0) < 1e-9

def test_missing_district_is_refused():
    with pytest.raises(ValueError):
        split_rows(_scen(), [r for r in _ds() if r["district_key"] != "B"], class_mix(_counts(), 300))

def test_validator_catches_problems():
    rows = split_rows(_scen(), _ds(), class_mix(_counts(), 300))
    rows[0]["new_evs"] = 5000.0
    assert validate_split(rows, _scen(), 5)
    rows = split_rows(_scen(), _ds(), class_mix(_counts(), 300)); rows[1]["new_evs"] = -1.0
    assert any("negative" in m for m in validate_split(rows, _scen(), 5))

def test_roundtrip(tmp_path):
    rows = split_rows(_scen(), _ds(), class_mix(_counts(), 300))
    f = tmp_path / "t.csv"; write_split(rows, str(f))
    back = read_split(str(f))
    assert len(back) == len(rows) and abs(back[0]["new_evs"] - rows[0]["new_evs"]) < 1e-3

def test_rank_corr_known_values():
    assert abs(rank_corr([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(rank_corr([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert math.isnan(rank_corr([1, 1, 1], [1, 2, 3]))
    assert abs(rank_corr([1, 2, 2, 4], [1, 2, 3, 4]) - 0.9486832980505138) < 1e-9          # ties get average ranks

def test_population_only_check_perfect_when_shares_match():
    counts = {"A": {c: 10.0 for c in CLASSES}, "B": {c: 30.0 for c in CLASSES}}
    ds = [{"district_key": "A", "population": 100}, {"district_key": "B", "population": 300}]
    rho, gap, _ = population_only_check(counts, ds, "car")
    assert abs(gap) < 1e-9
