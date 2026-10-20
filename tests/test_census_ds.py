import os
import pytest
from src.data.census_ds import aggregate_ds, check_rows, write_ds, read_ds, AGE_KEYS

def _gn(district="Colombo", dcode=11, ds="Kaduwela", scode=7, total=100.0, male=60.0, female=40.0, ages=(20.0, 60.0, 5.0, 15.0)):
    return {"district_code": dcode, "district": district, "ds_code": scode, "ds_name": ds, "total": total, "male": male,
            "female": female, **dict(zip(AGE_KEYS, ages))}

def test_aggregate_sums_gn_rows_into_ds_and_shares_add_to_one():
    rows = [_gn(total=100), _gn(total=300, male=100, female=200, ages=(50, 200, 20, 30)),
            _gn(ds="Homagama", scode=8, total=600, male=300, female=300, ages=(100, 400, 40, 60))]
    ds = aggregate_ds(rows)
    kad = [r for r in ds if r["ds_name"] == "Kaduwela"][0]
    assert kad["population"] == 400 and kad["gn_count"] == 2 and kad["district_key"] == "COLOMBO"
    assert abs(sum(r["pop_share_in_district"] for r in ds) - 1.0) < 1e-12
    assert abs(kad["pop_share_in_district"] - 0.4) < 1e-12

def test_same_ds_code_in_two_districts_stays_separate():
    ds = aggregate_ds([_gn(), _gn(district="Gampaha", dcode=12, ds="Other", scode=7)])
    assert len(ds) == 2

def test_check_rows_flags_rows_that_do_not_add_up():
    assert check_rows([_gn()]) == []
    assert len(check_rows([_gn(male=10.0)])) == 1
    assert len(check_rows([_gn(ages=(1.0, 1.0, 1.0, 1.0))])) == 1

def test_roundtrip(tmp_path):
    f = tmp_path / "ds.csv"
    ds = aggregate_ds([_gn(), _gn(ds="Homagama", scode=8, total=200, male=100, female=100, ages=(40, 120, 10, 30))])
    write_ds(ds, str(f))
    back = read_ds(str(f))
    assert [r["ds_name"] for r in back] == [r["ds_name"] for r in ds] and back[0]["population"] == ds[0]["population"]

REAL = "data/external/census2024_ds_population.csv"

@pytest.mark.skipif(not os.path.isfile(REAL), reason="derived census file not present")
def test_real_file_matches_published_totals():
    ds = read_ds(REAL)
    assert len(ds) == 340 and len({r["district_key"] for r in ds}) == 25
    assert sum(r["population"] for r in ds) == 21781800                      # Census 2024 provisional national total
    colombo = sum(r["population"] for r in ds if r["district_key"] == "COLOMBO")
    assert colombo == 2375415                                                # matches table A1 (districts)
    names = {r["ds_name"] for r in ds if r["district_key"] == "COLOMBO"}
    assert {"Kaduwela", "Kolonnawa", "Homagama"} <= names
