import csv
from app.map_view import CLASSES, build_map, ds_values, load_crosswalk, load_geo, shape_names, view_for

GEO = "data/external/geo/lka_adm3_simple.geojson"
CW = "data/external/geo/ds_crosswalk.csv"
CENSUS = "data/external/census2024_ds_population.csv"

def _town(year=2030, scenario="baseline"):
    """One synthetic row per census DS and class, every value 1.0, so sums are easy to check."""
    with open(CENSUS, newline="", encoding="utf-8") as f:
        ds = [(r["district_key"], r["ds_name"]) for r in csv.DictReader(f)]
    return [dict(district=d, ds_name=n, year=year, scenario=scenario, vehicle_class=c, new_evs=1.0) for d, n in ds for c in CLASSES]

def test_every_census_area_has_a_shape_and_every_shape_is_used():
    geo, cw = load_geo(GEO), load_crosswalk(CW)
    names = shape_names(geo)
    assert len(names) == 339 and len(cw) == 340
    assert set(cw.values()) == set(names)                 # no census area points at a missing shape, no shape is left out

def test_totals_survive_the_join():
    cw = load_crosswalk(CW)
    town = _town()
    v = ds_values(town, cw, 2030, "baseline")
    assert len(v) == 339
    assert abs(sum(x["total"] for x in v.values()) - len(town)) < 1e-9     # nothing lost, nothing double counted

def test_kalmunai_shape_holds_both_census_areas():
    cw = load_crosswalk(CW)
    v = ds_values(_town(), cw, 2030, "baseline")
    assert cw[("AMPARA", "Kalmunai")] == cw[("AMPARA", "Kalmunai North Sub")]
    assert v[cw[("AMPARA", "Kalmunai")]]["total"] == 8.0   # 2 areas x 4 classes x 1.0

def test_other_year_or_scenario_is_ignored():
    cw = load_crosswalk(CW)
    assert ds_values(_town(2031, "slow"), cw, 2030, "baseline") == {}

def test_map_figure_and_zoom():
    geo, cw = load_geo(GEO), load_crosswalk(CW)
    names = shape_names(geo)
    v = ds_values(_town(), cw, 2030, "baseline")
    colombo = {c for (d, _), c in cw.items() if d == "COLOMBO"}
    sel = cw[("COLOMBO", "Kaduwela")]
    assert len(list(build_map(geo, v, names, colombo, sel, False).data)) == 2      # shapes + red outline
    assert len(list(build_map(geo, v, names, colombo, None, False).data)) == 1
    centre, zoom = view_for(geo, colombo)
    assert 6.5 < centre["lat"] < 7.2 and 79.7 < centre["lon"] < 80.4 and zoom > 8
    assert view_for(geo, set(names))[1] < 7
