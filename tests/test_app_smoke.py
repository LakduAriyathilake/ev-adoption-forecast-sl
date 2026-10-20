import os
import re
from pathlib import Path
import pytest

pytest.importorskip("streamlit")
RAW = os.environ.get("EV_RAW_DIR", "data/raw")
FILES = ["data/processed/forecast_v0.csv", "data/processed/scenarios_v0.csv", "data/processed/town_estimates_v0.csv",
         "data/external/census2024_ds_population.csv"]
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
HAVE = os.path.isdir(RAW) and all(os.path.isfile(f) for f in FILES)

def kpi(at, key):
    """Number shown in the card with data-kpi=<key> (cards are HTML, so read it from the rendered markdown)."""
    for m in at.markdown:
        found = re.search(r'data-kpi="' + key + r'".*?<b[^>]*>([^<]+)</b>', m.value)
        if found:
            return found.group(1)
    return None

def num(text):
    return float(re.sub(r"[^0-9.\-]", "", text))

@pytest.mark.skipif(not HAVE, reason="private DMT data or the processed files are not available")
def test_app_runs_and_controls_change_numbers():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("EV Adoption Forecast" in t.value for t in at.title)
    base = num(kpi(at, "district_evs"))
    at.selectbox(key="lt_scenario").select("slow").run()
    assert not at.exception and num(kpi(at, "district_evs")) < base
    at.selectbox(key="lt_scenario").select("accelerated").run()
    assert num(kpi(at, "district_evs")) > base
    before = kpi(at, "st_evs")
    at.selectbox(key="st_district").select("KANDY").run()
    assert not at.exception and kpi(at, "st_evs") != before
    at.selectbox(key="lt_district").select("KANDY").run()
    assert not at.exception

@pytest.mark.skipif(not HAVE, reason="data files not available")
def test_area_tiles_show_the_town_numbers_from_the_file():
    from streamlit.testing.v1 import AppTest
    from src.models.town_split import read_split
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.selectbox(key="lt_district").select("COLOMBO").run()
    at.selectbox(key="lt_area").select("Kaduwela").run()
    assert not at.exception
    town = read_split("data/processed/town_estimates_v0.csv")
    for cls in ("car", "motorcycle"):
        expected = sum(r["new_evs"] for r in town if r["district"] == "COLOMBO" and r["ds_name"] == "Kaduwela"
                       and r["scenario"] == "baseline" and r["year"] == 2030 and r["vehicle_class"] == cls)
        assert kpi(at, cls) == f"{expected:,.0f}"
    at.selectbox(key="lt_area").select("All areas").run()
    assert not at.exception and kpi(at, "car") is None