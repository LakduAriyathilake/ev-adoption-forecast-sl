import os
from pathlib import Path
import pytest

pytest.importorskip("streamlit")
RAW = os.environ.get("EV_RAW_DIR", "data/raw")
FORECAST = "data/processed/forecast_v0.csv"
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

@pytest.mark.skipif(not (os.path.isdir(RAW) and os.path.isfile(FORECAST)), reason="private DMT data or forecast file not available")
def test_app_runs_and_district_switch_works():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Dashboard" in t.value for t in at.title)
    assert len(at.metric) == 4
    before = at.metric[0].value
    at.selectbox[0].select("KANDY").run()
    assert not at.exception
    assert at.metric[0].value != before