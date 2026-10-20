import os
from pathlib import Path
import pytest

pytest.importorskip("streamlit")
RAW = os.environ.get("EV_RAW_DIR", "data/raw")
FILES = ["data/processed/forecast_v0.csv", "data/processed/scenarios_v0.csv"]
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

@pytest.mark.skipif(not (os.path.isdir(RAW) and all(os.path.isfile(f) for f in FILES)),
                    reason="private DMT data or the processed forecast files are not available")
def test_app_runs_and_controls_change_numbers():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Dashboard" in t.value for t in at.title)
    assert len(at.metric) == 8                                   # 4 cards in each of the 2 tabs
    long_before = at.metric[0].value
    at.selectbox(key="lt_scenario").select("slow").run()
    assert not at.exception
    assert int(at.metric[0].value.replace(",", "")) < int(long_before.replace(",", ""))
    short_before = at.metric[4].value
    at.selectbox(key="st_district").select("KANDY").run()
    assert not at.exception
    assert at.metric[4].value != short_before
    at.selectbox(key="lt_district").select("KANDY").run()
    assert not at.exception
