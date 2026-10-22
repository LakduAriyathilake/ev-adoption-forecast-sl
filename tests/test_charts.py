from app.charts import model_bars, month_chart, rank_bars, scenario_chart
from app.model_check import LABEL, LEARNED

def _nat():
    rows = []
    for sc, k in (("slow", 1.0), ("baseline", 1.5), ("accelerated", 2.0)):
        for y in range(2027, 2038):
            rows.append(dict(level="national", scenario=sc, year=y, new_evs=1000.0 * k * (y - 2025)))
    return rows

def test_scenario_chart_has_every_line_and_a_marker():
    hist = {y: 100.0 * (y - 2018) for y in range(2019, 2027)}
    d = scenario_chart(hist, _nat(), 2030, "baseline", False).to_dict()
    names = {t.get("name") for t in d["data"] if t.get("name")}
    assert {"Observed", "Slow", "Baseline", "Accelerated"} <= names
    assert scenario_chart(hist, _nat(), 2030, "baseline", True).to_dict()["layout"]["yaxis"]["type"] == "log"

def test_month_chart_draws_forecast_band_and_events():
    periods = [f"2025-{m:02d}" for m in range(1, 13)]
    nat = [dict(period=f"2026-{m:02d}", p10=5.0, p50=10.0, p90=15.0) for m in range(1, 5)]
    ev = [dict(event_id="E01", date="2025-06-15")]
    d = month_chart([10.0] * 12, periods, [True] * 11 + [False], nat, "2026-04", ev).to_dict()
    assert len(d["layout"]["annotations"]) == 1 and "Forecast" in [t.get("name") for t in d["data"]]
    d2 = month_chart([10.0] * 12, periods, [True] * 12, nat, "2026-04", []).to_dict()
    assert len(d2["layout"].get("annotations", [])) == 0

def test_bars_highlight_the_selected_name():
    bar = rank_bars(["A", "B", "C"], [3.0, 2.0, 1.0], "B").to_dict()["data"][0]
    colours = dict(zip(bar["y"], bar["marker"]["color"]))
    assert colours["B"] != colours["A"] and colours["A"] == colours["C"]
    mb = model_bars({"naive": (1.0, 5), "ridge_lag": (2.0, 5)}, LABEL, LEARNED, "T").to_dict()["data"][0]
    assert len(mb["y"]) == 2
