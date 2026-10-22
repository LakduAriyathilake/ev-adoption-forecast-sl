"""
EV adoption forecast dashboard (prototype v0).

Streamlit only DISPLAYS numbers; the scripts below make them first:
  data/processed/forecast_v0.csv            python -m src.models.forecast_v0 --raw-dir data/raw
  data/processed/scenarios_v0.csv           python -m src.models.scenarios --raw-dir data/raw
  data/processed/town_estimates_v0.csv      python -m src.models.town_split --raw-dir data/raw
  data/processed/trained_vs_baselines.csv   python -m src.models.trained_v0 --raw-dir data/raw

Look and feel lives in app/theme.py, charts in app/charts.py, the map in app/map_view.py.
Run from the repo root:  streamlit run app/streamlit_app.py
"""
import csv
import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so 'src' and 'app' imports work when launched by streamlit
from app.charts import month_chart, model_bars, rank_bars, scenario_chart  # noqa: E402
from app.map_view import build_map, ds_values, load_crosswalk, load_geo, shape_names  # noqa: E402
from app.model_check import LABEL, LEARNED, mae_table  # noqa: E402
from app.theme import (CLASSES, CSS, SC_ICON, SC_LABEL, card_title, chips_html, header, html, kpi, kpi_donut,  # noqa: E402
                       kpi_range, mix_bar, tiles, top_list)
from src.data.census_ds import read_ds  # noqa: E402
from src.models.forecast_v0 import load_district_totals, load_national_monthly, read_contract  # noqa: E402
from src.models.scenarios import read_scenarios  # noqa: E402
from src.models.town_split import read_split  # noqa: E402

POLICY_CSV = "data/external/policy_events_lk.csv"
CENSUS_CSV = "data/external/census2024_ds_population.csv"
GEO_DIR = "data/external/geo"
TRAINED_CSV = "data/processed/trained_vs_baselines.csv"
Row = dict[str, Any]
PLOT = {"displayModeBar": False}

def nice(name: str) -> str:
    return name.title()

@st.cache_data
def load_history(raw_dir: str) -> tuple[list[float], list[str], list[bool]]:
    return load_national_monthly(raw_dir, start="2019-01")

@st.cache_data
def load_events(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]

@st.cache_data
def annual_history(raw_dir: str) -> dict[int, float]:
    return {y: sum(load_district_totals(raw_dir, y).values()) for y in range(2019, 2027)}

@st.cache_data
def load_town(path: str) -> list[Row]:
    return read_split(path)

@st.cache_data
def load_census(path: str) -> list[Row]:
    return read_ds(path)

@st.cache_data
def load_map_files() -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    return load_geo(f"{GEO_DIR}/lka_adm3_simple.geojson"), load_crosswalk(f"{GEO_DIR}/ds_crosswalk.csv")

def top_with_selected(items: list[tuple[str, float]], selected: str, n: int = 8) -> tuple[list[str], list[float]]:
    items = sorted(items, key=lambda t: -t[1])
    shown = items[:n]
    if selected not in [d for d, _ in shown]:
        shown += [t for t in items if t[0] == selected]
    return [nice(d) for d, _ in shown], [v for _, v in shown]

# ---------------------------------------------------------------- tab 1: 2027-2037
def area_card(town: list[Row], district: str, area: str, year: int, scenario: str) -> None:
    rows = [r for r in town if r["district"] == district and r["scenario"] == scenario and r["year"] == year]
    chosen = rows if area == "All areas" else [r for r in rows if r["ds_name"] == area]
    by: dict[str, float] = {c: float(sum(r["new_evs"] for r in chosen if r["vehicle_class"] == c)) for c in CLASSES}
    pop = [r for r in load_census(CENSUS_CSV) if r["district_key"] == district and (area == "All areas" or r["ds_name"] == area)]
    h1, h2 = st.columns([5, 1])
    with h1:
        card_title("📍", area if area != "All areas" else f"{nice(district)} · all areas", f"{year} · {SC_ICON[scenario]} {SC_LABEL[scenario]}")
    with h2.popover("ⓘ"):
        people = f"{sum(int(r['population']) for r in pop):,} people (Census 2024, provisional)" if pop else "the district population"
        st.markdown("**Estimate, not an observation.** District forecast × the district's vehicle mix × this area's share of "
                    f"{people}.\n\nPopulation is the only weight, so wealthier urban areas may hold more cars than shown. "
                    "DMT records where a vehicle is registered, not where the owner lives.")
    tiles(by)
    mix_bar(by)
    per_area: dict[str, float] = {}
    for r in rows:
        per_area[r["ds_name"]] = per_area.get(r["ds_name"], 0.0) + float(r["new_evs"])
    ranked = sorted(per_area.items(), key=lambda t: -t[1])
    shown = ranked[:5]
    if area != "All areas" and area not in [n for n, _ in shown]:
        shown += [t for t in ranked if t[0] == area]
    top_list(shown, None if area == "All areas" else area, f"Top areas in {nice(district)}")

def map_card(town: list[Row], district: str, area: str, year: int, scenario: str) -> None:
    geo, cw = load_map_files()
    card_title("🗺️", "Where", f"{nice(district)} · {year}")
    t1, t2, _ = st.columns([1.3, 1.5, 2])
    street = t1.toggle("Street map", value=False, key="lt_street")
    whole = t2.toggle("Whole island", value=False, key="lt_whole")
    focus = {code for (d, _), code in cw.items() if d == district}
    selected = cw.get((district, area)) if area != "All areas" else None
    fig = build_map(geo, ds_values(town, cw, year, scenario), shape_names(geo), focus, selected, whole, dark=not street, height=520)
    st.plotly_chart(fig, width="stretch", config={**PLOT, "scrollZoom": True})
    st.markdown('<div class="cap">Darker or brighter = more new EVs (estimate, all vehicle types). White or red outline = selected area. '
                'Boundaries: OCHA / Survey Department of Sri Lanka (2022, simplified). Map data © OpenStreetMap contributors'
                + ('' if street else ', style © CARTO') + '.</div>', unsafe_allow_html=True)

def long_term(scen: list[Row], raw_dir: str, town: list[Row] | None) -> None:
    nat = [r for r in scen if r["level"] == "national"]
    dist_rows = [r for r in scen if r["level"] == "district"]
    districts = sorted({r["district"] for r in dist_rows})
    years = sorted({r["year"] for r in nat})
    hist = annual_history(raw_dir)
    ann26 = hist[2026] * 12.0 / 8.0

    with st.container(key="card_controls"):
        c1, c2, c3, c4 = st.columns([1.3, 1.3, 0.8, 1.7])
        district = c1.selectbox("📍 District", districts, index=districts.index("COLOMBO") if "COLOMBO" in districts else 0,
                                format_func=nice, key="lt_district") or districts[0]
        ds_names = sorted({r["ds_name"] for r in town if r["district"] == district}) if town else []
        area = c2.selectbox("🏘️ Area", ["All areas"] + ds_names, key="lt_area", disabled=not town) or "All areas"
        year = c3.selectbox("📅 Year", years, index=years.index(2030) if 2030 in years else 0, key="lt_year") or years[0]
        scenario = c4.segmented_control("🎚️ Scenario", list(SC_LABEL), default="baseline",
                                        format_func=lambda k: f"{SC_ICON[k]} {SC_LABEL[k]}", key="lt_scenario") or "baseline"

    def d_val(sc: str, y: int) -> float:
        return [r["new_evs"] for r in dist_rows if r["district"] == district and r["scenario"] == sc and r["year"] == y][0]
    def n_val(sc: str, y: int) -> float:
        return [r["new_evs"] for r in nat if r["scenario"] == sc and r["year"] == y][0]
    prev = ann26 if year == years[0] else n_val(scenario, year - 1)
    growth = n_val(scenario, year) / prev - 1
    share = [r["national_ev_share_pct"] for r in nat if r["scenario"] == scenario and r["year"] == year][0]

    html('<div style="height:14px"></div>')
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("district_evs", '<span class="ico">⚡<span class="leaf">🍃</span></span>', f"{d_val(scenario, year):,.0f}", f"new EVs · {nice(district)} · {year}")
    with k2:
        kpi("growth", f'<span class="ico">{"📈" if growth >= 0 else "📉"}</span>', f"{growth:+.0%}", "vs last year · national")
    with k3:
        kpi_range("range", d_val("slow", year), d_val(scenario, year), d_val("accelerated", year), f"{SC_ICON[scenario]}")
    with k4:
        kpi_donut("share", share, "of new vehicles are EVs")
    html('<div style="height:6px"></div>')

    if town:
        m, a = st.columns([3, 2], gap="medium")
        with m, st.container(key="card_map"):
            map_card(town, district, area, year, scenario)
        with a, st.container(key="card_area"):
            area_card(town, district, area, year, scenario)
    else:
        st.info("Area list is off. Create town estimates:  python -m src.models.town_split --raw-dir data/raw")

    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(key="card_chart"):
        card_title("📊", "National: history and scenarios", "new EVs a year")
        log = st.toggle("Log scale", value=False, key="lt_log")
        st.plotly_chart(scenario_chart(hist, nat, year, scenario, log), width="stretch", config=PLOT)
    with right, st.container(key="card_rank"):
        card_title("🎚️", nice(district), "new EVs a year")
        shown_years = [y for y in (2027, 2030, 2033, 2035, 2037) if y in years]
        st.dataframe([{"Year": y, **{f"{SC_ICON[sc]} {SC_LABEL[sc]}": round(d_val(sc, y)) for sc in SC_LABEL}} for y in shown_years],
                     hide_index=True, width="stretch")
        card_title("🏆", "Top districts", f"{year}")
        names, vals = top_with_selected([(r["district"], r["new_evs"]) for r in dist_rows if r["year"] == year and r["scenario"] == scenario], district)
        st.plotly_chart(rank_bars(names, vals, nice(district), height=300), width="stretch", config=PLOT)

    with st.expander("ⓘ How these scenarios are made"):
        n_assumed = n_val(scenario, year) / (share / 100.0)
        st.markdown("**Method.** Stage-matched pace: what other countries' EV shares did from a similar starting share "
                    "(🐢 slow = 25th percentile, 📈 baseline = median, 🚀 accelerated = 75th).\n\n"
                    f"**Volume.** Total new registrations held flat at about {round(n_assumed, -3):,.0f} a year (the 2026 pace).\n\n"
                    "**Limits.** Country data are cars only, Sri Lankan EVs are mostly motorcycles; the same pace is held for all 11 years; "
                    "district numbers use 2025 shares.\n\n"
                    f"Status `{scen[0]['status']}` · run {scen[0]['run_id']}")

# ---------------------------------------------------------------- tab 2: next 4 months
def model_check() -> None:
    with st.container(key="card_model"):
        card_title("🧪", "Model check", "average miss in new EVs, next month · lower is better")
        if not Path(TRAINED_CSV).exists():
            st.info("Not made yet. Run:  python -m src.models.trained_v0 --raw-dir data/raw")
            return
        tab = mae_table(TRAINED_CSV)
        html(chips_html([("🔬 Walk-forward, no future data", "ok"), ("🟦 Simple rules", ""), ("🟧 Learned models", "warn")]))
        c1, c2 = st.columns(2)
        for col, key, title in ((c1, "development", "Development"), (c2, "holdout", "2026 holdout")):
            n = next(iter(tab[key].values()))[1]
            with col:
                st.plotly_chart(model_bars(tab[key], LABEL, LEARNED, f"{title} ({n} months)"), width="stretch", config=PLOT)
        best = min(tab["development"].items(), key=lambda t: t[1][0])[0]
        st.markdown(f'<div class="cap">Best in development: {LABEL[best]}. The learned models did not clearly beat the simple rules '
                    '(very little high-volume data), and the holdout has only a few months.</div>', unsafe_allow_html=True)

def short_term(rows: list[Row], raw_dir: str) -> None:
    nat = sorted((r for r in rows if r["level"] == "national"), key=lambda r: r["horizon_months"])
    dist_rows = [r for r in rows if r["level"] == "district"]
    districts = sorted({r["district"] for r in dist_rows})
    months = [r["period"] for r in nat]
    values, periods, observed = load_history(raw_dir)

    with st.container(key="card_controls_st"):
        c1, c2 = st.columns(2)
        district = c1.selectbox("📍 District", districts, index=districts.index("COLOMBO") if "COLOMBO" in districts else 0,
                                format_func=nice, key="st_district") or districts[0]
        month = c2.selectbox("📅 Month", months, index=len(months) - 1, key="st_month") or months[-1]

    pick = [r for r in dist_rows if r["district"] == district and r["period"] == month][0]
    change = [r for r in nat if r["period"] == month][0]["p50"] / values[-1] - 1
    html('<div style="height:14px"></div>')
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("st_evs", '<span class="ico">⚡<span class="leaf">🍃</span></span>', f"{pick['p50']:,.0f}", f"new EVs · {nice(district)} · {month}")
    with k2:
        kpi("st_change", f'<span class="ico">{"📈" if change >= 0 else "📉"}</span>', f"{change:+.0%}", f"vs {periods[-1]} · national")
    with k3:
        kpi_range("st_range", pick["p10"], pick["p50"], pick["p90"], "80%")
    html('<div style="height:6px"></div>')

    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(key="card_chart_st"):
        card_title("📈", "National: last months and the next 4", "new EVs a month")
        events = st.toggle("🏛️ Policy events", value=True, key="st_events")
        ev = [e for e in load_events(POLICY_CSV) if e["date_precision"] in ("day", "month")] if events else []
        st.plotly_chart(month_chart(values, periods, observed, nat, month, ev), width="stretch", config=PLOT)
    with right, st.container(key="card_rank_st"):
        card_title("📅", nice(district), "next 4 months")
        st.dataframe([{"Month": r["period"], "low": round(r["p10"]), "mid": round(r["p50"]), "high": round(r["p90"])}
                      for r in sorted((r for r in dist_rows if r["district"] == district), key=lambda r: r["horizon_months"])],
                     hide_index=True, width="stretch")
        card_title("🏆", "Top districts", month)
        names, vals = top_with_selected([(r["district"], r["p50"]) for r in dist_rows if r["period"] == month], district)
        st.plotly_chart(rank_bars(names, vals, nice(district), height=300), width="stretch", config=PLOT)
    model_check()
    with st.expander("ⓘ How this is made"):
        st.markdown(f"Method `{pick['method']}`: last observed month × growth ratios seen since 2023. District = national × its 2025 share. "
                    f"Status `{pick['status']}` · run {pick['run_id']} · contract v{pick['contract_version']}. "
                    "Interval coverage has not been measured yet.")

def main() -> None:
    st.set_page_config(page_title="EV Adoption Forecast", page_icon="🍃", layout="wide")
    html(CSS)
    header([("🧪 Prototype v0", "warn"), ("🔮 If-paths, not predictions", ""), ("📍 District → Town", "ok")])
    with st.sidebar:
        st.header("Inputs")
        raw_dir = st.text_input("DMT raw folder", "data/raw")
        forecast_path = st.text_input("Monthly forecast file", "data/processed/forecast_v0.csv")
        scenario_path = st.text_input("Scenario file", "data/processed/scenarios_v0.csv")
        town_path = st.text_input("Town estimates file", "data/processed/town_estimates_v0.csv")
    if not Path(raw_dir).exists():
        st.error(f"Folder not found: {raw_dir}. Run streamlit from the repo root.")
        return
    tab_long, tab_short = st.tabs(["🔮 2027–2037", "📅 Next 4 months"])
    with tab_long:
        try:
            town = load_town(town_path) if Path(town_path).exists() and Path(GEO_DIR, "lka_adm3_simple.geojson").exists() else None
            long_term(read_scenarios(scenario_path), raw_dir, town)
        except (FileNotFoundError, ValueError) as err:
            st.error(f"Scenario file not usable: {err}")
            st.code("python -m src.models.scenarios --raw-dir data/raw")
    with tab_short:
        try:
            short_term(read_contract(forecast_path), raw_dir)
        except (FileNotFoundError, ValueError) as err:
            st.error(f"Forecast file not usable: {err}")
            st.code("python -m src.models.forecast_v0 --raw-dir data/raw")

main()
