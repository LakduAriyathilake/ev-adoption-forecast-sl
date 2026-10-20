"""
Prototype v0 (PP1) - Component 1: Adaptive EV Adoption Forecasting, laid out like the proposal mock-up.

Every number on the page comes from real data (DMT history) or from the forecast file that honours
contract v0 (read through read_contract, which refuses a broken file). Anything not built yet says so.

Run from the repo root:   streamlit run app/streamlit_app.py
Needs: data/raw (private DMT CSVs) and data/processed/forecast_v0.csv
       (create it with: python -m src.models.forecast_v0 --raw-dir data/raw)
"""
import csv
import datetime
import sys
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so 'src' imports work when launched by streamlit
from src.models.forecast_v0 import RATIO_FROM, load_district_totals, load_national_monthly, read_contract  # noqa: E402
from src.models.scenarios import read_scenarios  # noqa: E402

POLICY_CSV = "data/external/policy_events_lk.csv"
Row = dict[str, Any]

def to_num(period: str) -> float:
    return float(mdates.date2num(datetime.datetime(int(period[:4]), int(period[5:7]), 1)))

@st.cache_data
def load_history(raw_dir: str) -> tuple[list[float], list[str], list[bool]]:
    return load_national_monthly(raw_dir, start="2019-01")

@st.cache_data
def load_events(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]

def nice(name: str) -> str:
    return name.title()

def history_chart(values: list[float], periods: list[str], observed: list[bool], nat: list[Row],
                  month: str, show_events: bool, show_fc: bool) -> None:
    x = [to_num(p) for p in periods]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(x, values, color="#1f77b4", lw=1.8, label="Observed (national)")
    gaps = [i for i, ob in enumerate(observed) if not ob]
    ax.scatter([x[i] for i in gaps], [values[i] for i in gaps], color="#999999", s=30, zorder=3, label="Blank month in DMT")
    if show_fc:
        fx = [to_num(r["period"]) for r in nat]
        ax.plot([x[-1]] + fx, [values[-1]] + [r["p50"] for r in nat], color="#d62728", lw=1.8, ls="--", label="Forecast p50 (placeholder)")
        ax.fill_between([x[-1]] + fx, [values[-1]] + [r["p10"] for r in nat], [values[-1]] + [r["p90"] for r in nat],
                        color="#d62728", alpha=0.18, label="p10-p90")
        pick = [r for r in nat if r["period"] == month][0]
        ax.scatter([to_num(month)], [pick["p50"]], color="#d62728", s=45, zorder=4)
    if show_events:
        top = ax.get_ylim()[1]
        for e in load_events(POLICY_CSV):
            if e["date_precision"] in ("day", "month"):
                ex = to_num(e["date"][:7])
                if x[0] <= ex <= x[-1]:
                    ax.axvline(ex, color="#555555", lw=0.8, ls=":")
                    ax.text(ex, top * 0.97, e["event_id"], rotation=90, va="top", ha="right", fontsize=7, color="#333333")
    ax.xaxis_date()
    ax.set_ylabel("New EV registrations / month")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", fontsize=7)
    st.pyplot(fig)

def district_bars(dist_rows: list[Row], month: str, selected: str) -> None:
    this = sorted((r for r in dist_rows if r["period"] == month), key=lambda r: -r["p50"])
    shown = this[:10]
    if selected not in [r["district"] for r in shown]:
        shown = shown + [r for r in this if r["district"] == selected]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    names = [nice(r["district"]) for r in shown][::-1]
    vals = [r["p50"] for r in shown][::-1]
    cols = ["#d62728" if n.upper() == selected else "#1f77b4" for n in names]
    ax.barh(names, vals, color=cols)
    ax.set_xlabel(f"Forecast new EVs, {month} (p50)")
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig)

def short_term(rows: list[Row], raw_dir: str) -> None:
    st.caption("Monthly forecast for the next 4 months, read from the contract v0 file.")
    nat = sorted((r for r in rows if r["level"] == "national"), key=lambda r: r["horizon_months"])
    dist_rows = [r for r in rows if r["level"] == "district"]
    districts = sorted({r["district"] for r in dist_rows})
    months = [r["period"] for r in nat]
    values, periods, observed = load_history(raw_dir)

    c1, c2, c3, c4 = st.columns(4)
    district = c1.selectbox("District", districts, index=districts.index("COLOMBO") if "COLOMBO" in districts else 0, format_func=nice, key="st_district") or districts[0]
    c2.selectbox("Area", ["All areas"], disabled=True, key="st_area", help="Town-level disaggregation is not built yet.")
    month = c3.selectbox("Forecast month", months, index=len(months) - 1, key="st_month") or months[-1]
    c4.selectbox("Scenario", ["Baseline (v0 placeholder)"], disabled=True, key="st_scenario", help="Slow / Accelerated scenarios for 2027-2037 are in the other tab.")

    pick = [r for r in dist_rows if r["district"] == district and r["period"] == month][0]
    nat_pick = [r for r in nat if r["period"] == month][0]
    change = nat_pick["p50"] / values[-1] - 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Forecast new EVs, {nice(district)}", f"{pick['p50']:,.0f}", help="Median (p50) for the chosen month. District = national forecast x its 2025 share.")
    m2.metric(f"Change vs {periods[-1]} (national)", f"{change:+.1%}", help="National median forecast against the last observed month.")
    m3.metric("80% range", f"{pick['p10']:,.0f} - {pick['p90']:,.0f}")
    m4.metric("Interval coverage", "Pending", help="Not validated yet. Real coverage will be measured on the walk-forward backtest.")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Historical and forecast EV adoption (national)")
        o1, o2 = st.columns(2)
        show_events = o1.checkbox("Show policy events", value=True, key="st_events")
        show_fc = o2.checkbox("Show forecast", value=True, key="st_fc")
        history_chart(values, periods, observed, nat, month, show_events, show_fc)
    with right:
        st.subheader(f"Forecast table: {nice(district)}")
        table = [{"Month": r["period"], "p10": round(r["p10"]), "p50 (baseline)": round(r["p50"]), "p90": round(r["p90"])}
                 for r in sorted((r for r in dist_rows if r["district"] == district), key=lambda r: r["horizon_months"])]
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Scenario columns (Slow, Accelerated) will be added when the scenario layer is built. Values are new EV registrations per month.")

    st.subheader("Spatial EV distribution (district level)")
    s1, s2 = st.columns([1, 1])
    with s1:
        district_bars(dist_rows, month, district)
    with s2:
        st.info("District values are the national forecast split by each district's share of 2025 registrations (a static placeholder). "
                "City / town estimates and the map need the spatial disaggregation step, which is not built yet.")

    st.subheader("Why this forecast?")
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(f"**Method**\n\n`{pick['method']}`")
    d2.markdown(f"**Growth ratios learned from**\n\n{RATIO_FROM} onwards")
    d3.markdown(f"**Status**\n\n`{pick['status']}`")
    d4.markdown("**Drivers (income, fuel price...)**\n\nPlanned, not built yet")
    st.caption(f"Run: {pick['run_id']} | contract v{pick['contract_version']}. "
               "The final system will list the most influential drivers and the strategy the adaptive agent chose.")


SC_LABEL = {"slow": "Slow", "baseline": "Baseline", "accelerated": "Accelerated"}
SC_COLOR = {"slow": "#2ca02c", "baseline": "#1f77b4", "accelerated": "#d62728"}

@st.cache_data
def annual_history(raw_dir: str) -> dict[int, float]:
    return {y: sum(load_district_totals(raw_dir, y).values()) for y in range(2019, 2027)}

def long_chart(hist: dict[int, float], nat: list[Row], year: int, scenario: str, log: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    hy = [y for y in sorted(hist) if y <= 2025]
    ax.plot(hy, [hist[y] for y in hy], color="#333333", lw=2, marker="o", label="Observed (DMT, full years)")
    ann26 = hist[2026] * 12.0 / 8.0
    ax.scatter([2026], [ann26], facecolors="none", edgecolors="#333333", s=60, zorder=4, label="2026 (Jan-Aug x 12/8)")
    years = sorted({r["year"] for r in nat})
    by = {sc: [r["new_evs"] for r in sorted((r for r in nat if r["scenario"] == sc), key=lambda r: r["year"])] for sc in SC_LABEL}
    ax.fill_between([2026] + years, [ann26] + by["slow"], [ann26] + by["accelerated"], color="#999999", alpha=0.15, label="Slow to accelerated")
    for sc in SC_LABEL:
        ax.plot([2026] + years, [ann26] + by[sc], color=SC_COLOR[sc], lw=2.4 if sc == scenario else 1.2,
                ls="--", label=f"{SC_LABEL[sc]} (if-path)")
    pick = [r for r in nat if r["scenario"] == scenario and r["year"] == year][0]
    ax.scatter([year], [pick["new_evs"]], color=SC_COLOR[scenario], s=55, zorder=5)
    if log:
        ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("New EV registrations per year (national)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=7)
    st.pyplot(fig)

def long_bars(dist_rows: list[Row], year: int, scenario: str, selected: str) -> None:
    this = sorted((r for r in dist_rows if r["year"] == year and r["scenario"] == scenario), key=lambda r: -r["new_evs"])
    shown = this[:10]
    if selected not in [r["district"] for r in shown]:
        shown = shown + [r for r in this if r["district"] == selected]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    names = [nice(r["district"]) for r in shown][::-1]
    ax.barh(names, [r["new_evs"] for r in shown][::-1], color=["#d62728" if n.upper() == selected else "#1f77b4" for n in names])
    ax.set_xlabel(f"New EVs in {year} ({SC_LABEL[scenario]})")
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig)

def long_term(scen: list[Row], raw_dir: str) -> None:
    nat = [r for r in scen if r["level"] == "national"]
    dist_rows = [r for r in scen if r["level"] == "district"]
    districts = sorted({r["district"] for r in dist_rows})
    years = sorted({r["year"] for r in nat})
    hist = annual_history(raw_dir)
    ann26 = hist[2026] * 12.0 / 8.0

    c1, c2, c3, c4 = st.columns(4)
    district = c1.selectbox("District", districts, index=districts.index("COLOMBO") if "COLOMBO" in districts else 0,
                            format_func=nice, key="lt_district") or districts[0]
    c2.selectbox("Area", ["All areas"], disabled=True, key="lt_area", help="Town / DS-division level is designed but not built yet.")
    year = c3.selectbox("Forecast year", years, index=years.index(2030) if 2030 in years else 0, key="lt_year") or years[0]
    scenario = c4.selectbox("Scenario", list(SC_LABEL), index=1, format_func=lambda k: SC_LABEL[k], key="lt_scenario") or "baseline"

    def d_val(sc: str, y: int) -> float:
        return [r["new_evs"] for r in dist_rows if r["district"] == district and r["scenario"] == sc and r["year"] == y][0]
    def n_val(sc: str, y: int) -> float:
        return [r["new_evs"] for r in nat if r["scenario"] == sc and r["year"] == y][0]
    prev = ann26 if year == years[0] else n_val(scenario, year - 1)
    share = [r["national_ev_share_pct"] for r in nat if r["scenario"] == scenario and r["year"] == year][0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"New EVs, {nice(district)} {year}", f"{d_val(scenario, year):,.0f}", help=f"{SC_LABEL[scenario]} scenario, new registrations in that year.")
    m2.metric("Growth vs prior year (national)", f"{n_val(scenario, year) / prev - 1:+.1%}", help="For 2027 the base is 2026 annualised (Jan-Aug x 12/8).")
    m3.metric("Range across scenarios", f"{d_val('slow', year) / 1000:,.1f}k - {d_val('accelerated', year) / 1000:,.1f}k")
    m4.metric("EV share of new registrations", f"{share:.1f}%", help="National share of all new vehicle registrations that are electric.")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Historical and scenario EV adoption (national)")
        log = st.checkbox("Log scale", value=False, key="lt_log")
        long_chart(hist, nat, year, scenario, log)
    with right:
        st.subheader(f"Scenario comparison: {nice(district)}")
        shown_years = [y for y in (2027, 2030, 2033, 2035, 2037) if y in years]
        st.dataframe([{"Year": y, **{SC_LABEL[sc]: round(d_val(sc, y)) for sc in SC_LABEL}} for y in shown_years], hide_index=True, width="stretch")
        st.caption("New EV registrations per year. These are if-paths, not predictions.")

    st.subheader("Spatial EV distribution (district level)")
    s1, s2 = st.columns([1, 1])
    with s1:
        long_bars(dist_rows, year, scenario, district)
    with s2:
        st.info("District values are the national scenario split by each district's share of 2025 EV registrations (a static placeholder). "
                "Town / DS-division estimates need the spatial step, which is designed and waiting for Census 2024 DS-division data.")

    st.subheader("Why these scenarios?")
    n_assumed = n_val(scenario, year) / (share / 100.0)
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown("**Method**\n\nStage-matched pace: what other countries' EV shares did from a similar starting share (slow = 25th, baseline = median, accelerated = 75th percentile)")
    d2.markdown(f"**Volume assumption**\n\nTotal new registrations held flat at about {round(n_assumed, -3):,.0f} per year (the 2026 pace)")
    d3.markdown(f"**Status**\n\n`{scen[0]['status']}`")
    d4.markdown("**Limits**\n\nCountry data are cars only; Sri Lankan EVs are mostly motorcycles. Same pace is held for all 11 years.")
    st.caption(f"Run: {scen[0]['run_id']} | scenario file v{scen[0]['scenario_version']}")

def main() -> None:
    st.set_page_config(page_title="EV Adoption Forecast - Component 1", layout="wide")
    st.title("EV Adoption Forecast Dashboard")
    st.caption("Sri Lanka | district-level forecasting prototype (v0). Forecasts are baseline placeholders and if-paths, not the final adaptive model.")
    with st.sidebar:
        st.header("Inputs")
        raw_dir = st.text_input("DMT raw folder", "data/raw")
        forecast_path = st.text_input("Monthly forecast file (contract v0)", "data/processed/forecast_v0.csv")
        scenario_path = st.text_input("Scenario file (2027-2037)", "data/processed/scenarios_v0.csv")
    if not Path(raw_dir).exists():
        st.error(f"Folder not found: {raw_dir}. Run streamlit from the repo root.")
        return
    tab_long, tab_short = st.tabs(["Long-term scenarios 2027-2037", "Next 4 months (v0 forecast)"])
    with tab_long:
        try:
            long_term(read_scenarios(scenario_path), raw_dir)
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
