"""
Prototype v0 (PP1) - Component 1: Adaptive EV Adoption Forecasting. Icon-first, low-text dashboard.

Every number comes from real data (DMT history) or from files that honour our output contracts
(read through their readers, which refuse a broken file). Caveats live behind the (i) buttons.

Run from the repo root:   streamlit run app/streamlit_app.py
Needs (create with the commands shown in the app if missing):
  data/raw                                  private DMT CSVs
  data/processed/forecast_v0.csv            python -m src.models.forecast_v0 --raw-dir data/raw
  data/processed/scenarios_v0.csv           python -m src.models.scenarios --raw-dir data/raw
  data/processed/town_estimates_v0.csv      python -m src.models.town_split --raw-dir data/raw
"""
import csv
import datetime
import sys
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so 'src' imports work when launched by streamlit
from app.model_check import LABEL, LEARNED, mae_table  # noqa: E402
from app.map_view import build_map, ds_values, load_crosswalk, load_geo, shape_names  # noqa: E402
from src.data.census_ds import read_ds  # noqa: E402
from src.models.forecast_v0 import load_district_totals, load_national_monthly, read_contract  # noqa: E402
from src.models.scenarios import read_scenarios  # noqa: E402
from src.models.town_split import CLASSES, read_split  # noqa: E402

POLICY_CSV = "data/external/policy_events_lk.csv"
CENSUS_CSV = "data/external/census2024_ds_population.csv"
GEO_DIR = "data/external/geo"
TRAINED_CSV = "data/processed/trained_vs_baselines.csv"
Row = dict[str, Any]

GREEN, BLUE, RED, ORANGE, GREY = "#2e9e5b", "#1f77b4", "#d62728", "#f0a030", "#8a8f98"
SC_LABEL = {"slow": "Slow", "baseline": "Baseline", "accelerated": "Accelerated"}
SC_ICON = {"slow": "🐢", "baseline": "📈", "accelerated": "🚀"}
SC_COLOR = {"slow": "#2ca02c", "baseline": BLUE, "accelerated": RED}
CLASS_ICON = {"car": "🚗", "motorcycle": "🏍️", "three_wheeler": "🛺", "other": "🚚"}
CLASS_COLOR = {"car": BLUE, "motorcycle": GREEN, "three_wheeler": ORANGE, "other": GREY}
CLASS_NAME = {"car": "Cars", "motorcycle": "Bikes", "three_wheeler": "Three-wheelers", "other": "Other"}

CSS = """
<style>
.kpi{display:flex;align-items:center;gap:14px;padding:14px 16px;border-radius:16px;min-height:96px;background:rgba(127,127,127,.10)}
.kpi b{display:block;font-size:30px;line-height:1.05}
.kpi small{display:block;opacity:.65;font-size:12px;margin-top:2px}
.ico{position:relative;font-size:36px;line-height:1;min-width:44px;text-align:center}
.leaf{position:absolute;right:-8px;top:-10px;font-size:17px}
.chip{display:inline-block;padding:3px 12px;margin:0 6px 6px 0;border-radius:999px;font-size:12.5px;background:rgba(127,127,127,.16)}
.chip.warn{background:rgba(240,160,48,.25)}
.chip.ok{background:rgba(46,158,91,.22)}
.rng{position:relative;height:10px;border-radius:5px;background:linear-gradient(90deg,#2ca02c,#1f77b4,#d62728);margin:10px 0 4px}
.rng i{position:absolute;top:-4px;width:6px;height:18px;border-radius:3px;background:#fff;border:2px solid #333}
.rngl{display:flex;justify-content:space-between;font-size:12px;opacity:.75}
.donutwrap{position:relative;width:64px;height:64px;flex:none}
.donut{width:64px;height:64px;border-radius:50%;-webkit-mask:radial-gradient(farthest-side,transparent 60%,#000 61%);mask:radial-gradient(farthest-side,transparent 60%,#000 61%)}
.donutwrap span{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700}
.mix{display:flex;height:16px;border-radius:8px;overflow:hidden;margin:6px 0 4px}
.mixl{display:flex;gap:14px;flex-wrap:wrap;font-size:13px}
.tile{padding:16px;border-radius:18px;text-align:center;background:rgba(127,127,127,.10)}
.tile .ico{font-size:46px;display:inline-block;margin-bottom:6px}
.tile b{display:block;font-size:34px;line-height:1.1}
.tile small{opacity:.65}
</style>
"""

def html(s: str) -> None:
    """Render raw HTML (indentation would turn markdown into a code block, so lines are stripped and joined)."""
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)

def vehicle(emoji: str) -> str:
    return f'<span class="ico">{emoji}<span class="leaf">🍃</span></span>'

def kpi(key: str, icon: str, value: str, sub: str) -> None:
    html(f'<div class="kpi" data-kpi="{key}">{icon}<div><b>{value}</b><small>{sub}</small></div></div>')

def kpi_range(key: str, low: float, mid: float, high: float, sub: str) -> None:
    pos = 50.0 if high <= low else min(100.0, max(0.0, 100.0 * (mid - low) / (high - low)))
    html(f'<div class="kpi" data-kpi="{key}"><span class="ico">↔️</span><div style="flex:1;min-width:120px">'
         f'<b style="font-size:20px">{low / 1000:,.1f}k – {high / 1000:,.1f}k</b>'
         f'<div class="rng"><i style="left:calc({pos:.0f}% - 3px)"></i></div>'
         f'<div class="rngl"><span>low</span><span>{sub}</span><span>high</span></div></div></div>')

def kpi_donut(key: str, pct: float, sub: str) -> None:
    html(f'<div class="kpi" data-kpi="{key}"><div class="donutwrap"><div class="donut" '
         f'style="background:conic-gradient({GREEN} {pct:.1f}%,rgba(127,127,127,.28) 0)"></div><span>🍃</span></div>'
         f'<div><b>{pct:.1f}%</b><small>{sub}</small></div></div>')

def tile(key: str, emoji: str, value: float, name: str) -> None:
    html(f'<div class="tile" data-kpi="{key}">{vehicle(emoji)}<b>{value:,.0f}</b><small>{name}</small></div>')

def mix_bar(by: dict[str, float]) -> None:
    tot = sum(by.values()) or 1.0
    segs = "".join(f'<div style="width:{100 * by[c] / tot:.2f}%;background:{CLASS_COLOR[c]}"></div>' for c in CLASSES)
    leg = "".join(f'<span>{CLASS_ICON[c]} {100 * by[c] / tot:.0f}%</span>' for c in CLASSES)
    html(f'<div class="mix">{segs}</div><div class="mixl">{leg}</div>')

def chips(items: list[tuple[str, str]]) -> None:
    html("".join(f'<span class="chip {tone}">{text}</span>' for text, tone in items))

def to_num(period: str) -> float:
    return float(mdates.date2num(datetime.datetime(int(period[:4]), int(period[5:7]), 1)))

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

# ---------------------------------------------------------------- charts
def month_chart(values: list[float], periods: list[str], observed: list[bool], nat: list[Row], month: str, events: bool) -> None:
    x = [to_num(p) for p in periods]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, values, color=BLUE, lw=1.8, label="Observed")
    gaps = [i for i, ob in enumerate(observed) if not ob]
    ax.scatter([x[i] for i in gaps], [values[i] for i in gaps], color=GREY, s=28, zorder=3, label="No data")
    fx = [to_num(r["period"]) for r in nat]
    ax.plot([x[-1]] + fx, [values[-1]] + [r["p50"] for r in nat], color=RED, lw=1.8, ls="--", label="Forecast")
    ax.fill_between([x[-1]] + fx, [values[-1]] + [r["p10"] for r in nat], [values[-1]] + [r["p90"] for r in nat], color=RED, alpha=0.18)
    ax.scatter([to_num(month)], [[r for r in nat if r["period"] == month][0]["p50"]], color=RED, s=45, zorder=4)
    if events:
        top = ax.get_ylim()[1]
        for e in load_events(POLICY_CSV):
            if e["date_precision"] in ("day", "month"):
                ex = to_num(e["date"][:7])
                if x[0] <= ex <= x[-1]:
                    ax.axvline(ex, color="#555555", lw=0.8, ls=":")
                    ax.text(ex, top * 0.97, e["event_id"], rotation=90, va="top", ha="right", fontsize=7, color="#333333")
    ax.xaxis_date()
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", fontsize=7)
    st.pyplot(fig)

def year_chart(hist: dict[int, float], nat: list[Row], year: int, scenario: str, log: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    hy = [y for y in sorted(hist) if y <= 2025]
    ax.plot(hy, [hist[y] for y in hy], color="#333333", lw=2, marker="o", label="Observed")
    ann26 = hist[2026] * 12.0 / 8.0
    ax.scatter([2026], [ann26], facecolors="none", edgecolors="#333333", s=60, zorder=4, label="2026 (annualised)")
    years = sorted({r["year"] for r in nat})
    by = {sc: [r["new_evs"] for r in sorted((r for r in nat if r["scenario"] == sc), key=lambda r: r["year"])] for sc in SC_LABEL}
    ax.fill_between([2026] + years, [ann26] + by["slow"], [ann26] + by["accelerated"], color=GREY, alpha=0.15)
    for sc in SC_LABEL:
        ax.plot([2026] + years, [ann26] + by[sc], color=SC_COLOR[sc], lw=2.6 if sc == scenario else 1.1, ls="--", label=SC_LABEL[sc])
    pick = [r for r in nat if r["scenario"] == scenario and r["year"] == year][0]
    ax.scatter([year], [pick["new_evs"]], color=SC_COLOR[scenario], s=60, zorder=5)
    if log:
        ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    st.pyplot(fig)

def bars(names: list[str], vals: list[float], selected: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.tick_params(labelsize=8)
    ax.barh(names[::-1], vals[::-1], color=[RED if n.upper() == selected else BLUE for n in names[::-1]])
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig)

def top_with_selected(items: list[tuple[str, float]], selected: str, n: int = 10) -> tuple[list[str], list[float]]:
    items = sorted(items, key=lambda t: -t[1])
    shown = items[:n]
    if selected not in [d for d, _ in shown]:
        shown += [t for t in items if t[0] == selected]
    return [nice(d) for d, _ in shown], [v for _, v in shown]

# ---------------------------------------------------------------- tab 1: long term
def town_tiles(town: list[Row], district: str, area: str, year: int, scenario: str) -> None:
    rows = [r for r in town if r["district"] == district and r["ds_name"] == area and r["scenario"] == scenario and r["year"] == year]
    by: dict[str, float] = {c: float(sum(r["new_evs"] for r in rows if r["vehicle_class"] == c)) for c in CLASSES}
    pop = [r for r in load_census(CENSUS_CSV) if r["district_key"] == district and r["ds_name"] == area]
    h1, h2 = st.columns([4, 1])
    h1.subheader(f"📍 {area}  ·  {year}  ·  {SC_ICON[scenario]} {SC_LABEL[scenario]}")
    with h2.popover("ⓘ Notes"):
        st.markdown("**Estimate, not an observation.** District forecast × the district's vehicle-class mix × this area's share of the "
                    "district's population" + (f" ({pop[0]['population']:,} people, Census 2024 provisional)." if pop else ".") +
                    "\n\nPopulation is the only weight, so wealthier urban areas may hold more cars than shown. "
                    "DMT records where a vehicle is registered, not where the owner lives.")
    cols = st.columns(len(CLASSES))
    for col, c in zip(cols, CLASSES):
        with col:
            tile(c, CLASS_ICON[c], by[c], CLASS_NAME[c])
    mix_bar(by)

def map_section(town: list[Row], district: str, area: str, year: int, scenario: str) -> None:
    if not Path(GEO_DIR, "lka_adm3_simple.geojson").exists():
        st.info("Map files not found. Put them in data/external/geo/")
        return
    geo, cw = load_map_files()
    st.subheader("🗺️ Map")
    whole = st.toggle("Whole island", value=False, key="lt_whole")
    focus = {code for (d, _), code in cw.items() if d == district}
    selected = cw.get((district, area)) if area != "All areas" else None
    fig = build_map(geo, ds_values(town, cw, year, scenario), shape_names(geo), focus, selected, whole)
    st.plotly_chart(fig, width="stretch")
    st.caption("Darker = more new EVs (estimate, all vehicle types). Red outline = selected area. "
               "Boundaries: OCHA / Survey Department of Sri Lanka (2022, simplified). Map © OpenStreetMap contributors.")

def long_term(scen: list[Row], raw_dir: str, town: list[Row] | None) -> None:
    nat = [r for r in scen if r["level"] == "national"]
    dist_rows = [r for r in scen if r["level"] == "district"]
    districts = sorted({r["district"] for r in dist_rows})
    years = sorted({r["year"] for r in nat})
    hist = annual_history(raw_dir)
    ann26 = hist[2026] * 12.0 / 8.0

    chips([("🧪 Prototype v0", "warn"), ("🔮 If-paths, not predictions", ""), ("📍 District → Town", "ok")])
    c1, c2, c3, c4 = st.columns(4)
    district = c1.selectbox("📍 District", districts, index=districts.index("COLOMBO") if "COLOMBO" in districts else 0,
                            format_func=nice, key="lt_district") or districts[0]
    ds_names = sorted({r["ds_name"] for r in town if r["district"] == district}) if town else []
    area = c2.selectbox("🏘️ Area", ["All areas"] + ds_names, key="lt_area", disabled=not town) or "All areas"
    year = c3.selectbox("📅 Year", years, index=years.index(2030) if 2030 in years else 0, key="lt_year") or years[0]
    scenario = c4.selectbox("🎚️ Scenario", list(SC_LABEL), index=1, format_func=lambda k: f"{SC_ICON[k]} {SC_LABEL[k]}", key="lt_scenario") or "baseline"

    def d_val(sc: str, y: int) -> float:
        return [r["new_evs"] for r in dist_rows if r["district"] == district and r["scenario"] == sc and r["year"] == y][0]
    def n_val(sc: str, y: int) -> float:
        return [r["new_evs"] for r in nat if r["scenario"] == sc and r["year"] == y][0]
    prev = ann26 if year == years[0] else n_val(scenario, year - 1)
    growth = n_val(scenario, year) / prev - 1
    share = [r["national_ev_share_pct"] for r in nat if r["scenario"] == scenario and r["year"] == year][0]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("district_evs", vehicle("⚡"), f"{d_val(scenario, year):,.0f}", f"{nice(district)} · {year}")
    with k2:
        kpi("growth", f'<span class="ico">{"📈" if growth >= 0 else "📉"}</span>', f"{growth:+.0%}", "vs last year · national")
    with k3:
        kpi_range("range", d_val("slow", year), d_val(scenario, year), d_val("accelerated", year), f"{SC_ICON[scenario]}")
    with k4:
        kpi_donut("share", share, "of new vehicles are EVs")

    if town and area != "All areas":
        town_tiles(town, district, area, year, scenario)
    if town:
        map_section(town, district, area, year, scenario)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("📊 National: history and scenarios")
        log = st.toggle("Log scale", value=False, key="lt_log")
        year_chart(hist, nat, year, scenario, log)
    with right:
        st.subheader(f"🎚️ {nice(district)}")
        shown_years = [y for y in (2027, 2030, 2033, 2035, 2037) if y in years]
        st.dataframe([{"Year": y, **{f"{SC_ICON[sc]}": round(d_val(sc, y)) for sc in SC_LABEL}} for y in shown_years], hide_index=True, width="stretch")
        st.subheader("🗺️ Where?")
        names, vals = top_with_selected([(r["district"], r["new_evs"]) for r in dist_rows if r["year"] == year and r["scenario"] == scenario], district)
        bars(names, vals, district)

    with st.expander("ⓘ How these scenarios are made"):
        n_assumed = n_val(scenario, year) / (share / 100.0)
        st.markdown("**Method.** Stage-matched pace: what other countries' EV shares did from a similar starting share "
                    "(🐢 slow = 25th percentile, 📈 baseline = median, 🚀 accelerated = 75th).\n\n"
                    f"**Volume.** Total new registrations held flat at about {round(n_assumed, -3):,.0f} a year (the 2026 pace).\n\n"
                    "**Limits.** Country data are cars only, Sri Lankan EVs are mostly motorcycles; the same pace is held for all 11 years; "
                    "district numbers use 2025 shares.\n\n"
                    f"Status `{scen[0]['status']}` · run {scen[0]['run_id']}")

# ---------------------------------------------------------------- tab 2: next 4 months
def model_chart(table: dict[str, tuple[float, int]], title: str) -> None:
    items = sorted(table.items(), key=lambda t: t[1][0])
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.barh([LABEL[m] for m, _ in items][::-1], [v[0] for _, v in items][::-1],
            color=[ORANGE if m in LEARNED else BLUE for m, _ in items][::-1], height=0.62)
    for i, (_, v) in enumerate(items[::-1]):
        ax.text(v[0], i, f" {v[0]:,.0f}", va="center", fontsize=9)
    ax.set_title(title, loc="left", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, max(v[0] for _, v in items) * 1.18)
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig)

def model_check() -> None:
    st.subheader("🧪 Model check")
    if not Path(TRAINED_CSV).exists():
        st.info("Not made yet. Run:  python -m src.models.trained_v0 --raw-dir data/raw")
        return
    tab = mae_table(TRAINED_CSV)
    chips([("🔬 Walk-forward · no future data", "ok"), ("🟦 Rules", ""), ("🟧 Learned models", "warn")])
    c1, c2 = st.columns(2)
    with c1:
        n = next(iter(tab["development"].values()))[1]
        model_chart(tab["development"], f"Development ({n} months)")
    with c2:
        n = next(iter(tab["holdout"].values()))[1]
        model_chart(tab["holdout"], f"2026 holdout ({n} months)")
    best = min(tab["development"].items(), key=lambda t: t[1][0])[0]
    st.caption(f"Average miss in new EVs, next month · lower is better · best in development: {LABEL[best]}. "
               "Learned models did not clearly beat simple rules (very little high-volume data); the holdout has only a few months.")

def short_term(rows: list[Row], raw_dir: str) -> None:
    nat = sorted((r for r in rows if r["level"] == "national"), key=lambda r: r["horizon_months"])
    dist_rows = [r for r in rows if r["level"] == "district"]
    districts = sorted({r["district"] for r in dist_rows})
    months = [r["period"] for r in nat]
    values, periods, observed = load_history(raw_dir)

    chips([("🧪 Baseline placeholder", "warn"), ("📏 Interval check: pending", "")])
    c1, c2 = st.columns(2)
    district = c1.selectbox("📍 District", districts, index=districts.index("COLOMBO") if "COLOMBO" in districts else 0,
                            format_func=nice, key="st_district") or districts[0]
    month = c2.selectbox("📅 Month", months, index=len(months) - 1, key="st_month") or months[-1]

    pick = [r for r in dist_rows if r["district"] == district and r["period"] == month][0]
    change = [r for r in nat if r["period"] == month][0]["p50"] / values[-1] - 1
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("st_evs", vehicle("⚡"), f"{pick['p50']:,.0f}", f"{nice(district)} · {month}")
    with k2:
        kpi("st_change", f'<span class="ico">{"📈" if change >= 0 else "📉"}</span>', f"{change:+.0%}", f"vs {periods[-1]} · national")
    with k3:
        kpi_range("st_range", pick["p10"], pick["p50"], pick["p90"], "80%")

    left, right = st.columns([2, 1])
    with left:
        events = st.toggle("🏛️ Policy events", value=True, key="st_events")
        month_chart(values, periods, observed, nat, month, events)
    with right:
        st.subheader(f"📅 {nice(district)}")
        st.dataframe([{"Month": r["period"], "low": round(r["p10"]), "mid": round(r["p50"]), "high": round(r["p90"])}
                      for r in sorted((r for r in dist_rows if r["district"] == district), key=lambda r: r["horizon_months"])],
                     hide_index=True, width="stretch")
        st.subheader("🗺️ Where?")
        names, vals = top_with_selected([(r["district"], r["p50"]) for r in dist_rows if r["period"] == month], district)
        bars(names, vals, district)
    model_check()
    with st.expander("ⓘ How this is made"):
        st.markdown(f"Method `{pick['method']}`: last observed month × growth ratios seen since 2023. District = national × its 2025 share. "
                    f"Status `{pick['status']}` · run {pick['run_id']} · contract v{pick['contract_version']}. "
                    "Interval coverage has not been measured yet.")

def main() -> None:
    st.set_page_config(page_title="EV Adoption Forecast", page_icon="🍃", layout="wide")
    html(CSS)
    st.title("🍃 EV Adoption Forecast · Sri Lanka")
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
            town = load_town(town_path) if Path(town_path).exists() else None
            if town is None:
                st.info("Area list is off. Create town estimates:  python -m src.models.town_split --raw-dir data/raw")
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