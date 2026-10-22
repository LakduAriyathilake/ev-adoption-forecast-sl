"""
Map view for the dashboard: DS divisions coloured by estimated new EVs, on OpenStreetMap.

Three small steps (each is its own function so each can be tested):
  1. load_geo / load_crosswalk   read the boundary shapes and the census-name -> shape-code table
  2. ds_values                   add up the town estimates per DS shape for one year and scenario
  3. build_map                   draw the shapes with Plotly; the chosen area gets a red outline

Boundaries: OCHA / Survey Department of Sri Lanka (2022 version), simplified for speed.
"""
import csv
import json
import math
from typing import Any

import plotly.graph_objects as go

Row = dict[str, Any]
CLASSES = ("car", "motorcycle", "three_wheeler", "other")
NAMES = {"car": "Cars", "motorcycle": "Bikes", "three_wheeler": "Three-wheelers", "other": "Other"}

def load_geo(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_crosswalk(path: str) -> dict[tuple[str, str], str]:
    """(DMT district, census DS name) -> boundary shape code."""
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["district_key"], r["census_ds_name"]): r["adm3_pcode"] for r in csv.DictReader(f)}

def shape_names(geo: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """shape code -> (DS name, district name) as written in the boundary file."""
    return {f["properties"]["adm3_pcode"]: (f["properties"]["adm3_name"], f["properties"]["adm2_name"]) for f in geo["features"]}

def ds_values(town: list[Row], cw: dict[tuple[str, str], str], year: int, scenario: str) -> dict[str, dict[str, float]]:
    """shape code -> {car, motorcycle, three_wheeler, other, total}. Two census areas on one shape are added together."""
    out: dict[str, dict[str, float]] = {}
    for r in town:
        if r["year"] != year or r["scenario"] != scenario:
            continue
        code = cw[(r["district"], r["ds_name"])]
        d = out.setdefault(code, {c: 0.0 for c in CLASSES})
        d[r["vehicle_class"]] += float(r["new_evs"])
    for d in out.values():
        d["total"] = sum(d[c] for c in CLASSES)
    return out

def _bbox(geo: dict[str, Any], codes: set[str]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    def walk(c: Any) -> None:
        if isinstance(c[0], (int, float)):
            xs.append(float(c[0])); ys.append(float(c[1]))
        else:
            for p in c:
                walk(p)
    for f in geo["features"]:
        if f["properties"]["adm3_pcode"] in codes:
            walk(f["geometry"]["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)

def view_for(geo: dict[str, Any], codes: set[str]) -> tuple[dict[str, float], float]:
    x0, y0, x1, y1 = _bbox(geo, codes)
    span = max(x1 - x0, (y1 - y0) * 1.3, 0.05)          # how wide the area is, in degrees
    zoom = max(5.5, min(10.0, math.log2(360 * 1.2 / span) - 0.3))   # web maps: each +1 zoom halves the width shown
    return {"lon": (x0 + x1) / 2, "lat": (y0 + y1) / 2}, zoom

def build_map(geo: dict[str, Any], values: dict[str, dict[str, float]], names: dict[str, tuple[str, str]],
              focus_codes: set[str], selected: str | None, whole_island: bool, style: str = "open-street-map") -> go.Figure:
    codes = [c for c in values if c in names]
    hover = []
    for c in codes:
        v = values[c]
        hover.append(f"<b>{names[c][0]}</b> · {names[c][1].title()}<br>" +
                     "<br>".join(f"{NAMES[k]}: {v[k]:,.0f}" for k in CLASSES) + f"<br><b>All: {v['total']:,.0f}</b>")
    fig = go.Figure(go.Choroplethmap(
        geojson=geo, locations=codes, featureidkey="properties.adm3_pcode", z=[values[c]["total"] for c in codes],
        colorscale="YlGn", marker_opacity=0.75, marker_line_width=0.4, marker_line_color="#4b5563",
        hovertext=hover, hoverinfo="text", colorbar={"title": "EVs", "thickness": 12, "len": 0.6}))
    if selected and selected in names:
        fig.add_trace(go.Choroplethmap(
            geojson=geo, locations=[selected], featureidkey="properties.adm3_pcode", z=[1],
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False, hoverinfo="skip",
            marker_line_width=3, marker_line_color="#d62728"))
    center, zoom = ({"lon": 80.7, "lat": 7.8}, 6.6) if whole_island or not focus_codes else view_for(geo, focus_codes)
    fig.update_layout(map_style=style, map_center=center, map_zoom=zoom, margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=460)
    return fig
