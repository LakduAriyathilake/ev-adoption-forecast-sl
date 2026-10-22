"""
Plotly charts for the dashboard. Plotly draws vector charts that scale with the page, so they stay sharp and
aligned at any browser zoom (the older matplotlib pictures were fixed-size images).

Rules used everywhere: transparent background, thin lines, quiet grid, hover tooltips, legend when 2+ series.
"""
from typing import Any

import plotly.graph_objects as go

from app.theme import AQUA, BLUE, GREY, INK, MUTED, ORANGE, SC_COLOR, SC_LABEL

Row = dict[str, Any]
GRID = "rgba(143,169,179,.14)"

def _base(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin={"l": 8, "r": 8, "t": 8, "b": 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans, Segoe UI, sans-serif", "color": MUTED, "size": 12},
        hoverlabel={"bgcolor": "#0f212a", "bordercolor": "#1f3f4c", "font": {"color": INK, "size": 12}},
        legend={"orientation": "h", "x": 0, "y": 1.1, "traceorder": "normal", "font": {"color": INK}, "bgcolor": "rgba(0,0,0,0)"})
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#1f3f4c", tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont={"color": MUTED})
    return fig

def scenario_chart(hist: dict[int, float], nat: list[Row], year: int, scenario: str, log: bool, height: int = 540) -> go.Figure:
    """Yearly new EVs: what happened up to 2025, 2026 scaled to a full year, and the three scenarios to 2037."""
    hy = [y for y in sorted(hist) if y <= 2025]
    ann26 = hist[2026] * 12.0 / 8.0
    years = sorted({r["year"] for r in nat})
    by = {sc: [r["new_evs"] for r in sorted((r for r in nat if r["scenario"] == sc), key=lambda r: r["year"])] for sc in SC_LABEL}
    xs = [2026] + years
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[ann26] + by["accelerated"], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=[ann26] + by["slow"], mode="lines", line={"width": 0}, fill="tonexty",
                             fillcolor="rgba(143,169,179,.12)", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=hy, y=[hist[y] for y in hy], mode="lines+markers", name="Observed", line={"color": INK, "width": 2.5},
                             marker={"size": 8, "color": INK}, hovertemplate="%{x}: %{y:,.0f}<extra>Observed</extra>"))
    fig.add_trace(go.Scatter(x=[2026], y=[ann26], mode="markers", name="2026 (full-year pace)",
                             marker={"size": 11, "color": "rgba(0,0,0,0)", "line": {"color": INK, "width": 2}},
                             hovertemplate="2026 at this pace: %{y:,.0f}<extra></extra>"))
    for sc in SC_LABEL:
        fig.add_trace(go.Scatter(x=xs, y=[ann26] + by[sc], mode="lines", name=SC_LABEL[sc],
                                 line={"color": SC_COLOR[sc], "width": 3.5 if sc == scenario else 2, "dash": "dash"},
                                 hovertemplate="%{x}: %{y:,.0f}<extra>" + SC_LABEL[sc] + "</extra>"))
    pick = [r for r in nat if r["scenario"] == scenario and r["year"] == year][0]
    fig.add_trace(go.Scatter(x=[year], y=[pick["new_evs"]], mode="markers", showlegend=False, hoverinfo="skip",
                             marker={"size": 13, "color": SC_COLOR[scenario], "line": {"color": "#08141a", "width": 2}}))
    _base(fig, height)
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(type="log" if log else "linear", tickformat="~s", title={"text": "new EVs a year", "font": {"size": 11}})
    return fig

def month_chart(values: list[float], periods: list[str], observed: list[bool], nat: list[Row], month: str,
                events: list[Row], height: int = 500) -> go.Figure:
    """Monthly new EVs: observed line, the 4-month forecast with its 80% band, and (optional) policy events."""
    x = [p + "-01" for p in periods]
    gaps = [i for i, ob in enumerate(observed) if not ob]
    fx = [x[-1]] + [r["period"] + "-01" for r in nat]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fx, y=[values[-1]] + [r["p90"] for r in nat], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=fx, y=[values[-1]] + [r["p10"] for r in nat], mode="lines", line={"width": 0}, fill="tonexty",
                             fillcolor="rgba(217,89,38,.30)", name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=values, mode="lines", name="Observed", line={"color": BLUE, "width": 2.2},
                             hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra>Observed</extra>"))
    if gaps:
        fig.add_trace(go.Scatter(x=[x[i] for i in gaps], y=[values[i] for i in gaps], mode="markers", name="No data",
                                 marker={"size": 8, "color": GREY}, hovertemplate="%{x|%b %Y}: no data<extra></extra>"))
    fig.add_trace(go.Scatter(x=fx, y=[values[-1]] + [r["p50"] for r in nat], mode="lines", name="Forecast",
                             line={"color": ORANGE, "width": 2.5, "dash": "dash"},
                             hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra>Forecast</extra>"))
    pick = [r for r in nat if r["period"] == month][0]
    fig.add_trace(go.Scatter(x=[month + "-01"], y=[pick["p50"]], mode="markers", showlegend=False, hoverinfo="skip",
                             marker={"size": 12, "color": ORANGE, "line": {"color": "#08141a", "width": 2}}))
    for e in events:
        ex = e["date"][:7] + "-01"
        if x[0] <= ex <= x[-1]:
            fig.add_vline(x=ex, line={"color": "rgba(230,239,242,.35)", "width": 1, "dash": "dot"})
            fig.add_annotation(x=ex, y=1, yref="paper", text=e["event_id"], showarrow=False, textangle=-90, yanchor="top",
                               xanchor="right", font={"size": 9, "color": MUTED})
    _base(fig, height)
    fig.update_yaxes(tickformat="~s", title={"text": "new EVs a month", "font": {"size": 11}})
    return fig

def rank_bars(names: list[str], vals: list[float], selected: str, height: int = 330) -> go.Figure:
    """Horizontal bars, biggest on top; the selected district is highlighted."""
    fig = go.Figure(go.Bar(x=vals[::-1], y=names[::-1], orientation="h", marker_color=[AQUA if n == selected else "#35566a" for n in names[::-1]],
                           marker_cornerradius=4, text=[f"{v:,.0f}" for v in vals[::-1]], textposition="outside", cliponaxis=False,
                           textfont={"color": INK, "size": 11}, hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
    _base(fig, height)
    fig.update_layout(showlegend=False, margin={"l": 8, "r": 40, "t": 4, "b": 4}, bargap=0.35)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(tickfont={"color": INK, "size": 12})
    return fig

def model_bars(table: dict[str, tuple[float, int]], labels: dict[str, str], learned: set[str], title: str, height: int = 300) -> go.Figure:
    """One panel of the model check: average miss per method, lower is better; learned models in orange."""
    items = sorted(table.items(), key=lambda t: t[1][0])
    fig = go.Figure(go.Bar(x=[v[0] for _, v in items][::-1], y=[labels[m] for m, _ in items][::-1], orientation="h",
                           marker_color=[ORANGE if m in learned else BLUE for m, _ in items][::-1], marker_cornerradius=4,
                           text=[f"{v[0]:,.0f}" for _, v in items][::-1], textposition="outside", cliponaxis=False,
                           textfont={"color": INK, "size": 11}, hovertemplate="%{y}: %{x:,.0f} EVs off on average<extra></extra>"))
    _base(fig, height)
    fig.update_layout(showlegend=False, margin={"l": 8, "r": 40, "t": 34, "b": 4}, bargap=0.35,
                      title={"text": title, "x": 0, "font": {"size": 14, "color": INK}})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(tickfont={"color": INK, "size": 12})
    return fig
