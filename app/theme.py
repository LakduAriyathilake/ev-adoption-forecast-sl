"""
Look and feel for the dashboard: colours, CSS and the small HTML pieces (KPI cards, vehicle tiles, chips).

Colours: the three series colours (blue, aqua, orange) and the yellow are the validated dark-mode slots of the
reference data-viz palette. Scenarios run cool to hot: slow = aqua, baseline = blue, accelerated = orange.
"""
from typing import Any

import streamlit as st

BG, PANEL, PANEL2, LINE = "#08141a", "#0f212a", "#15303b", "#1f3f4c"
INK, MUTED = "#e6eff2", "#8fa9b3"
BLUE, AQUA, ORANGE, YELLOW, GREY = "#3987e5", "#199e70", "#d95926", "#c98500", "#6f8791"

CLASSES = ("car", "motorcycle", "three_wheeler", "other")
CLASS_ICON = {"car": "🚗", "motorcycle": "🏍️", "three_wheeler": "🛺", "other": "🚚"}
CLASS_COLOR = {"car": BLUE, "motorcycle": AQUA, "three_wheeler": YELLOW, "other": GREY}
CLASS_NAME = {"car": "Cars", "motorcycle": "Bikes", "three_wheeler": "Three-wheelers", "other": "Other"}
SC_LABEL = {"slow": "Slow", "baseline": "Baseline", "accelerated": "Accelerated"}
SC_ICON = {"slow": "🐢", "baseline": "📈", "accelerated": "🚀"}
SC_COLOR = {"slow": AQUA, "baseline": BLUE, "accelerated": ORANGE}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{{--bg:{BG};--panel:{PANEL};--panel2:{PANEL2};--line:{LINE};--ink:{INK};--muted:{MUTED};--aqua:{AQUA};--blue:{BLUE};--orange:{ORANGE}}}
html,body,[class*="css"],.stApp{{font-family:'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif;font-variant-numeric:tabular-nums}}
.stApp{{background:radial-gradient(900px 420px at 88% -8%,rgba(25,158,112,.16),transparent 62%),radial-gradient(700px 400px at -5% 8%,rgba(57,135,229,.10),transparent 60%),var(--bg)}}
header[data-testid="stHeader"]{{background:transparent}}
#MainMenu,footer,[data-testid="stToolbar"]{{visibility:hidden}}
.block-container{{max-width:1480px;padding:1.2rem 2rem 3rem}}
h1,h2,h3{{letter-spacing:-.01em}}

/* header */
.top{{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin:2px 0 14px}}
.brand{{display:flex;align-items:center;gap:14px}}
.logo{{width:50px;height:50px;border-radius:15px;display:flex;align-items:center;justify-content:center;font-size:26px;background:linear-gradient(135deg,#199e70,#3987e5);box-shadow:0 8px 24px rgba(25,158,112,.28)}}
.brand h1{{margin:0;font-size:28px;font-weight:800;line-height:1.1;color:var(--ink)}}
.brand p{{margin:2px 0 0;font-size:13.5px;color:var(--muted)}}
.chip{{display:inline-flex;align-items:center;gap:6px;padding:5px 13px;margin:0 0 0 8px;border-radius:999px;font-size:12.5px;font-weight:600;background:rgba(143,169,179,.14);color:var(--ink);border:1px solid rgba(143,169,179,.18)}}
.chip.warn{{background:rgba(201,133,0,.18);border-color:rgba(201,133,0,.35)}}
.chip.ok{{background:rgba(25,158,112,.18);border-color:rgba(25,158,112,.4)}}

/* tabs as pills */
.stTabs [data-baseweb="tab-list"]{{gap:8px;border-bottom:none;margin-bottom:6px}}
.stTabs [data-baseweb="tab"]{{height:42px;padding:0 20px;border-radius:999px;background:var(--panel);border:1px solid var(--line);color:var(--muted);font-weight:600}}
.stTabs [aria-selected="true"]{{background:linear-gradient(135deg,rgba(25,158,112,.28),rgba(57,135,229,.22));color:var(--ink);border-color:rgba(25,158,112,.55)}}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{{display:none}}

/* cards: any container whose key starts with card_ */
[class*="st-key-card_"]{{background:linear-gradient(180deg,var(--panel),#0c1c24);border:1px solid var(--line);border-radius:20px;padding:18px 20px;box-shadow:0 10px 30px rgba(0,0,0,.28)}}
.st-key-card_map,.st-key-card_area{{min-height:700px}}
.st-key-card_chart,.st-key-card_rank{{min-height:670px}}
.st-key-card_chart_st,.st-key-card_rank_st{{min-height:650px}}
.cardh{{display:flex;align-items:center;gap:10px;margin:0 0 10px;font-size:16px;font-weight:700;color:var(--ink)}}
.cardh small{{font-weight:500;color:var(--muted);font-size:13px}}

/* inputs */
label[data-testid="stWidgetLabel"] p{{font-size:12.5px;font-weight:600;color:var(--muted)}}
div[data-baseweb="select"]>div{{background:var(--panel2);border:1px solid var(--line);border-radius:12px;min-height:44px}}
div[data-baseweb="select"]>div:hover{{border-color:rgba(25,158,112,.6)}}
[data-testid="stSegmentedControl"] button{{border-radius:12px}}

/* KPI cards */
.kpi{{display:flex;align-items:center;gap:14px;padding:16px 18px;height:104px;border-radius:18px;background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line)}}
.kpi b{{display:block;font-size:30px;line-height:1.05;font-weight:800;color:var(--ink)}}
.kpi small{{display:block;margin-top:3px;font-size:12.5px;color:var(--muted)}}
.ico{{position:relative;font-size:34px;line-height:1;min-width:46px;text-align:center}}
.leaf{{position:absolute;right:-8px;top:-10px;font-size:16px}}
.rng{{position:relative;height:8px;border-radius:4px;background:linear-gradient(90deg,{AQUA},{BLUE},{ORANGE});margin:10px 0 5px}}
.rng i{{position:absolute;top:-5px;width:6px;height:18px;border-radius:3px;background:#fff;border:2px solid #08141a}}
.rngl{{display:flex;justify-content:space-between;font-size:11.5px;color:var(--muted)}}
.donutwrap{{position:relative;width:62px;height:62px;flex:none}}
.donut{{width:62px;height:62px;border-radius:50%;-webkit-mask:radial-gradient(farthest-side,transparent 62%,#000 63%);mask:radial-gradient(farthest-side,transparent 62%,#000 63%)}}
.donutwrap span{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:18px}}

/* vehicle tiles + mix */
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px}}
.tile{{padding:14px 10px 12px;border-radius:16px;text-align:center;background:var(--panel2);border:1px solid var(--line)}}
.tile .ico{{font-size:38px;display:inline-block;margin-bottom:4px}}
.tile b{{display:block;font-size:30px;line-height:1.1;font-weight:800;color:var(--ink)}}
.tile small{{display:block;color:var(--muted);font-size:12.5px;margin-top:2px}}
.mix{{display:flex;gap:3px;height:12px;margin:16px 0 8px}}
.mix div{{border-radius:6px}}
.mixl{{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--ink)}}
.toph{{margin:18px 0 8px;font-size:13px;font-weight:700;color:var(--muted)}}
.row{{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:13px;color:var(--ink)}}
.row span.n{{width:42%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.row .bar{{flex:1;height:8px;border-radius:4px;background:rgba(143,169,179,.16);overflow:hidden}}
.row .bar i{{display:block;height:100%;border-radius:4px;background:{BLUE}}}
.row.sel .bar i{{background:{AQUA}}}
.row.sel span.n{{font-weight:800}}
.row span.v{{width:54px;text-align:right;color:var(--muted)}}
.cap{{font-size:12px;color:var(--muted);margin-top:6px;line-height:1.45}}
</style>
"""

def html(s: str) -> None:
    """Render raw HTML (indentation would turn markdown into a code block, so lines are stripped and joined)."""
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)

def chips_html(items: list[tuple[str, str]]) -> str:
    return "".join(f'<span class="chip {tone}">{text}</span>' for text, tone in items)

def header(items: list[tuple[str, str]]) -> None:
    html(f'<div class="top"><div class="brand"><div class="logo">🍃</div><div><h1>EV Adoption Forecast</h1>'
         f'<p>Sri Lanka · from districts down to towns · 2027–2037</p></div></div><div>{chips_html(items)}</div></div>')

def card_title(icon: str, text: str, sub: str = "") -> None:
    html(f'<div class="cardh"><span>{icon}</span><span>{text}</span><small>{sub}</small></div>')

def vehicle(emoji: str) -> str:
    return f'<span class="ico">{emoji}<span class="leaf">🍃</span></span>'

def kpi(key: str, icon: str, value: str, sub: str) -> None:
    html(f'<div class="kpi" data-kpi="{key}">{icon}<div><b>{value}</b><small>{sub}</small></div></div>')

def kpi_range(key: str, low: float, mid: float, high: float, sub: str) -> None:
    pos = 50.0 if high <= low else min(100.0, max(0.0, 100.0 * (mid - low) / (high - low)))
    html(f'<div class="kpi" data-kpi="{key}"><span class="ico">↔️</span><div style="flex:1;min-width:120px">'
         f'<b style="font-size:21px">{low / 1000:,.1f}k – {high / 1000:,.1f}k</b>'
         f'<div class="rng"><i style="left:calc({pos:.0f}% - 3px)"></i></div>'
         f'<div class="rngl"><span>low</span><span>{sub}</span><span>high</span></div></div></div>')

def kpi_donut(key: str, pct: float, sub: str) -> None:
    html(f'<div class="kpi" data-kpi="{key}"><div class="donutwrap"><div class="donut" '
         f'style="background:conic-gradient({AQUA} {pct:.1f}%,rgba(143,169,179,.22) 0)"></div><span>🍃</span></div>'
         f'<div><b>{pct:.1f}%</b><small>{sub}</small></div></div>')

def tiles(by: dict[str, float]) -> None:
    cells = "".join(f'<div class="tile" data-kpi="{c}">{vehicle(CLASS_ICON[c])}<b>{by[c]:,.0f}</b><small>{CLASS_NAME[c]}</small></div>'
                    for c in CLASSES)
    html(f'<div class="grid2">{cells}</div>')

def mix_bar(by: dict[str, float]) -> None:
    tot = sum(by.values()) or 1.0
    segs = "".join(f'<div style="width:{100 * by[c] / tot:.2f}%;background:{CLASS_COLOR[c]}"></div>' for c in CLASSES)
    leg = "".join(f'<span>{CLASS_ICON[c]} {100 * by[c] / tot:.0f}%</span>' for c in CLASSES)
    html(f'<div class="mix">{segs}</div><div class="mixl">{leg}</div>')

def top_list(items: list[tuple[str, float]], selected: str | None, title: str) -> None:
    """Small ranked list with bars: (name, value) pairs, biggest first; the selected name is highlighted."""
    top = max((v for _, v in items), default=1.0) or 1.0
    rows = "".join(f'<div class="row{" sel" if n == selected else ""}"><span class="n">{n}</span>'
                   f'<div class="bar"><i style="width:{100 * v / top:.1f}%"></i></div><span class="v">{v:,.0f}</span></div>'
                   for n, v in items)
    html(f'<div class="toph">{title}</div>{rows}')
