"""
SAINT — Agent-in-the-loop monitoring dashboard (Plotly Dash).

Panels:
  • KPI cards: total / allowed / flagged / blocked / review queue / avg confidence
  • Attack timeline  — stacked area chart per threat class over time
  • Threat class distribution — bar chart
  • Action breakdown — donut pie
  • Confidence histogram
  • Burst alert banner
  • Review queue — flagged decisions awaiting human action
  • Live feed table — color-coded by action
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import redis
from dash import Input, Output, dash_table, dcc, html

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DASH_PORT, DASHBOARD_REFRESH_INTERVAL_MS,
    REDIS_DB, REDIS_HOST, REDIS_PORT,
)

# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

_redis: redis.Redis | None = None

def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    return _redis


def fetch_decisions(n: int = 500) -> list[dict]:
    try:
        r = get_redis()
        raw = r.lrange("saint:decisions", 0, n - 1)
        return [json.loads(item) for item in raw]
    except redis.RedisError:
        return []


def fetch_reviews() -> dict:
    try:
        r = get_redis()
        raw = r.hgetall("saint:reviews")
        return {k: json.loads(v) for k, v in raw.items()}
    except redis.RedisError:
        return {}


# ---------------------------------------------------------------------------
# Color palettes & chart defaults
# ---------------------------------------------------------------------------

ACTION_COLORS = {"allow": "#2ECC71", "flag": "#F39C12", "block": "#E74C3C"}
CLASS_COLORS  = {
    "normal": "#3498DB",
    "dos":    "#E74C3C",
    "probe":  "#F39C12",
    "r2l":    "#9B59B6",
    "u2r":    "#1ABC9C",
}
_DARK_BG  = "#0d0d1a"
_CARD_BG  = "#16213e"
_PANEL_BG = "#1a1a2e"
_ACCENT   = "#00F5D4"

_LAYOUT_BASE = dict(
    paper_bgcolor=_CARD_BG,
    plot_bgcolor=_PANEL_BG,
    font=dict(family="Inter, sans-serif", color="#ccd"),
    margin=dict(l=20, r=20, t=36, b=20),
    xaxis=dict(gridcolor="#ffffff0a", linecolor="#ffffff10", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#ffffff0a", linecolor="#ffffff10", tickfont=dict(size=10)),
)

def _hex_to_rgba(hex_color: str, alpha: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------

def _kpi_card(card_id: str, label: str, icon: str, color: str) -> html.Div:
    return html.Div(className="kpi-card", style={
        "borderLeft": f"3px solid {color}",
        "boxShadow": f"0 0 20px {color}18",
    }, children=[
        html.Div(icon, className="kpi-icon"),
        html.Div(label, className="kpi-label"),
        html.Div("—", id=card_id, className="kpi-value", style={"color": color}),
    ])


def _chart_card(title: str, title_color: str, graph_id: str, height: str = "260px") -> dbc.Card:
    return dbc.Card([
        html.Div(title, className="section-header", style={"color": title_color}),
        dbc.CardBody(dcc.Graph(
            id=graph_id,
            style={"height": height},
            config={"displayModeBar": False},
        ), style={"padding": "8px"}),
    ], className="chart-card")


_FEED_COLS = [
    {"name": "Time",         "id": "ts"},
    {"name": "Threat Class", "id": "predicted_class"},
    {"name": "Confidence",   "id": "confidence"},
    {"name": "Action",       "id": "action"},
    {"name": "Review",       "id": "needs_review"},
    {"name": "Rationale",    "id": "rationale"},
]

_REVIEW_COLS = [
    {"name": "Time",        "id": "ts"},
    {"name": "ID",          "id": "decision_id"},
    {"name": "Class",       "id": "predicted_class"},
    {"name": "Confidence",  "id": "confidence"},
    {"name": "Rationale",   "id": "rationale"},
]

_TABLE_BASE = dict(
    style_table={"overflowX": "auto", "border": "none"},
    style_header={
        "backgroundColor": _PANEL_BG,
        "color": _ACCENT,
        "fontWeight": "600",
        "border": "none",
        "fontSize": "0.68rem",
        "letterSpacing": "1.5px",
        "textTransform": "uppercase",
    },
    style_data={
        "backgroundColor": _CARD_BG,
        "color": "#ccd",
        "border": "none",
    },
)


# ---------------------------------------------------------------------------
# App & layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="SAINT — Network IDS",
)

app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": _DARK_BG, "minHeight": "100vh", "padding": "16px 20px"},
    children=[

        # ── Header ──────────────────────────────────────────────────────────
        dbc.Row(dbc.Col(
            html.Div(className="saint-header d-flex justify-content-between align-items-center", children=[
                html.Div([
                    html.H1("S.A.I.N.T.", className="saint-title"),
                    html.Div("Autonomous Intrusion Detection · Agent-in-the-Loop", className="saint-subtitle"),
                ]),
                html.Div(className="live-badge", children=[
                    html.Div(className="live-dot"),
                    "Live",
                ]),
            ])
        ), className="mb-3"),

        # ── Burst alert ─────────────────────────────────────────────────────
        dbc.Row(dbc.Col(dbc.Alert(
            html.Div(className="burst-alert-body", children=[
                html.Span("⚠", style={"fontSize": "1.1rem"}),
                html.Strong("BURST ATTACK PATTERN DETECTED"),
                html.Span("— sustained malicious traffic in recent window"),
            ]),
            id="burst-alert", color="danger", is_open=False, dismissable=True,
        )), className="mb-3"),

        # ── KPI row ─────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(_kpi_card("kpi-total",  "Total Processed", "⬡", _ACCENT),    width=2),
            dbc.Col(_kpi_card("kpi-allow",  "Allowed",         "✓", "#2ECC71"),  width=2),
            dbc.Col(_kpi_card("kpi-flag",   "Flagged",         "⚑", "#F39C12"),  width=2),
            dbc.Col(_kpi_card("kpi-block",  "Blocked",         "✕", "#E74C3C"),  width=2),
            dbc.Col(_kpi_card("kpi-review", "Needs Review",    "◉", "#9B59B6"),  width=2),
            dbc.Col(_kpi_card("kpi-conf",   "Avg Confidence",  "◎", "#3498DB"),  width=2),
        ], className="mb-3 g-2"),

        # ── Attack timeline ──────────────────────────────────────────────────
        dbc.Row(dbc.Col(
            _chart_card("Attack Timeline", _ACCENT, "timeline-chart", "200px")
        ), className="mb-3"),

        # ── Charts row ──────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(_chart_card("Threat Distribution", "#3498DB", "class-bar"),   width=5),
            dbc.Col(_chart_card("Agent Actions",       "#9B59B6", "action-pie"),  width=3),
            dbc.Col(_chart_card("Confidence",          _ACCENT,   "conf-hist"),   width=4),
        ], className="mb-3 g-2"),

        # ── Review queue + live feed ─────────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div("⚑  Review Queue", className="section-header",
                         style={"color": "#F39C12"}),
                dbc.CardBody(dash_table.DataTable(
                    id="review-table",
                    columns=_REVIEW_COLS,
                    data=[],
                    page_size=8,
                    **_TABLE_BASE,
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": _PANEL_BG},
                    ],
                ), style={"padding": "4px"}),
            ], className="chart-card"), width=5),

            dbc.Col(dbc.Card([
                html.Div("◈  Live Decision Feed", className="section-header",
                         style={"color": _ACCENT}),
                dbc.CardBody(dash_table.DataTable(
                    id="feed-table",
                    columns=_FEED_COLS,
                    data=[],
                    page_size=8,
                    **_TABLE_BASE,
                    style_data_conditional=[
                        {"if": {"filter_query": '{action} = "block"'},
                         "color": "#E74C3C", "fontWeight": "bold"},
                        {"if": {"filter_query": '{action} = "flag"'},  "color": "#F39C12"},
                        {"if": {"filter_query": '{action} = "allow"'}, "color": "#2ECC71"},
                        {"if": {"row_index": "odd"}, "backgroundColor": _PANEL_BG},
                    ],
                ), style={"padding": "4px"}),
            ], className="chart-card"), width=7),
        ], className="mb-3 g-2"),

        # ── Footer ──────────────────────────────────────────────────────────
        dbc.Row(dbc.Col(
            html.Div(id="footer-ts", className="saint-footer")
        )),

        dcc.Interval(id="interval", interval=DASHBOARD_REFRESH_INTERVAL_MS, n_intervals=0),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("timeline-chart", "figure"),
    Output("class-bar",      "figure"),
    Output("action-pie",     "figure"),
    Output("conf-hist",      "figure"),
    Output("feed-table",     "data"),
    Output("review-table",   "data"),
    Output("burst-alert",    "is_open"),
    Output("kpi-total",      "children"),
    Output("kpi-allow",      "children"),
    Output("kpi-flag",       "children"),
    Output("kpi-block",      "children"),
    Output("kpi-review",     "children"),
    Output("kpi-conf",       "children"),
    Output("footer-ts",      "children"),
    Input("interval",        "n_intervals"),
)
def refresh(_):
    decisions = fetch_decisions(500)
    reviews   = fetch_reviews()

    empty = go.Figure()
    empty.update_layout(**_LAYOUT_BASE)
    ts_str = f"Last updated {datetime.now().strftime('%H:%M:%S')}  ·  S.A.I.N.T. v1.0"

    if not decisions:
        return empty, empty, empty, empty, [], [], False, "—","—","—","—","—","—", ts_str

    # ── Single-pass aggregation ──────────────────────────────────────────────
    class_counts:  dict[str, int] = defaultdict(int)
    action_counts: dict[str, int] = defaultdict(int)
    confidences:   list[float]    = []
    burst    = False
    n_review = 0
    reviewed_ids  = set(reviews.keys())
    review_rows: list[dict] = []
    timeline_buckets: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for d in decisions:
        cls = d.get("predicted_class", "unknown")
        act = d.get("action", "flag")
        class_counts[cls]  += 1
        action_counts[act] += 1
        confidences.append(d.get("confidence", 0.0))
        if d.get("needs_review"):
            n_review += 1
            if d.get("decision_id") not in reviewed_ids:
                t = datetime.fromtimestamp(d.get("timestamp", 0)).strftime("%H:%M:%S")
                review_rows.append({
                    "ts":              t,
                    "decision_id":     d.get("decision_id", "")[:12] + "…",
                    "predicted_class": cls,
                    "confidence":      f"{d.get('confidence', 0):.3f}",
                    "rationale":       d.get("rationale", "")[:90] + "…",
                })
        if "burst" in d.get("rationale", "").lower():
            burst = True
        bucket = int(d.get("timestamp", 0) // 10) * 10
        timeline_buckets[bucket][cls] += 1

    # ── Attack timeline ───────────────────────────────────────────────────────
    sorted_buckets = sorted(timeline_buckets.keys())
    x_times = [datetime.fromtimestamp(b).strftime("%H:%M:%S") for b in sorted_buckets]
    timeline_fig = go.Figure()
    for cls in ["normal", "dos", "probe", "r2l", "u2r"]:
        y_vals = [timeline_buckets[b].get(cls, 0) for b in sorted_buckets]
        if any(v > 0 for v in y_vals):
            color = CLASS_COLORS.get(cls, "#888888")
            timeline_fig.add_trace(go.Scatter(
                x=x_times, y=y_vals, name=cls.upper(),
                mode="lines", stackgroup="one",
                line=dict(width=1, color=color),
                fillcolor=_hex_to_rgba(color),
            ))
    timeline_fig.update_layout(
        title=dict(text="Attack Timeline · 10-second buckets", font=dict(size=12)),
        xaxis_title=None, yaxis_title=None,
        legend=dict(orientation="h", y=1.18, font=dict(size=10)),
        **_LAYOUT_BASE,
    )

    # ── Class bar chart ───────────────────────────────────────────────────────
    bar_fig = go.Figure(go.Bar(
        x=list(class_counts.keys()),
        y=list(class_counts.values()),
        marker=dict(
            color=[CLASS_COLORS.get(k, "#888") for k in class_counts],
            line=dict(width=0),
        ),
        text=list(class_counts.values()),
        textposition="outside",
        textfont=dict(size=11),
    ))
    bar_fig.update_layout(
        title=dict(text="Threat Distribution", font=dict(size=12)),
        showlegend=False,
        **_LAYOUT_BASE,
    )

    # ── Action pie ────────────────────────────────────────────────────────────
    pie_fig = go.Figure(go.Pie(
        labels=list(action_counts.keys()),
        values=list(action_counts.values()),
        marker=dict(
            colors=[ACTION_COLORS.get(k, "#888") for k in action_counts],
            line=dict(color=_CARD_BG, width=2),
        ),
        hole=0.55,
        textinfo="percent",
        textfont=dict(size=11),
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    pie_fig.update_layout(
        title=dict(text="Agent Actions", font=dict(size=12)),
        showlegend=True,
        legend=dict(font=dict(size=10), orientation="v"),
        paper_bgcolor=_CARD_BG,
        font=dict(family="Inter, sans-serif", color="#ccd"),
        margin=dict(l=10, r=10, t=36, b=10),
    )

    # ── Confidence histogram ──────────────────────────────────────────────────
    hist_fig = go.Figure(go.Histogram(
        x=confidences, nbinsx=20,
        marker=dict(color=_ACCENT, opacity=0.8, line=dict(width=0)),
    ))
    hist_fig.update_layout(
        title=dict(text="Confidence Distribution", font=dict(size=12)),
        bargap=0.05,
        **_LAYOUT_BASE,
    )

    # ── Feed table ────────────────────────────────────────────────────────────
    feed_rows = [{
        "ts":              datetime.fromtimestamp(d.get("timestamp", 0)).strftime("%H:%M:%S"),
        "predicted_class": d.get("predicted_class", ""),
        "confidence":      f"{d.get('confidence', 0):.3f}",
        "action":          d.get("action", ""),
        "needs_review":    "YES" if d.get("needs_review") else "—",
        "rationale":       d.get("rationale", "")[:100] + "…",
    } for d in decisions[:50]]

    avg_conf = f"{sum(confidences)/len(confidences):.3f}"

    return (
        timeline_fig, bar_fig, pie_fig, hist_fig,
        feed_rows, review_rows,
        burst,
        str(len(decisions)),
        str(action_counts.get("allow", 0)),
        str(action_counts.get("flag",  0)),
        str(action_counts.get("block", 0)),
        str(n_review),
        avg_conf,
        ts_str,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, port=DASH_PORT)
