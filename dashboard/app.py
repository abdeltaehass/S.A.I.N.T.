"""
SAINT — Agent-in-the-loop monitoring dashboard (Plotly Dash).

Panels:
  • KPI cards: total / allowed / flagged / blocked / review queue
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
# Color palettes
# ---------------------------------------------------------------------------

ACTION_COLORS = {"allow": "#2ECC71", "flag": "#F39C12", "block": "#E74C3C"}
CLASS_COLORS  = {
    "normal": "#3498DB",
    "dos":    "#E74C3C",
    "probe":  "#F39C12",
    "r2l":    "#9B59B6",
    "u2r":    "#1ABC9C",
}
_DARK_BG   = "#0d0d1a"
_CARD_BG   = "#16213e"
_PANEL_BG  = "#1a1a2e"
_ACCENT    = "#00F5D4"

_LAYOUT_BASE = dict(
    paper_bgcolor=_CARD_BG,
    plot_bgcolor=_PANEL_BG,
    font_color="#eee",
    margin=dict(l=20, r=20, t=40, b=20),
)

# ---------------------------------------------------------------------------
# Reusable components
# ---------------------------------------------------------------------------

def _kpi_card(card_id: str, title: str, color: str = _ACCENT) -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6(title, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
        html.H3("—", id=card_id, style={"color": color, "marginBottom": 0}),
    ]), style={"backgroundColor": _CARD_BG, "border": f"1px solid {color}22"})


_FEED_COLS = [
    {"name": "Time",         "id": "ts"},
    {"name": "Threat Class", "id": "predicted_class"},
    {"name": "Confidence",   "id": "confidence"},
    {"name": "Action",       "id": "action"},
    {"name": "Review",       "id": "needs_review"},
    {"name": "Rationale",    "id": "rationale"},
]

_REVIEW_COLS = [
    {"name": "Time",         "id": "ts"},
    {"name": "Decision ID",  "id": "decision_id"},
    {"name": "Pred. Class",  "id": "predicted_class"},
    {"name": "Confidence",   "id": "confidence"},
    {"name": "Rationale",    "id": "rationale"},
]

_TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_header={"backgroundColor": _PANEL_BG, "color": _ACCENT, "fontWeight": "bold"},
    style_data={"backgroundColor": _CARD_BG, "color": "#eee", "fontSize": "0.82rem"},
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="SAINT — Network IDS",
)

app.layout = dbc.Container(fluid=True, style={"backgroundColor": _DARK_BG, "minHeight": "100vh", "padding": "12px"}, children=[

    # ── Header ──────────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.H2(
        "S.A.I.N.T. — Network Intrusion Detection",
        className="text-center my-3",
        style={"color": _ACCENT, "letterSpacing": "2px"},
    ))),

    # ── Burst alert ─────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(dbc.Alert(
        [html.Strong("BURST ATTACK DETECTED  "), "— sustained malicious traffic pattern in recent window"],
        id="burst-alert", color="danger", is_open=False, dismissable=True,
        style={"textAlign": "center"},
    ))),

    # ── KPI row ─────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(_kpi_card("kpi-total",   "Total Processed",   _ACCENT),     width=2),
        dbc.Col(_kpi_card("kpi-allow",   "Allowed",           "#2ECC71"),   width=2),
        dbc.Col(_kpi_card("kpi-flag",    "Flagged",           "#F39C12"),   width=2),
        dbc.Col(_kpi_card("kpi-block",   "Blocked",           "#E74C3C"),   width=2),
        dbc.Col(_kpi_card("kpi-review",  "Needs Review",      "#9B59B6"),   width=2),
        dbc.Col(_kpi_card("kpi-conf",    "Avg Confidence",    "#3498DB"),   width=2),
    ], className="mb-3 g-2"),

    # ── Attack timeline ──────────────────────────────────────────────────────
    dbc.Row(dbc.Col(dbc.Card(dbc.CardBody(
        dcc.Graph(id="timeline-chart", style={"height": "220px"}),
    ), style={"backgroundColor": _CARD_BG})), className="mb-3"),

    # ── Charts row ──────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="class-bar",   style={"height": "260px"})),
                         style={"backgroundColor": _CARD_BG}), width=5),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="action-pie",  style={"height": "260px"})),
                         style={"backgroundColor": _CARD_BG}), width=3),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="conf-hist",   style={"height": "260px"})),
                         style={"backgroundColor": _CARD_BG}), width=4),
    ], className="mb-3 g-2"),

    # ── Review queue + live feed ─────────────────────────────────────────────
    dbc.Row([
        # Review queue (flagged, awaiting analyst)
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.Strong("Review Queue", style={"color": "#F39C12"}),
                           style={"backgroundColor": _PANEL_BG}),
            dbc.CardBody(dash_table.DataTable(
                id="review-table",
                columns=_REVIEW_COLS,
                data=[],
                page_size=8,
                **_TABLE_STYLE,
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": _PANEL_BG},
                ],
            )),
        ], style={"backgroundColor": _CARD_BG}), width=5),

        # Live feed
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.Strong("Live Decision Feed", style={"color": _ACCENT}),
                           style={"backgroundColor": _PANEL_BG}),
            dbc.CardBody(dash_table.DataTable(
                id="feed-table",
                columns=_FEED_COLS,
                data=[],
                page_size=8,
                **_TABLE_STYLE,
                style_data_conditional=[
                    {"if": {"filter_query": '{action} = "block"'},
                     "color": "#E74C3C", "fontWeight": "bold"},
                    {"if": {"filter_query": '{action} = "flag"'},  "color": "#F39C12"},
                    {"if": {"filter_query": '{action} = "allow"'}, "color": "#2ECC71"},
                    {"if": {"row_index": "odd"}, "backgroundColor": _PANEL_BG},
                ],
            )),
        ], style={"backgroundColor": _CARD_BG}), width=7),
    ], className="mb-3 g-2"),

    dcc.Interval(id="interval", interval=DASHBOARD_REFRESH_INTERVAL_MS, n_intervals=0),
])


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
    Input("interval",        "n_intervals"),
)
def refresh(_n):
    decisions = fetch_decisions(500)
    reviews   = fetch_reviews()

    empty = go.Figure()
    empty.update_layout(**_LAYOUT_BASE)

    if not decisions:
        return empty, empty, empty, empty, [], [], False, "—","—","—","—","—","—"

    # ── Aggregate ────────────────────────────────────────────────────────────
    class_counts:  dict[str, int]   = defaultdict(int)
    action_counts: dict[str, int]   = defaultdict(int)
    confidences:   list[float]      = []
    burst = False
    n_review = 0

    # bucket by 10-second intervals for timeline
    timeline_buckets: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for d in decisions:
        cls = d.get("predicted_class", "unknown")
        act = d.get("action", "flag")
        class_counts[cls]  += 1
        action_counts[act] += 1
        confidences.append(d.get("confidence", 0.0))
        if d.get("needs_review"):
            n_review += 1
        if "burst" in d.get("rationale", "").lower():
            burst = True
        ts = d.get("timestamp", 0)
        bucket = int(ts // 10) * 10
        timeline_buckets[bucket][cls] += 1

    # ── Attack timeline (stacked area) ───────────────────────────────────────
    all_classes = ["normal", "dos", "probe", "r2l", "u2r"]
    sorted_buckets = sorted(timeline_buckets.keys())
    x_times = [datetime.fromtimestamp(b).strftime("%H:%M:%S") for b in sorted_buckets]

    timeline_fig = go.Figure()
    for cls in all_classes:
        y_vals = [timeline_buckets[b].get(cls, 0) for b in sorted_buckets]
        if any(v > 0 for v in y_vals):
            timeline_fig.add_trace(go.Scatter(
                x=x_times, y=y_vals,
                name=cls.upper(),
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5, color=CLASS_COLORS.get(cls, "#888")),
                fillcolor=CLASS_COLORS.get(cls, "#888") + "99",
            ))
    timeline_fig.update_layout(
        title="Attack Timeline (10-second buckets)",
        xaxis_title="Time", yaxis_title="Connections",
        legend=dict(orientation="h", y=1.15),
        **_LAYOUT_BASE,
    )

    # ── Class bar chart ───────────────────────────────────────────────────────
    bar_fig = go.Figure(go.Bar(
        x=list(class_counts.keys()),
        y=list(class_counts.values()),
        marker_color=[CLASS_COLORS.get(k, "#888") for k in class_counts],
        text=list(class_counts.values()),
        textposition="outside",
    ))
    bar_fig.update_layout(title="Threat Class Distribution", **_LAYOUT_BASE)

    # ── Action pie ────────────────────────────────────────────────────────────
    pie_fig = go.Figure(go.Pie(
        labels=list(action_counts.keys()),
        values=list(action_counts.values()),
        marker_colors=[ACTION_COLORS.get(k, "#888") for k in action_counts],
        hole=0.45,
        textinfo="percent+label",
    ))
    pie_fig.update_layout(title="Agent Actions", showlegend=False, **_LAYOUT_BASE)

    # ── Confidence histogram ──────────────────────────────────────────────────
    hist_fig = go.Figure(go.Histogram(
        x=confidences, nbinsx=20,
        marker_color=_ACCENT, opacity=0.85,
    ))
    hist_fig.update_layout(title="Confidence Distribution",
                           xaxis_title="Confidence", yaxis_title="Count",
                           **_LAYOUT_BASE)

    # ── Feed table (most recent first) ────────────────────────────────────────
    feed_rows = []
    for d in decisions[:50]:
        ts_str = datetime.fromtimestamp(d.get("timestamp", 0)).strftime("%H:%M:%S")
        feed_rows.append({
            "ts":              ts_str,
            "predicted_class": d.get("predicted_class", ""),
            "confidence":      f"{d.get('confidence', 0):.3f}",
            "action":          d.get("action", ""),
            "needs_review":    "YES" if d.get("needs_review") else "—",
            "rationale":       d.get("rationale", "")[:100] + "…",
        })

    # ── Review queue (flagged, not yet reviewed by analyst) ───────────────────
    reviewed_ids = set(reviews.keys())
    review_rows = []
    for d in decisions:
        if d.get("needs_review") and d.get("decision_id") not in reviewed_ids:
            ts_str = datetime.fromtimestamp(d.get("timestamp", 0)).strftime("%H:%M:%S")
            review_rows.append({
                "ts":              ts_str,
                "decision_id":     d.get("decision_id", "")[:12] + "…",
                "predicted_class": d.get("predicted_class", ""),
                "confidence":      f"{d.get('confidence', 0):.3f}",
                "rationale":       d.get("rationale", "")[:90] + "…",
            })

    avg_conf = f"{sum(confidences)/len(confidences):.3f}" if confidences else "—"

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
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, port=DASH_PORT)
