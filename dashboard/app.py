"""
SAINT — Agent-in-the-loop monitoring dashboard (Plotly Dash).

Panels:
  • Live threat feed table (color-coded by action)
  • Threat class distribution (bar chart, live)
  • Action breakdown (pie chart)
  • Confidence histogram
  • Burst alert banner
"""

import json
import time
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import redis
from dash import Input, Output, State, dash_table, dcc, html

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DASH_PORT, DASHBOARD_REFRESH_INTERVAL_MS,
    REDIS_DB, REDIS_HOST, REDIS_PORT,
)

# ---------------------------------------------------------------------------
# Redis connection
# ---------------------------------------------------------------------------

def get_redis() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


def fetch_decisions(n: int = 200) -> list[dict]:
    try:
        r = get_redis()
        raw = r.lrange("saint:decisions", 0, n - 1)
        return [json.loads(item) for item in raw]
    except redis.RedisError:
        return []


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------

ACTION_COLORS = {"allow": "#2ECC71", "flag": "#F39C12", "block": "#E74C3C"}
CLASS_COLORS  = {
    "normal": "#3498DB",
    "dos":    "#E74C3C",
    "probe":  "#F39C12",
    "r2l":    "#9B59B6",
    "u2r":    "#1ABC9C",
}

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="SAINT — Network IDS",
)

def _kpi_card(card_id: str, title: str, value: str, color: str = "#00F5D4") -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6(title, className="text-muted mb-1"),
        html.H3(value, id=card_id, style={"color": color}),
    ]), style={"backgroundColor": "#16213e"})


_table_cols = [
    {"name": "Time", "id": "ts"},
    {"name": "Threat Class", "id": "predicted_class"},
    {"name": "Confidence", "id": "confidence"},
    {"name": "Action", "id": "action"},
    {"name": "Needs Review", "id": "needs_review"},
    {"name": "Rationale", "id": "rationale"},
]

app.layout = dbc.Container(
    fluid=True,
    children=[
        # Header
        dbc.Row(dbc.Col(html.H2(
            "S.A.I.N.T. — Network Intrusion Detection",
            className="text-center my-3",
            style={"color": "#00F5D4", "letterSpacing": "2px"},
        ))),

        # Burst alert banner (hidden by default)
        dbc.Row(dbc.Col(
            dbc.Alert(
                "BURST ATTACK PATTERN DETECTED — elevated traffic anomaly",
                id="burst-alert",
                color="danger",
                is_open=False,
                dismissable=True,
                style={"textAlign": "center", "fontWeight": "bold"},
            )
        )),

        # KPI cards
        dbc.Row([
            dbc.Col(_kpi_card("total-kpi",   "Total Processed", "—"), width=3),
            dbc.Col(_kpi_card("allow-kpi",   "Allowed",         "—", color="#2ECC71"), width=3),
            dbc.Col(_kpi_card("flag-kpi",    "Flagged",         "—", color="#F39C12"), width=3),
            dbc.Col(_kpi_card("block-kpi",   "Blocked",         "—", color="#E74C3C"), width=3),
        ], className="mb-3"),

        # Charts row
        dbc.Row([
            dbc.Col(dcc.Graph(id="class-bar"),  width=6),
            dbc.Col(dcc.Graph(id="action-pie"), width=3),
            dbc.Col(dcc.Graph(id="conf-hist"),  width=3),
        ], className="mb-3"),

        # Live feed table
        dbc.Row(dbc.Col(
            dash_table.DataTable(
                id="feed-table",
                columns=_table_cols,
                data=[],
                page_size=20,
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#1a1a2e",
                    "color": "#00F5D4",
                    "fontWeight": "bold",
                },
                style_data={"backgroundColor": "#16213e", "color": "#eee"},
                style_data_conditional=[
                    {"if": {"filter_query": '{action} = "block"'},
                     "color": "#E74C3C", "fontWeight": "bold"},
                    {"if": {"filter_query": '{action} = "flag"'},
                     "color": "#F39C12"},
                    {"if": {"filter_query": '{action} = "allow"'},
                     "color": "#2ECC71"},
                ],
            )
        )),

        # Polling interval
        dcc.Interval(id="interval", interval=DASHBOARD_REFRESH_INTERVAL_MS, n_intervals=0),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("class-bar",   "figure"),
    Output("action-pie",  "figure"),
    Output("conf-hist",   "figure"),
    Output("feed-table",  "data"),
    Output("burst-alert", "is_open"),
    Output("total-kpi",   "children"),
    Output("allow-kpi",   "children"),
    Output("flag-kpi",    "children"),
    Output("block-kpi",   "children"),
    Input("interval",     "n_intervals"),
)
def refresh(_n):
    decisions = fetch_decisions(200)

    if not decisions:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor="#16213e", plot_bgcolor="#16213e", font_color="#eee"
        )
        return empty_fig, empty_fig, empty_fig, [], False, "—", "—", "—", "—"

    # --- Aggregate ---
    class_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {"allow": 0, "flag": 0, "block": 0}
    confidences: list[float] = []
    burst = False

    for d in decisions:
        cls = d.get("predicted_class", "unknown")
        act = d.get("action", "flag")
        class_counts[cls] = class_counts.get(cls, 0) + 1
        action_counts[act] = action_counts.get(act, 0) + 1
        confidences.append(d.get("confidence", 0.0))
        if "burst" in d.get("rationale", "").lower():
            burst = True

    # --- Class bar chart ---
    bar_fig = go.Figure(go.Bar(
        x=list(class_counts.keys()),
        y=list(class_counts.values()),
        marker_color=[CLASS_COLORS.get(k, "#888") for k in class_counts],
    ))
    bar_fig.update_layout(
        title="Threat Class Distribution",
        paper_bgcolor="#16213e", plot_bgcolor="#16213e",
        font_color="#eee", margin=dict(l=20, r=20, t=40, b=20),
    )

    # --- Action pie ---
    pie_fig = go.Figure(go.Pie(
        labels=list(action_counts.keys()),
        values=list(action_counts.values()),
        marker_colors=[ACTION_COLORS.get(k, "#888") for k in action_counts],
        hole=0.4,
    ))
    pie_fig.update_layout(
        title="Actions",
        paper_bgcolor="#16213e", font_color="#eee",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # --- Confidence histogram ---
    hist_fig = go.Figure(go.Histogram(
        x=confidences, nbinsx=20,
        marker_color="#00F5D4", opacity=0.8,
    ))
    hist_fig.update_layout(
        title="Confidence Distribution",
        paper_bgcolor="#16213e", plot_bgcolor="#16213e",
        font_color="#eee", margin=dict(l=20, r=20, t=40, b=20),
    )

    # --- Feed table ---
    table_rows = []
    for d in decisions[:100]:
        ts = time.strftime("%H:%M:%S", time.localtime(d.get("timestamp", 0)))
        table_rows.append({
            "ts":              ts,
            "predicted_class": d.get("predicted_class", ""),
            "confidence":      f"{d.get('confidence', 0):.3f}",
            "action":          d.get("action", ""),
            "needs_review":    "YES" if d.get("needs_review") else "no",
            "rationale":       d.get("rationale", "")[:120] + "…",
        })

    total    = len(decisions)
    n_allow  = action_counts.get("allow", 0)
    n_flag   = action_counts.get("flag",  0)
    n_block  = action_counts.get("block", 0)

    return (
        bar_fig, pie_fig, hist_fig, table_rows,
        burst,
        str(total), str(n_allow), str(n_flag), str(n_block),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, port=DASH_PORT)
