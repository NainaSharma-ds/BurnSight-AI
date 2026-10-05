import plotly.graph_objects as go

NAVY_CARD = "#111a2e"
GRID = "#1e2a44"
TEXT = "#c7d2e5"
CYAN, AMBER, RED, GREEN = "#22d3ee", "#fbbf24", "#f87171", "#34d399"


def style(fig, height=420):
    """Common dark look used by every chart."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=NAVY_CARD,
        font=dict(color=TEXT),
        margin=dict(l=10, r=10, t=90, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hoverlabel=dict(namelength=-1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def prediction_chart(trend, threshold):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["Hour"], y=trend["Actual"], name="Actual (burn-in)",
        mode="lines+markers", line=dict(color=CYAN, width=3), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=trend["Hour"], y=trend["Predicted"], name="Predicted",
        mode="lines+markers", line=dict(color=AMBER, width=3, dash="dash"), marker=dict(size=8),
    ))
    fig.add_hline(y=threshold, line_dash="dot", line_color=RED,
                  annotation_text="Acceptance threshold", annotation_font_color=RED)
    fig.update_xaxes(
        title="Burn-in hours", tickmode="array",
        tickvals=[0, 24, 48, 96, 168], ticktext=["0h", "24h", "48h", "96h", "168h"],
    )
    fig.update_yaxes(title="Performance value")
    fig.update_layout(title="168-Hour Prediction Trend")
    return style(fig)



def anomaly_scatter(population, unit_xy, anomaly_score, unit_color):
    if {"x", "y"}.issubset(population.columns):
        x_values, y_values = population["x"], population["y"]
        x_unit, y_unit = unit_xy[0], unit_xy[1]
        x_title, y_title = "Temperature drift (scaled)", "Current drift (scaled)"
        scaled = True
    elif {"Value_0h", "Value_24h"}.issubset(population.columns):
        x_values, y_values = population["Value_0h"], population["Value_24h"]
        x_unit, y_unit = unit_xy[0], unit_xy[1]
        x_title, y_title = "0h performance value", "24h performance value"
        scaled = False
    else:
        raise ValueError("Population data must contain x/y or Value_0h/Value_24h.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, y=y_values, mode="markers", name="Units in selected lot",
        marker=dict(color=CYAN, size=7, opacity=0.45),
        hovertemplate="Dataset unit<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[x_unit], y=[y_unit], mode="markers", name="Selected unit",
        marker=dict(color=unit_color, size=17, symbol="star", line=dict(color="white", width=1)),
        hovertemplate=f"Selected unit<br>Anomaly score: {anomaly_score}<extra></extra>",
    ))

    if scaled:
        fig.add_shape(type="circle", x0=-3, y0=-3, x1=3, y1=3,
                      line=dict(color=RED, dash="dot", width=1.5))
        fig.update_xaxes(title=x_title, range=[-5, 5])
        fig.update_yaxes(title=y_title, range=[-5, 5], scaleanchor="x")
    else:
        x_min, x_max = min(min(x_values), x_unit), max(max(x_values), x_unit)
        y_min, y_max = min(min(y_values), y_unit), max(max(y_values), y_unit)
        x_padding = max((x_max - x_min) * 0.1, abs(x_min) * 0.01, 0.01)
        y_padding = max((y_max - y_min) * 0.1, abs(y_min) * 0.01, 0.01)
        fig.update_xaxes(title=x_title, range=[x_min - x_padding, x_max + x_padding])
        fig.update_yaxes(title=y_title, range=[y_min - y_padding, y_max + y_padding])

    fig.update_layout(title="Normal vs Anomalous Units")
    style(fig)
    fig.update_layout(hovermode="closest")
    return fig



def risk_gauge(value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix="%", font=dict(color="#f1f5f9", size=46)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=GRID, tickfont=dict(color=TEXT)),
            bar=dict(color=color, thickness=0.28),
            bgcolor=NAVY_CARD,
            borderwidth=0,
            steps=[
                dict(range=[0, 35], color="rgba(52,211,153,0.15)"),
                dict(range=[35, 60], color="rgba(34,211,238,0.12)"),
                dict(range=[60, 80], color="rgba(251,191,36,0.15)"),
                dict(range=[80, 100], color="rgba(248,113,113,0.18)"),
            ],
        ),
    ))
    fig.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=TEXT), margin=dict(l=25, r=25, t=30, b=10))
    return fig



def importance_chart(importance):
    fig = go.Figure(go.Bar(
        x=importance["Importance"], y=importance["Feature"], orientation="h",
        marker=dict(color=CYAN),
        hovertemplate="%{y}<br>Importance: %{x}<extra></extra>",
    ))
    fig.update_xaxes(title="Importance (drop in model accuracy when shuffled)")
    fig.update_layout(title="Permutation Feature Importance")
    style(fig, height=340)
    fig.update_layout(hovermode="closest")
    return fig


def lime_chart(contributions):
    data = contributions.iloc[::-1]          # biggest on top
    colors = [RED if v > 0 else GREEN for v in data["Contribution"]]
    limit = max(abs(data["Contribution"])) * 1.4      # extra room so labels are not cut
    fig = go.Figure(go.Bar(
        x=data["Contribution"], y=data["Feature"], orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.2f}" for v in data["Contribution"]], textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Contribution: %{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=TEXT, line_width=1)
    fig.update_xaxes(title="Pushes risk down  ←  →  pushes risk up", range=[-limit, limit])
    fig.update_layout(title="LIME Feature Contribution")
    style(fig, height=340)
    fig.update_layout(hovermode="closest")
    return fig



def trend_line(history, column, title, color, threshold=None):
    fig = go.Figure(go.Scatter(
        x=history["Time"], y=history[column], mode="lines+markers",
        line=dict(color=color, width=3), marker=dict(size=7),
        hovertemplate="%{x|%d %b, %H:%M}<br>" + column + ": %{y}<extra></extra>",
    ))
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dot", line_color=RED)
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(title=title)
    style(fig, height=280)
    fig.update_layout(hovermode="closest", showlegend=False,
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig