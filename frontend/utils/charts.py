import plotly.graph_objects as go

NAVY_CARD = "#111a2e"
GRID = "#1e2a44"
TEXT = "#c7d2e5"
CYAN, AMBER, RED = "#22d3ee", "#fbbf24", "#f87171"


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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=population["x"], y=population["y"], mode="markers", name="Normal units",
        marker=dict(color=CYAN, size=7, opacity=0.45),
        hovertemplate="Normal unit<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[unit_xy[0]], y=[unit_xy[1]], mode="markers", name="Selected unit",
        marker=dict(color=unit_color, size=17, symbol="star", line=dict(color="white", width=1)),
        hovertemplate=f"Selected unit<br>Anomaly score: {anomaly_score}<extra></extra>",
    ))
    # dotted circle = "normal" boundary
    fig.add_shape(type="circle", x0=-3, y0=-3, x1=3, y1=3,
                  line=dict(color=RED, dash="dot", width=1.5))
    fig.update_xaxes(title="Temperature drift (scaled)", range=[-5, 5])
    fig.update_yaxes(title="Current drift (scaled)", range=[-5, 5], scaleanchor="x")
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