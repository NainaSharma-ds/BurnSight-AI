from datetime import datetime
import streamlit as st
import time
from utils.demo_data import get_demo_data
from utils.report import build_report
from utils.api import get_api_data, backend_online
from utils.charts import prediction_chart, anomaly_scatter, risk_gauge, importance_chart, lime_chart, trend_line
USE_API = False     # False = demo data (works anywhere).  True = real FastAPI backend.
PREDICTED_UNIT = ""     # unit of the predicted 168h value. Not defined yet, so left empty. Ask the model team.

def demo_note():
    if not USE_API:
        st.caption("Demo values: sample data, not real model output.")
st.set_page_config(page_title="SIH26170 | AI Risk Monitoring", layout="wide")

# ---------- STYLES (all custom CSS lives here) ----------
CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1400px; }

.header { display: flex; justify-content: space-between; align-items: flex-end;
          flex-wrap: wrap; gap: 1rem; padding-bottom: 1.2rem;
          border-bottom: 1px solid #1e2a44; margin-bottom: 1.5rem; }
.project-id { color: #22d3ee; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.25em; }
.title { font-size: 2rem; font-weight: 700; color: #f1f5f9; margin: 0.2rem 0; }
.tagline { color: #8a9bb8; font-size: 0.9rem; }

.pills { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
.pill { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
        padding: 0.35rem 0.8rem; border-radius: 999px; border: 1px solid; }
.pill-demo { color: #fbbf24; border-color: #fbbf2455; background: #fbbf2414; }
.pill-ok   { color: #34d399; border-color: #34d39955; background: #34d39914; }
.pill-time { color: #8a9bb8; border-color: #1e2a44; background: #111a2e; }

.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
       background: #34d399; margin-right: 6px; animation: pulse 2.5s infinite; }
@keyframes pulse { 0% {opacity: 1;} 50% {opacity: 0.3;} 100% {opacity: 1;} }

.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 1rem; margin-bottom: 1.5rem; }
.kpi { background: #111a2e; border: 1px solid #1e2a44; border-top: 3px solid var(--accent);
       border-radius: 10px; padding: 1.1rem 1.2rem;
       animation: fadeUp 0.5s ease both; transition: transform 0.2s, box-shadow 0.2s; }
.kpi:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.35); }
.kpi-label { color: #8a9bb8; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; }
.kpi-value { color: #f1f5f9; font-size: clamp(1.4rem, 2.2vw, 2.2rem); font-weight: 700; margin: 0.3rem 0; white-space: nowrap; }
.kpi-sub { font-size: 0.82rem; font-weight: 600; }
@keyframes fadeUp { from {opacity: 0; transform: translateY(8px);} to {opacity: 1; transform: none;} }

.section-title { color: #8a9bb8; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.2em;
                 text-transform: uppercase; margin: 2rem 0 0.8rem 0; }
.banner { background: #111a2e; border: 1px solid #1e2a44; border-left: 4px solid var(--accent);
          border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
.banner-title { color: var(--accent); font-weight: 700; letter-spacing: 0.08em; font-size: 1.05rem; }
.banner-text { color: #c7d2e5; font-size: 0.88rem; margin-top: 0.35rem; line-height: 1.5; }

.meter { background: #111a2e; border: 1px solid #1e2a44; border-radius: 10px;
         padding: 0.9rem 1.1rem; margin-bottom: 0.8rem; }
.meter-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.55rem; }
.meter-name { color: #c7d2e5; font-size: 0.9rem; font-weight: 600; }
.meter-status { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; }
.meter-track { height: 8px; background: #1e2a44; border-radius: 999px; overflow: hidden; }
.meter-fill { height: 100%; border-radius: 999px; background: var(--accent);
              width: var(--pct); animation: grow 0.9s ease; }
@keyframes grow { from {width: 0;} }

.decision { background: #111a2e; border: 1px solid var(--accent); border-radius: 12px;
            padding: 1.4rem 1.5rem; box-shadow: 0 0 24px -8px var(--accent);
            animation: fadeUp 0.5s ease both; }
.decision-label { color: var(--accent); font-size: 2.6rem; font-weight: 800;
                  letter-spacing: 0.06em; margin: 0.4rem 0 0.6rem 0; }
.rules-box { background: #111a2e; border: 1px solid #1e2a44; border-radius: 12px;
             padding: 1.2rem 1.4rem; }
.rules-box .kpi-label { margin-bottom: 0.4rem; }
.rule { display: flex; gap: 0.7rem; align-items: flex-start; color: #c7d2e5;
        font-size: 0.92rem; padding: 0.55rem 0; border-bottom: 1px solid #1e2a44; }
.rule:last-child { border-bottom: none; }
.rule-icon { font-weight: 700; width: 1.1rem; }

.why { background: #111a2e; border: 1px solid #1e2a44; border-left: 4px solid #22d3ee;
       border-radius: 10px; padding: 1rem 1.3rem; margin-top: 0.5rem; }
.why-title { color: #22d3ee; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.15em;
             text-transform: uppercase; }
.why-text { color: #e6edf7; font-size: 1.02rem; line-height: 1.6; margin-top: 0.4rem; }

/* ---- POLISH OVERRIDES ---- */
.section-title { font-size: 0.85rem; color: #9fb0cc; margin: 2rem 0 0.8rem 0; }
.kpi-label { font-size: 0.78rem; }
.kpi-sub { font-size: 0.85rem; }
.pill { font-size: 0.75rem; }
.pill-bad { color: #f87171; border-color: #f8717155; background: #f8717114; }
.dot { background: currentColor; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- HEADER ----------
now = datetime.now().strftime("%d %b %Y, %H:%M:%S")
if not USE_API:
    mode_pill = '<span class="pill pill-demo">DEMO MODE</span>'
    backend_pill = '<span class="pill pill-demo"><span class="dot"></span>Backend: Not connected</span>'
elif backend_online():
    mode_pill = '<span class="pill pill-ok">LIVE MODE</span>'
    backend_pill = '<span class="pill pill-ok"><span class="dot"></span>Backend connected</span>'
else:
    mode_pill = '<span class="pill pill-bad">LIVE MODE</span>'
    backend_pill = '<span class="pill pill-bad"><span class="dot"></span>Backend unreachable</span>'

st.markdown(f"""
<div class="header">
<div>
<div class="project-id">SIH26170</div>
<div class="title">AI Predictive Risk Monitoring</div>
<div class="tagline">Burn-in intelligence • Anomaly detection • 168h prediction • QA decision support</div>
</div>
<div class="pills">
{mode_pill}
{backend_pill}
<span class="pill pill-time">Updated {now}</span>
</div>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### Controls")

    component = st.selectbox(
        "Component",
        ["COMP-001", "COMP-002", "COMP-003"],
        help="Which component's burn-in data to analyse.",
    )
    lot = st.selectbox(
        "Lot",
        ["LOT-2026-A", "LOT-2026-B"],
        help="Manufacturing lot the component belongs to.",
    )
    time_window = st.select_slider(
        "Time Window",
        options=["0h", "24h", "48h", "96h", "168h"],
        value="168h",
        help="How many burn-in hours of data to include.",
    )
    mode = st.radio(
        "Analysis mode",
        ["Overview", "Detailed Analysis"],
        help="Detailed Analysis expands the technical sections.",
    )

    run_clicked = st.button("Run Analysis", type="primary", width="stretch")

    st.divider()
    st.caption("SYSTEM INFORMATION")
    mode_text = "Live backend" if USE_API else "Demo data"
    backend_text = "Live (API)" if USE_API else "Not connected"
    st.markdown(
        f"""
- **Mode:** {mode_text}
- **Backend:** {backend_text}
- **Module A:** Mahalanobis + LOF
- **Module B:** XGBoost
"""
    )

# ---------- LOAD DATA (demo for now) ----------
if USE_API:
    try:
        data = get_api_data(component, lot, time_window)
    except Exception as error:
        st.error(f"Could not load data from the backend ({error}). "
                 "Check that the FastAPI server is running, or set USE_API = False in app.py.")
        st.stop()
else:
    data = get_demo_data(component, lot, time_window)
detailed = (mode == "Detailed Analysis")

# ---------- RUN ANALYSIS FEEDBACK ----------
if run_clicked:
    with st.status("Running analysis...", expanded=True) as status:
        for step in ["Loading burn-in data", "Preprocessing features",
                     "Module A: anomaly detection", "Module B: 168h prediction",
                     "Risk engine and QA decision", "Generating explanations"]:
            st.write(f"✓ {step}")
            time.sleep(0.4)
        status.update(label="Analysis complete (demo data)", state="complete", expanded=False)

# ---------- KPI CARDS ----------
GREEN, AMBER, ORANGE, RED, CYAN = "#34d399", "#fbbf24", "#fb923c", "#f87171", "#22d3ee"

RISK_COLORS = {"Low": GREEN, "Moderate": CYAN, "Elevated": AMBER, "Critical": RED}
DECISION_COLORS = {"PASS": GREEN, "MONITOR": AMBER, "EXTEND": CYAN, "REJECT": RED}


def kpi_card(label, value, sub, color, tip, delay):
    return (
        f'<div class="kpi" title="{tip}" style="--accent:{color}; animation-delay:{delay}s">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub" style="color:{color}">{sub}</div>'
        f'</div>'
    )


anomaly = data["anomaly_score"]
if anomaly > 3:
    anomaly_sub, anomaly_color = "High deviation", RED
elif anomaly > 2:
    anomaly_sub, anomaly_color = "Elevated deviation", AMBER
else:
    anomaly_sub, anomaly_color = "Within normal range", GREEN

predicted = data["predicted_168h"]
if predicted >= data["threshold"]:
    pred_sub, pred_color = f'Above threshold ({data["threshold"]})', GREEN
else:
    pred_sub, pred_color = f'Below threshold ({data["threshold"]})', RED

cards = [
    kpi_card("Overall Risk Score", f'{data["risk_score"]}%', data["risk_label"],
             RISK_COLORS[data["risk_label"]],
             "Combined risk from anomaly, drift and prediction signals.", 0.0),
    kpi_card("Predicted 168h Value", f"{predicted}{PREDICTED_UNIT}", pred_sub, pred_color,
             "Model forecast of the performance value after 168 burn-in hours.", 0.1),
    kpi_card("Anomaly Score", anomaly, anomaly_sub, anomaly_color,
             "How unusual this unit is compared with normal units. Higher means more unusual.", 0.2),
    kpi_card("QA Decision", data["qa_decision"], "Recommended action",
             DECISION_COLORS[data["qa_decision"]],
             "PASS / MONITOR / EXTEND / REJECT, decided by the QA rule engine.", 0.3),
]

st.markdown('<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)

# ---------- PREDICTION TREND ----------
hours_available = int(time_window.replace("h", ""))
trend_view = data["trend"].copy()
trend_view.loc[trend_view["Hour"] > hours_available, "Actual"] = None   # no measurements after the window

st.plotly_chart(prediction_chart(trend_view, data["threshold"]), width="stretch")

# ---------- ANOMALY ANALYSIS ----------
st.markdown('<div class="section-title">Anomaly Analysis</div>', unsafe_allow_html=True)
with st.expander("Anomaly details: Mahalanobis, LOF and scatter plot", expanded=detailed):

    status = data["anomaly_status"]
    status_color = {"NORMAL": GREEN, "ELEVATED": AMBER, "ANOMALY DETECTED": RED}[status]

    maha = data["mahalanobis"]
    maha_sub, maha_color = ("High", RED) if maha > 4 else ("Moderate", AMBER) if maha > 2.5 else ("Low", GREEN)

    lof = data["lof"]
    lof_sub, lof_color = ("Isolated point", RED) if lof > 1.5 else ("Normal density", GREEN)

    left, right = st.columns([3, 2])

    with left:
        st.plotly_chart(
            anomaly_scatter(data["population"], data["unit_xy"], anomaly, status_color),
            width="stretch",
        )

    with right:
        st.markdown(
            f'<div class="banner" style="--accent:{status_color}">'
            f'<div class="banner-title">{status}</div>'
            f'<div class="banner-text">{data["anomaly_note"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        mini_cards = [
            kpi_card("Mahalanobis Distance", maha, maha_sub, maha_color,
                    "Distance from the centre of normal data, accounting for feature correlations.", 0.0),
            kpi_card("LOF Score", lof, lof_sub, lof_color,
                    "Local Outlier Factor. Near 1 = similar density to neighbours; much higher = isolated.", 0.1),
            kpi_card("Anomaly Score", anomaly, anomaly_sub, anomaly_color,
                    "Combined anomaly score from Module A.", 0.2),
        ]
        st.markdown('<div class="kpi-grid">' + "".join(mini_cards) + "</div>", unsafe_allow_html=True)

# ---------- RISK ENGINE ----------
st.markdown('<div class="section-title">Risk Engine</div>', unsafe_allow_html=True)


def meter(name, pct, status, color, tip):
    return (
        f'<div class="meter" title="{tip}" style="--accent:{color}; --pct:{pct}%">'
        f'<div class="meter-top"><span class="meter-name">{name}</span>'
        f'<span class="meter-status" style="color:{color}">{status}</span></div>'
        f'<div class="meter-track"><div class="meter-fill"></div></div>'
        f'</div>'
    )


def level(pct):
    if pct >= 67:
        return "HIGH", RED
    if pct >= 34:
        return "MEDIUM", AMBER
    return "LOW", GREEN


anomaly_pct = min(100, round(anomaly / 4.2 * 100))
a_status, a_color = level(anomaly_pct)
d_status, d_color = level(data["drift_pct"])

slope_pct = data["slope_pct"]
if slope_pct < 50:
    s_status, s_color = "NORMAL", GREEN
elif slope_pct < 80:
    s_status, s_color = "WATCH", AMBER
else:
    s_status, s_color = "STEEP", RED

g_col, m_col = st.columns([2, 3])

with g_col:
    st.markdown(
        f'<div class="kpi-label" style="text-align:center">Overall Risk &nbsp;•&nbsp; '
        f'<span style="color:{RISK_COLORS[data["risk_label"]]}">{data["risk_label"]}</span></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(risk_gauge(data["risk_score"], RISK_COLORS[data["risk_label"]]),
                    width="stretch")

with m_col:
    st.markdown(
        meter("Anomaly Risk", anomaly_pct, a_status, a_color,
              "Risk from Module A anomaly detection (Mahalanobis + LOF).")
        + meter("Drift Risk", data["drift_pct"], d_status, d_color,
                "How far readings drift away from their starting values over time.")
        + meter("Safety Slope", slope_pct, s_status, s_color,
                "How much of the allowed degradation slope has been used. 100% means the limit is reached."),
        unsafe_allow_html=True,
    )

# ---------- QA DECISION ENGINE ----------
st.markdown('<div class="section-title">QA Decision Engine</div>', unsafe_allow_html=True)

decision = data["qa_decision"]
decision_color = DECISION_COLORS[decision]
ICONS = {"ok": ("✓", GREEN), "warn": ("⚠", AMBER), "fail": ("✕", RED)}

rule_rows = ""
for kind, text in data["qa_rules"]:
    icon, icon_color = ICONS[kind]
    rule_rows += f'<div class="rule"><span class="rule-icon" style="color:{icon_color}">{icon}</span>{text}</div>'

dec_col, rules_col = st.columns([2, 3])

with dec_col:
    st.markdown(
        f'<div class="decision" style="--accent:{decision_color}">'
        f'<div class="kpi-label">QA Decision</div>'
        f'<div class="decision-label">{decision}</div>'
        f'<div class="banner-text">{data["qa_reason"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with rules_col:
    st.markdown(
        f'<div class="rules-box"><div class="kpi-label">Rules that contributed</div>{rule_rows}</div>',
        unsafe_allow_html=True,
    )

# ---------- EXPLAINABILITY ----------
st.markdown('<div class="section-title">Explainability</div>', unsafe_allow_html=True)
demo_note()

with st.expander("Feature importance and LIME charts", expanded=detailed):
    ex_left, ex_right = st.columns(2)
    with ex_left:
        st.plotly_chart(importance_chart(data["importance"]), width="stretch")
    with ex_right:
        st.plotly_chart(lime_chart(data["contributions"]), width="stretch")

st.markdown(
    f'<div class="why"><div class="why-title">Why this prediction?</div>'
    f'<div class="why-text">{data["why_text"]}</div></div>',
    unsafe_allow_html=True,
)

# ---------- TRENDS & RECENT ANALYSIS ----------
st.markdown('<div class="section-title">Trends &amp; Recent Analysis</div>', unsafe_allow_html=True)

with st.expander("Trend charts and recent analysis table", expanded=detailed):
    history = data["history"]

    t1, t2, t3 = st.columns(3)
    with t1:
        st.plotly_chart(trend_line(history, "Risk", "Risk Trend (%)", AMBER), width="stretch")
    with t2:
        st.plotly_chart(trend_line(history, "Anomaly", "Anomaly Trend", RED), width="stretch")
    with t3:
        st.plotly_chart(trend_line(history, "Prediction", "Prediction Trend (168h)", CYAN,
                                   threshold=data["threshold"]), width="stretch")

    st.markdown('<div class="kpi-label" style="margin:0.6rem 0 0.4rem 0">Recent analysis (newest first)</div>',
                unsafe_allow_html=True)

    table = history.iloc[::-1].copy()
    table["Time"] = table["Time"].dt.strftime("%d %b, %H:%M")
    table["Risk"] = table["Risk"].astype(str) + "%"

    def color_decision(value):
        return f"color: {DECISION_COLORS[value]}; font-weight: 700"

    styled = (table.style
              .format({"Anomaly": "{:.2f}", "Prediction": "{:.1f}"})
              .map(color_decision, subset=["QA Decision"]))
    st.dataframe(styled, hide_index=True, width="stretch")

# ---------- EXPORT ----------
st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)

csv_bytes = data["history"].to_csv(index=False).encode("utf-8")
report_html = build_report(component, lot, time_window, data)

d1, d2 = st.columns(2)
with d1:
    st.download_button("Download CSV", csv_bytes,
                       file_name=f"SIH26170_{component}_{lot}_results.csv",
                       mime="text/csv", width="stretch")
with d2:
    st.download_button("Download HTML Report", report_html,
                       file_name=f"SIH26170_{component}_{lot}_report.html",
                       mime="text/html", width="stretch")