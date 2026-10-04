from datetime import datetime
import streamlit as st
from utils.demo_data import get_demo_data
from utils.charts import prediction_chart, anomaly_scatter

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
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- HEADER ----------
now = datetime.now().strftime("%d %b %Y, %H:%M:%S")

st.markdown(f"""
<div class="header">
<div>
<div class="project-id">SIH26170</div>
<div class="title">AI Predictive Risk Monitoring</div>
<div class="tagline">Burn-in intelligence • Anomaly detection • 168h prediction • QA decision support</div>
</div>
<div class="pills">
<span class="pill pill-demo">DEMO MODE</span>
<span class="pill pill-ok"><span class="dot"></span>SYSTEM ONLINE</span>
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
    st.markdown(
        """
- **Mode:** Demo data
- **Backend:** Not connected
- **Module A:** Mahalanobis + LOF
- **Module B:** XGBoost
"""
    )

# ---------- LOAD DATA (demo for now) ----------
data = get_demo_data(component, lot, time_window)

# ---------- KPI CARDS ----------
GREEN, AMBER, ORANGE, RED, CYAN = "#34d399", "#fbbf24", "#fb923c", "#f87171", "#22d3ee"

RISK_COLORS = {"Low": GREEN, "Moderate": CYAN, "Elevated": AMBER, "Critical": RED}
DECISION_COLORS = {"PASS": GREEN, "MONITOR": AMBER, "EXTEND": ORANGE, "REJECT": RED}


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
if predicted >= 75:
    pred_sub, pred_color = "Within acceptable range", GREEN
else:
    pred_sub, pred_color = "Below acceptable range", RED

cards = [
    kpi_card("Overall Risk Score", f'{data["risk_score"]}%', data["risk_label"],
             RISK_COLORS[data["risk_label"]],
             "Combined risk from anomaly, drift and prediction signals.", 0.0),
    kpi_card("Predicted 168h Value", predicted, pred_sub, pred_color,
             "Model forecast of the performance value after 168 burn-in hours.", 0.1),
    kpi_card("Anomaly Score", anomaly, anomaly_sub, anomaly_color,
             "How unusual this unit is compared with normal units. Higher means more unusual.", 0.2),
    kpi_card("QA Decision", data["qa_decision"], "Recommended action",
             DECISION_COLORS[data["qa_decision"]],
             "PASS / MONITOR / EXTEND / REJECT, decided by the QA rule engine.", 0.3),
]

st.markdown('<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)

# ---------- PREDICTION TREND ----------
st.plotly_chart(prediction_chart(data["trend"], data["threshold"]), use_container_width=True)

# ---------- ANOMALY ANALYSIS ----------
st.markdown('<div class="section-title">Anomaly Analysis</div>', unsafe_allow_html=True)

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
        use_container_width=True,
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