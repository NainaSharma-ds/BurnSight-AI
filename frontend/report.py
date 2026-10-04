from datetime import datetime

CSS = """
body { font-family: Arial, sans-serif; background: #ffffff; color: #1a2233; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
h1 { margin-bottom: 0; } h2 { border-bottom: 2px solid #22d3ee; padding-bottom: 4px; margin-top: 2rem; }
.demo { background: #fef3c7; border: 1px solid #fbbf24; padding: 8px 12px; border-radius: 6px; }
.decision { font-size: 2rem; font-weight: bold; }
table { border-collapse: collapse; width: 100%; } td, th { border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }
"""


def build_report(component, lot, time_window, data):
    icons = {"ok": "&#10003;", "warn": "&#9888;", "fail": "&#10007;"}
    rules = "".join(f"<li>{icons[kind]} {text}</li>" for kind, text in data["qa_rules"])
    contrib = "".join(
        f"<tr><td>{r.Feature}</td><td>{r.Contribution:+.2f}</td></tr>"
        for r in data["contributions"].itertuples()
    )
    stamp = datetime.now().strftime("%d %b %Y, %H:%M")

    body = f"""
<h1>SIH26170 &ndash; Analysis Report</h1>
<p>AI Predictive Risk Monitoring &bull; Generated {stamp}</p>
<p class="demo"><b>DEMO MODE:</b> all numbers in this report are sample data, not real model output.</p>

<h2>Selection</h2>
<p>Component: <b>{component}</b> &nbsp; Lot: <b>{lot}</b> &nbsp; Time window: <b>{time_window}</b></p>

<h2>Key results</h2>
<table>
<tr><th>Overall risk</th><td>{data["risk_score"]}% ({data["risk_label"]})</td></tr>
<tr><th>Predicted 168h value</th><td>{data["predicted_168h"]} (threshold {data["threshold"]})</td></tr>
<tr><th>Anomaly score</th><td>{data["anomaly_score"]} &ndash; {data["anomaly_status"]}</td></tr>
<tr><th>Mahalanobis distance</th><td>{data["mahalanobis"]}</td></tr>
<tr><th>LOF score</th><td>{data["lof"]}</td></tr>
<tr><th>Drift / safety slope used</th><td>{data["drift_pct"]}% / {data["slope_pct"]}%</td></tr>
</table>

<h2>QA decision</h2>
<p class="decision">{data["qa_decision"]}</p>
<p>{data["qa_reason"]}</p>
<ul>{rules}</ul>

<h2>Why this prediction?</h2>
<p>{data["why_text"]}</p>
<table><tr><th>Feature</th><th>LIME contribution</th></tr>{contrib}</table>
"""
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>SIH26170 Report</title><style>{CSS}</style></head><body>{body}</body></html>"