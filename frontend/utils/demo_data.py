import math
import random

import numpy as np
import pandas as pd

THRESHOLD = 75   # acceptance threshold for the predicted 168h value


def get_demo_data(component, lot, time_window):
    """Returns FAKE sample results so the dashboard works without a backend.
    Same component + lot always gives the same numbers (so the demo is stable)."""

    seed = sum(ord(c) for c in component + lot)
    rng = random.Random(seed)

    predicted_168h = round(rng.uniform(70, 95), 1)
    anomaly_score = round(rng.uniform(0.8, 4.2), 2)
    drift_pct = rng.randint(15, 85)    # how far readings drift over time (fake)
    slope_pct = rng.randint(10, 90)    # how much of the allowed degradation slope is used (fake)

    # Overall risk = weighted mix of the three risk signals
    anomaly_pct = min(100, round(anomaly_score / 4.2 * 100))
    risk_score = round(0.4 * anomaly_pct + 0.3 * drift_pct + 0.3 * slope_pct)

    if risk_score < 35:
        risk_label = "Low"
    elif risk_score < 60:
        risk_label = "Moderate"
    elif risk_score < 80:
        risk_label = "Elevated"
    else:
        risk_label = "Critical"

    # QA decision comes from the rules (so decision and explanation always agree)
    qa_decision, qa_reason, qa_rules = build_qa(predicted_168h, anomaly_score, drift_pct, slope_pct)

    trend = make_trend(rng, predicted_168h)

    # --- Module A style numbers (fake) ---
    mahalanobis = round(anomaly_score * rng.uniform(1.2, 1.8), 2)
    lof = round(1 + anomaly_score * rng.uniform(0.15, 0.35), 2)

    if anomaly_score > 3:
        anomaly_status = "ANOMALY DETECTED"
        anomaly_note = ("This unit sits far outside the cluster of normal burn-in behaviour "
                        "(beyond the dotted boundary). Both Mahalanobis distance and LOF flag it.")
    elif anomaly_score > 2:
        anomaly_status = "ELEVATED"
        anomaly_note = ("This unit is drifting away from the normal cluster but is still near "
                        "the boundary. Keep an eye on it.")
    else:
        anomaly_status = "NORMAL"
        anomaly_note = "This unit behaves like the normal population. No unusual deviation detected."

    population, unit_xy = make_population(seed, anomaly_score)

    return {
        "risk_score": risk_score,
        "risk_label": risk_label,
        "predicted_168h": predicted_168h,
        "anomaly_score": anomaly_score,
        "qa_decision": qa_decision,
        "qa_reason": qa_reason,
        "qa_rules": qa_rules,
        "trend": trend,
        "threshold": THRESHOLD,
        "mahalanobis": mahalanobis,
        "lof": lof,
        "anomaly_status": anomaly_status,
        "anomaly_note": anomaly_note,
        "population": population,
        "unit_xy": unit_xy,
        "drift_pct": drift_pct,
        "slope_pct": slope_pct,
    }


def build_qa(predicted, anomaly, drift_pct, slope_pct):
    """Checks each QA rule. Every rule is (status, text) where status is ok / warn / fail."""
    rules = []

    if predicted >= THRESHOLD:
        rules.append(("ok", "Prediction within expected range"))
    else:
        rules.append(("fail", "Predicted 168h value is below the acceptance threshold"))

    if anomaly <= 2:
        rules.append(("ok", "Anomaly score is normal"))
    elif anomaly <= 3:
        rules.append(("warn", "Elevated anomaly score"))
    else:
        rules.append(("fail", "High anomaly score: unit is an outlier"))

    if slope_pct < 50:
        rules.append(("ok", "Safety slope acceptable"))
    elif slope_pct < 80:
        rules.append(("warn", "Safety slope is approaching its limit"))
    else:
        rules.append(("fail", "Safety slope exceeds the allowed limit"))

    if drift_pct < 40:
        rules.append(("ok", "Drift over burn-in is low"))
    else:
        rules.append(("warn", "Noticeable drift over burn-in"))

    fails = sum(1 for status, _ in rules if status == "fail")
    warns = sum(1 for status, _ in rules if status == "warn")

    if fails >= 2 or rules[0][0] == "fail":
        decision = "REJECT"
        reason = ("Predicted 168h performance or multiple safety checks fall outside "
                  "acceptable limits. This unit should not be shipped.")
        rules.append(("fail", "Unit should be rejected"))
    elif fails == 1:
        decision = "EXTEND"
        reason = ("One safety check failed while the predicted 168h value is still acceptable. "
                  "Extending burn-in would give the model more data to confirm the result.")
        rules.append(("warn", "Extended burn-in recommended"))
    elif warns >= 1:
        decision = "MONITOR"
        reason = (f"{warns} warning signal(s) detected while predicted 168h performance "
                  "remains within the acceptable operating range.")
        rules.append(("warn", "Requires monitoring"))
    else:
        decision = "PASS"
        reason = ("All checks passed. Predicted 168h performance is within range and no "
                  "significant anomaly or drift was found.")
        rules.append(("ok", "No further action required"))

    return decision, reason, rules


def make_trend(rng, final_value):
    """Burn-in values at each checkpoint: 'actual' (measured) and 'predicted' (model)."""
    hours = [0, 24, 48, 96, 168]
    start = final_value - rng.uniform(8, 18)            # value at 0h is lower
    steps = [0, 0.35, 0.6, 0.85, 1.0]                   # how far along the climb we are
    predicted = [round(start + (final_value - start) * s, 1) for s in steps]
    actual = [round(p + rng.uniform(-1.5, 1.5), 1) for p in predicted]
    actual[-1] = None                                   # 168h hasn't happened yet
    return pd.DataFrame({"Hour": hours, "Actual": actual, "Predicted": predicted})


def make_population(seed, anomaly_score):
    """150 'normal' units around (0, 0) plus the selected unit, placed
    'anomaly_score' units away from the centre."""
    nrng = np.random.default_rng(seed)
    points = nrng.normal(0, 1, size=(150, 2))
    population = pd.DataFrame({"x": points[:, 0], "y": points[:, 1]})
    angle = nrng.uniform(0, 2 * math.pi)
    unit_xy = (anomaly_score * math.cos(angle), anomaly_score * math.sin(angle))
    return population, unit_xy