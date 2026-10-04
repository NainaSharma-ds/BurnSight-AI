import math
import random

import numpy as np
import pandas as pd


def get_demo_data(component, lot, time_window):
    """Returns FAKE sample results so the dashboard works without a backend.
    Same component + lot always gives the same numbers (so the demo is stable)."""

    seed = sum(ord(c) for c in component + lot)
    rng = random.Random(seed)

    risk_score = rng.randint(20, 95)                 # percent
    predicted_168h = round(rng.uniform(70, 95), 1)
    anomaly_score = round(rng.uniform(0.8, 4.2), 2)

    # Turn the risk number into a label
    if risk_score < 35:
        risk_label = "Low"
    elif risk_score < 60:
        risk_label = "Moderate"
    elif risk_score < 80:
        risk_label = "Elevated"
    else:
        risk_label = "Critical"

    # Turn the risk number into a QA decision
    if risk_score < 35:
        qa_decision = "PASS"
    elif risk_score < 80:
        qa_decision = "MONITOR"
    elif risk_score < 90:
        qa_decision = "EXTEND"
    else:
        qa_decision = "REJECT"

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

    drift_pct = rng.randint(15, 85)    # how far readings drift over time (fake)
    slope_pct = rng.randint(10, 90)    # how much of the allowed degradation slope is used (fake)
    population, unit_xy = make_population(seed, anomaly_score)

    return {
        "risk_score": risk_score,
        "risk_label": risk_label,
        "predicted_168h": predicted_168h,
        "anomaly_score": anomaly_score,
        "qa_decision": qa_decision,
        "trend": trend,
        "threshold": 75,
        "mahalanobis": mahalanobis,
        "lof": lof,
        "anomaly_status": anomaly_status,
        "anomaly_note": anomaly_note,
        "population": population,
        "unit_xy": unit_xy,
        "drift_pct": drift_pct,
        "slope_pct": slope_pct,
    }


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