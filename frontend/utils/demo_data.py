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
    features, importance, contributions = make_explain(seed, anomaly_score, drift_pct, slope_pct)
    why_text = build_why(contributions)

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
        "features": features,
        "importance": importance,
        "contributions": contributions,
        "why_text": why_text,
    }


def build_qa(predicted, anomaly, drift_pct, slope_pct):
    """Checks each QA rule. Every rule is (status, text) where status is ok / warn / fail.
    Also builds a sentence naming the exact problems."""
    rules = []     # shown as the checklist in the UI
    issues = []    # short names of problems, used in the reason sentence

    if predicted >= THRESHOLD:
        rules.append(("ok", "Prediction within expected range"))
    else:
        rules.append(("fail", "Predicted 168h value is below the acceptance threshold"))
        issues.append("predicted 168h value below threshold")

    if anomaly <= 2:
        rules.append(("ok", "Anomaly score is normal"))
    elif anomaly <= 3:
        rules.append(("warn", "Elevated anomaly score"))
        issues.append("elevated anomaly score")
    else:
        rules.append(("fail", "High anomaly score: unit is an outlier"))
        issues.append("high anomaly score")

    if slope_pct < 50:
        rules.append(("ok", "Safety slope acceptable"))
    elif slope_pct < 80:
        rules.append(("warn", "Safety slope is approaching its limit"))
        issues.append("safety slope approaching its limit")
    else:
        rules.append(("fail", "Safety slope exceeds the allowed limit"))
        issues.append("safety slope above the limit")

    if drift_pct < 40:
        rules.append(("ok", "Drift over burn-in is low"))
    else:
        rules.append(("warn", "Noticeable drift over burn-in"))
        issues.append("noticeable drift over burn-in")

    fails = sum(1 for status, _ in rules if status == "fail")
    warns = sum(1 for status, _ in rules if status == "warn")
    problems = ", ".join(issues).capitalize()

    if fails >= 2 or rules[0][0] == "fail":
        decision = "REJECT"
        reason = f"{problems}. This unit should not be shipped."
        rules.append(("fail", "Unit should be rejected"))
    elif fails == 1:
        decision = "EXTEND"
        reason = (f"{problems}; predicted 168h value is still acceptable. "
                  "Extending burn-in would give the model more data to confirm the result.")
        rules.append(("warn", "Extended burn-in recommended"))
    elif warns >= 1:
        decision = "MONITOR"
        reason = f"{problems} detected; predicted 168h value remains within acceptable range."
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



FEATURES = ["Temperature", "Pressure", "Vibration", "Humidity", "Current", "Voltage"]


def make_explain(seed, anomaly_score, drift_pct, slope_pct):
    """FAKE explainability numbers. Real ones will come from the backend
    (permutation importance + LIME)."""
    erng = np.random.default_rng(seed + 7)

    base = erng.uniform(0.05, 0.35, size=len(FEATURES))
    base[0] += anomaly_score / 12            # temperature matters more when anomaly is high
    base[1] += drift_pct / 400               # pressure matters more when drift is high
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": np.round(base, 3)})
    importance = importance.sort_values("Importance")      # smallest first -> biggest on top

    signs = erng.choice([-1, 1], size=len(FEATURES), p=[0.35, 0.65])
    contrib = np.round(signs * base * erng.uniform(0.6, 1.1, size=len(FEATURES)), 2)
    contributions = pd.DataFrame({"Feature": FEATURES, "Contribution": contrib})
    contributions["abs"] = contributions["Contribution"].abs()
    contributions = contributions.sort_values("abs", ascending=False).drop(columns="abs")
    return FEATURES, importance, contributions.reset_index(drop=True)


def build_why(contributions):
    """Turns the contribution table into a plain-English sentence (with the numbers)."""
    ups = contributions[contributions["Contribution"] > 0].head(2)
    downs = contributions[contributions["Contribution"] < 0].head(2)

    def describe(rows):
        return " and ".join(f"{r.Feature.lower()} ({r.Contribution:+.2f})" for r in rows.itertuples())

    parts = []
    if len(ups):
        parts.append(f"Risk is pushed up by {describe(ups)}")
    if len(downs):
        parts.append(f"{describe(downs)} pull it down")
    if not parts:
        return "No feature has a meaningful influence on this prediction."
    return "; ".join(parts) + "."