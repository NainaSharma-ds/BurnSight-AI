import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000"     # change this if the backend runs somewhere else


def get_api_data(component, lot, time_window):
    """Asks the FastAPI backend for results and returns them in the SAME shape
    as get_demo_data(), so the dashboard code does not need to change."""
    response = requests.get(
        f"{API_URL}/analyze",                      # endpoint name: agree this with the backend team
        params={"component": component, "lot": lot, "time_window": time_window},
        timeout=10,
    )
    response.raise_for_status()                    # raises an error if the backend replied with 4xx / 5xx
    return to_dashboard_format(response.json())


def to_dashboard_format(raw):
    """JSON only has lists and numbers. The charts need tables (DataFrames), so convert here.
    If the backend uses different key names, ONLY this function needs editing."""
    data = dict(raw)
    data["trend"] = pd.DataFrame(raw["trend"])                       # Hour, Actual, Predicted
    data["population"] = pd.DataFrame(raw["population"])             # x, y
    data["unit_xy"] = tuple(raw["unit_xy"])
    data["importance"] = pd.DataFrame(raw["importance"]).sort_values("Importance")
    contrib = pd.DataFrame(raw["contributions"])                     # Feature, Contribution
    data["contributions"] = contrib.reindex(
        contrib["Contribution"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    history = pd.DataFrame(raw["history"])                           # Time, Risk, Anomaly, Prediction, QA Decision
    history["Time"] = pd.to_datetime(history["Time"])
    data["history"] = history
    return data



def backend_online():
    """True if the backend answers at all (any HTTP reply counts). Used only for the status pill."""
    try:
        requests.get(API_URL, timeout=2)
        return True
    except requests.RequestException:
        return False