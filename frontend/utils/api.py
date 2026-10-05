import re
import requests
import pandas as pd


# ============================================================
# FASTAPI BACKEND
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# COMPONENT NORMALIZATION
# ============================================================

def normalize_component(component):
    """
    Convert different component ID formats into the format
    expected by the FastAPI backend.

    The dataset and backend use IDs such as C0001. Accept common
    alternate formats but send the backend's canonical format.
    """

    if component is None:
        raise ValueError("Component cannot be None.")

    value = str(component).strip().upper()

    match = re.fullmatch(r"(?:COMP-?|C)?(\d+)", value)

    if match:
        return f"C{int(match.group(1)):04d}"

    return value


# ============================================================
# LOT NORMALIZATION
# ============================================================

def normalize_lot(lot):
    """
    Keep lot identifiers consistent.
    """

    if lot is None:
        return ""

    return str(lot).strip().upper()


# ============================================================
# TIME WINDOW NORMALIZATION
# ============================================================

def normalize_time_window(time_window):
    """
    Return one of the time-window strings accepted by the backend.
    """

    if time_window is None:
        return "168h"

    value = str(time_window).strip().lower()

    match = re.fullmatch(r"(\d+)\s*(?:h|hours?)?", value)

    if not match:
        raise ValueError(f"Invalid time window: {time_window!r}")

    hours = int(match.group(1))
    if hours not in {24, 96, 168}:
        raise ValueError(
            f"Unsupported time window {hours}h; use 24h, 96h, or 168h."
        )

    return f"{hours}h"


# ============================================================
# BACKEND HEALTH CHECK
# ============================================================

def backend_online():
    """
    Check whether FastAPI backend is running.
    """

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )

        if response.status_code != 200:
            return False

        payload = response.json()
        return (
            isinstance(payload, dict)
            and payload.get("status") == "healthy"
        )

    except (requests.RequestException, ValueError):
        return False


def get_components():
    """Get the component IDs available in the backend dataset."""
    try:
        response = requests.get(f"{API_URL}/components", timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not load components from the backend at {API_URL}: {error}"
        ) from error
    except ValueError as error:
        raise RuntimeError("Backend returned invalid component data.") from error

    components = result.get("components") if isinstance(result, dict) else None
    if not isinstance(components, list) or not all(
        isinstance(component, str) for component in components
    ):
        raise RuntimeError("Backend returned an invalid components list.")

    return components


def get_lots(component):
    """Get the dataset lots that contain the selected component."""
    try:
        response = requests.get(
            f"{API_URL}/lots",
            params={"component": normalize_component(component)},
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not load lots for {component} from the backend: {error}"
        ) from error
    except ValueError as error:
        raise RuntimeError("Backend returned invalid lot data.") from error

    lots = result.get("lots") if isinstance(result, dict) else None
    if not isinstance(lots, list) or not all(
        isinstance(lot, str) for lot in lots
    ):
        raise RuntimeError("Backend returned an invalid lots list.")

    return lots


# ============================================================
# SAFE DATAFRAME CONVERSION
# ============================================================

def to_dataframe(value):
    """
    Safely convert backend JSON data into a pandas DataFrame.

    Handles:
        list of dictionaries
        dictionary
        existing DataFrame
        None
    """

    if value is None:
        return pd.DataFrame()

    if isinstance(value, pd.DataFrame):
        return value

    if isinstance(value, list):

        if not value:
            return pd.DataFrame()

        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    if isinstance(value, dict):

        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


# ============================================================
# SAFE LIST CONVERSION
# ============================================================

def to_list(value):
    """
    Safely convert a backend value into a Python list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")

    if isinstance(value, dict):
        return [value]

    return [value]


# ============================================================
# GET API DATA
# ============================================================

def get_api_data(component, lot, time_window):
    """
    Get real prediction data from FastAPI backend.

    The frontend may use C0001/C0003 while the backend expects
    COMP-001/COMP-003, so the component is normalized here.
    """

    frontend_component = str(component).strip()

    backend_component = normalize_component(
        frontend_component
    )

    backend_lot = normalize_lot(lot)

    backend_time_window = normalize_time_window(
        time_window
    )

    params = {
        "component": backend_component,
        "lot": backend_lot,
        "time_window": backend_time_window,
    }

    try:

        response = requests.get(
            f"{API_URL}/analyze",
            params=params,
            timeout=30,
        )

        response.raise_for_status()

    except requests.HTTPError as error:

        # Give Streamlit a useful backend error.
        try:
            detail = response.json()

        except Exception:
            detail = response.text

        raise RuntimeError(
            f"Backend analysis failed.\n"
            f"Frontend component: {frontend_component}\n"
            f"Backend component: {backend_component}\n"
            f"Lot: {backend_lot}\n"
            f"Time window: {backend_time_window}h\n"
            f"Backend response: {detail}"
        ) from error

    except requests.RequestException as error:

        raise RuntimeError(
            f"Could not connect to FastAPI backend at "
            f"{API_URL}.\n"
            f"Error: {error}"
        ) from error

    try:

        result = response.json()

    except ValueError as error:

        raise RuntimeError(
            "FastAPI returned an invalid JSON response."
        ) from error

    return convert_api_data(result)


# ============================================================
# API RESPONSE CONVERSION
# ============================================================

def convert_api_data(result):
    """
    Convert FastAPI JSON response into the format expected
    by the Streamlit frontend.
    """

    if result is None:
        raise ValueError(
            "Backend returned an empty response."
        )

    if not isinstance(result, dict):
        raise ValueError(
            "Backend response must be a JSON object."
        )

    # --------------------------------------------------------
    # BASIC KPI DATA
    # --------------------------------------------------------

    data = {

        "risk_score": float(
            result.get("risk_score", 0)
        ),

        "risk_label": result.get(
            "risk_label",
            "Moderate",
        ),

        "predicted_168h": float(
            result.get("predicted_168h", 0)
        ),

        "threshold": float(
            result.get("threshold", 0)
        ),

        "anomaly_score": float(
            result.get("anomaly_score", 0)
        ),

        "anomaly_status": result.get(
            "anomaly_status",
            "NORMAL",
        ),

        "mahalanobis": float(
            result.get("mahalanobis", 0)
        ),

        "lof": float(
            result.get("lof", 0)
        ),

        "anomaly_note": result.get(
            "anomaly_note",
            "No anomaly information available.",
        ),

        "drift_pct": float(
            result.get("drift_pct", 0)
        ),

        "slope_pct": float(
            result.get("slope_pct", 0)
        ),

        "qa_decision": result.get(
            "qa_decision",
            "REVIEW",
        ),

        "qa_reason": result.get(
            "qa_reason",
            "",
        ),

        "why_text": result.get(
            "why_text",
            "",
        ),
    }

    # --------------------------------------------------------
    # ANOMALY DATA
    # --------------------------------------------------------

    data["population"] = to_dataframe(
        result.get("population", [])
    )

    data["unit_xy"] = result.get(
        "unit_xy",
        [0, 0],
    )

    # Make sure unit_xy is usable.
    if not isinstance(
        data["unit_xy"],
        (list, tuple),
    ):
        data["unit_xy"] = [0, 0]

    if len(data["unit_xy"]) < 2:
        data["unit_xy"] = [0, 0]

    # --------------------------------------------------------
    # QA RULES
    # --------------------------------------------------------

    raw_rules = to_list(result.get("qa_rules", []))
    data["qa_rules"] = []
    rule_statuses = {"PASS": "ok", "WARN": "warn", "FAIL": "fail"}
    for rule in raw_rules:
        if isinstance(rule, dict):
            status = rule_statuses.get(str(rule.get("status", "")).upper())
            message = rule.get("message")
            if status is None or not isinstance(message, str):
                raise ValueError("Backend returned an invalid QA rule.")
            data["qa_rules"].append((status, message))
        elif isinstance(rule, (list, tuple)) and len(rule) == 2:
            data["qa_rules"].append(tuple(rule))
        else:
            raise ValueError("Backend returned an invalid QA rules list.")

    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------

    data["importance"] = to_dataframe(
        result.get("importance", [])
    )

    data["contributions"] = to_dataframe(
        result.get("contributions", [])
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    data["history"] = to_dataframe(
        result.get("history", [])
    )
    if "Time" in data["history"]:
        data["history"]["Time"] = pd.to_datetime(data["history"]["Time"])

    # --------------------------------------------------------
    # PREDICTION TREND
    # --------------------------------------------------------

    data["trend"] = to_dataframe(
        result.get("trend", [])
    )

    return data