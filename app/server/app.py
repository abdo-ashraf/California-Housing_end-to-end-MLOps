import json
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=False)

# Configuration
DEFAULT_BACKEND_URL = "http://localhost:8000"
BACKEND_URL = os.getenv("BACKEND_URL") or os.getenv("API_BASE_URL") or DEFAULT_BACKEND_URL
BACKEND_URL = BACKEND_URL.rstrip("/")

HEALTH_ENDPOINT = "/health"
MODEL_INFO_ENDPOINT = "/model_info"
RELOAD_ENDPOINT = "/reload_model"
PREDICT_ENDPOINT = "/predict"
BATCH_PREDICT_ENDPOINT = "/predict_batch"

FEATURE_FIELDS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
]

# Page configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown("""
<style>
    /* Improve spacing */
    .main {
        padding-top: 2rem;
    }

    /* Make metric cards theme-aware */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border-radius: 0.5rem;
        padding: 1rem;
    }

    /* Success box */
    .success-box {
        background-color: rgba(40, 167, 69, 0.15);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(40, 167, 69, 0.4);
        color: inherit;
    }

    /* Error box */
    .error-box {
        background-color: rgba(220, 53, 69, 0.15);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(220, 53, 69, 0.4);
        color: inherit;
    }

    /* Improve header contrast */
    h1, h2, h3 {
        color: inherit;
    }

</style>
""", unsafe_allow_html=True)


# API Helper Functions
def extract_error_message(response: requests.Response) -> str:
    """Safely extract a readable error message from API responses."""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return str(payload.get("detail") or payload.get("message") or payload)
        return str(payload)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def make_api_call(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Any]:
    """Make API call and return (success, parsed_response_or_error)."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        headers = {"Content-Type": "application/json"}

        normalized_method = method.strip().upper()
        if normalized_method not in {"GET", "POST"}:
            return False, f"Invalid method: {method}"

        request_kwargs: Dict[str, Any] = {
            "headers": headers,
            "timeout": 10,
            # Preserve HTTP method when handling redirects ourselves.
            "allow_redirects": False,
        }
        if normalized_method != "GET":
            request_kwargs["json"] = data

        response = requests.request(normalized_method, url, **request_kwargs)

        if response.status_code in {301, 302, 303, 307, 308}:
            redirect_target = response.headers.get("Location")
            if not redirect_target:
                return False, "Received redirect without Location header"

            redirect_url = urljoin(url, redirect_target)
            response = requests.request(normalized_method, redirect_url, **request_kwargs)

        if response.status_code in [200, 201]:
            return True, response.json()
        return False, extract_error_message(response)
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to backend at {BACKEND_URL}"
    except Exception as e:
        return False, str(e)


def refresh_api_state(show_messages: bool = True):
    """Refresh API and model status from FastAPI health/model-info endpoints."""
    success, response = make_api_call(HEALTH_ENDPOINT)

    if success:
        st.session_state.api_status = response.get("status", "unknown")
        st.session_state.model_loaded = response.get("model_loaded", False)
        st.session_state.model_name = response.get("model_name", "unknown")
        st.session_state.model_version = response.get("model_version")

        info_success, info_response = make_api_call(MODEL_INFO_ENDPOINT)
        if info_success:
            st.session_state.model_alias = info_response.get("model_alias", "unknown")
            st.session_state.model_name = info_response.get(
                "model_name",
                st.session_state.model_name,
            )
            st.session_state.model_version = info_response.get(
                "model_version",
                st.session_state.model_version,
            )

        if show_messages:
            st.success("API is reachable.")
            st.info(f"Status: {st.session_state.api_status}")
            st.info(f"Model Loaded: {st.session_state.model_loaded}")
    else:
        st.session_state.api_status = "error"
        st.session_state.model_loaded = False
        st.session_state.model_name = "unknown"
        st.session_state.model_alias = "unknown"
        st.session_state.model_version = None
        if show_messages:
            st.error(f"Error: {response}")


def validate_and_prepare_batch_records(records: Any) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Validate user batch JSON and keep only fields required by FastAPI schema."""
    if not isinstance(records, list):
        return False, "Input must be a JSON array", []

    prepared_records: List[Dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            return False, f"Record #{index} must be a JSON object", []

        missing_fields = [field for field in FEATURE_FIELDS if field not in record]
        if missing_fields:
            return (
                False,
                f"Record #{index} is missing fields: {', '.join(missing_fields)}",
                [],
            )

        prepared_records.append({field: record[field] for field in FEATURE_FIELDS})

    return True, "", prepared_records


# Session state initialization
if "health_checked" not in st.session_state:
    refresh_api_state(show_messages=False)
    st.session_state.health_checked = True

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"
if "model_name" not in st.session_state:
    st.session_state.model_name = "unknown"
if "model_alias" not in st.session_state:
    st.session_state.model_alias = "unknown"
if "model_version" not in st.session_state:
    st.session_state.model_version = None

# Header
st.title("🏠 Housing Price Predictor")
st.markdown("ML-powered housing price predictions using FastAPI + Streamlit")


# Sidebar: API Status & Controls
with st.sidebar:
    st.header("API Controls")

    # Health Check
    if st.button("🔍 Check API Health", use_container_width=True):
        refresh_api_state(show_messages=True)

    # Reload Model
    if st.button("🔄 Reload Model", use_container_width=True):
        success, response = make_api_call(RELOAD_ENDPOINT, method="POST")
        if success:
            st.success(f"{response.get('message')}")
            refresh_api_state(show_messages=False)
        else:
            st.session_state.model_loaded = False
            st.error(f"Error: {response}")

    st.divider()

    # Status Display
    st.metric("API Status", str(st.session_state.api_status).upper())
    st.metric("Model Loaded", "Yes" if st.session_state.model_loaded else "No")
    st.caption(f"Model: {st.session_state.model_name}")
    st.caption(f"Alias: {st.session_state.model_alias}")
    st.caption(f"Version: {st.session_state.model_version or 'unknown'}")

    st.divider()
    st.caption(f"Backend: {BACKEND_URL}")

# Main Content
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# Tab 1: Single Prediction
with tab1:
    st.header("Single Housing Record")

    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input("Longitude", value=-122.23, format="%.2f")
        latitude = st.number_input("Latitude", value=37.88, format="%.2f")
        housing_median_age = st.number_input("Housing Median Age", value=41.0, min_value=0.0, format="%.2f")
        total_rooms = st.number_input("Total Rooms", value=880.0, min_value=0.0, format="%.2f")
        total_bedrooms = st.number_input("Total Bedrooms", value=129.0, min_value=0.0, format="%.2f")
    
    with col2:
        population = st.number_input("Population", value=322.0, min_value=0.0, format="%.2f")
        households = st.number_input("Households", value=126.0, min_value=0.0, format="%.2f")
        median_income = st.number_input("Median Income", value=8.3252, format="%.2f")
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"],
            index=0
        )
    
    # Prediction Request
    if st.button("Predict Price", key="single_predict", use_container_width=True):
        if not st.session_state.model_loaded:
            st.warning("Model is not loaded. Please reload the model first.")
        else:
            payload = {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }

            with st.spinner("🔮 Making prediction..."):
                success, response = make_api_call(PREDICT_ENDPOINT, method="POST", data=payload)

            if success:
                prediction = response.get("prediction", 0)
                predicted_model_version = response.get("model_version")
                if predicted_model_version:
                    st.session_state.model_version = predicted_model_version

                st.markdown(f"""
                <div class="success-box">
                    <h3>Predicted House Price</h3>
                    <h2>${prediction:,.2f}</h2>
                    <p>Model Version: {st.session_state.model_version or 'unknown'}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                    <strong>Error:</strong> {response}
                </div>
                """, unsafe_allow_html=True)

# Tab 2: Batch Prediction
with tab2:
    st.header("Batch Predictions")

    batch_json = st.text_area(
        "Paste JSON array of housing records",
        value=json.dumps([
            {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY",
            },
            {
                "longitude": -122.22,
                "latitude": 37.86,
                "housing_median_age": 21.0,
                "total_rooms": 7099.0,
                "total_bedrooms": 1106.0,
                "population": 2401.0,
                "households": 1138.0,
                "median_income": 8.3014,
                "ocean_proximity": "NEAR BAY",
            },
        ], indent=2),
        height=250,
        help="Each object must have all required fields"
    )

    if st.button("Predict Batch", key="batch_predict", use_container_width=True):
        if not st.session_state.model_loaded:
            st.warning("Model is not loaded. Please reload the model first.")
        else:
            try:
                batch_data = json.loads(batch_json)

                is_valid, validation_error, prepared_records = validate_and_prepare_batch_records(batch_data)
                if not is_valid:
                    st.error(validation_error)
                else:
                    payload = {"data": prepared_records}

                    with st.spinner(f"🔮 Predicting {len(prepared_records)} records..."):
                        success, response = make_api_call(
                            BATCH_PREDICT_ENDPOINT,
                            method="POST",
                            data=payload,
                        )

                    if success:
                        predictions = response.get("predictions", [])
                        predicted_model_version = response.get("model_version")
                        if predicted_model_version:
                            st.session_state.model_version = predicted_model_version

                        if not predictions:
                            st.warning("No predictions were returned by the backend.")
                        else:
                            # Display results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                <div class="success-box">
                                    <strong>Records Processed:</strong> {len(predictions)}<br>
                                    <strong>Average Price:</strong> ${sum(predictions)/len(predictions):,.2f}<br>
                                    <strong>Min Price:</strong> ${min(predictions):,.2f}<br>
                                    <strong>Max Price:</strong> ${max(predictions):,.2f}<br>
                                    <strong>Model Version:</strong> {st.session_state.model_version or 'unknown'}
                                </div>
                                """, unsafe_allow_html=True)

                            with col2:
                                st.markdown(f"""
                                <div class="success-box">
                                    <strong>Prices</strong><br>
                                    {chr(10).join([f"${p:,.2f}" for p in predictions])}
                                </div>
                                """, unsafe_allow_html=True)

                            # Detailed results table
                            st.subheader("Detailed Results")
                            st.dataframe(
                                {
                                    "Record #": range(1, len(predictions) + 1),
                                    "Predicted Price": [f"${p:,.2f}" for p in predictions],
                                },
                                use_container_width=True,
                            )
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                            <strong>Error:</strong> {response}
                        </div>
                        """, unsafe_allow_html=True)
            
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your input.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.caption("🏠 Housing Price Predictor | Powered by FastAPI + Streamlit")
