import logging
import os
import threading
from typing import List
from enum import Enum
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=False)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================
# Configuration
# ==============================

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_PRODUCTION_ALIAS = os.getenv("MODEL_PRODUCTION_ALIAS")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

# FRONTEND_URL = os.getenv("FRONTEND_URL", "")

if not MODEL_NAME or not MODEL_PRODUCTION_ALIAS:
    raise RuntimeError("MODEL_NAME and MODEL_PRODUCTION_ALIAS must be set")

if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI must be set")

# ALLOWED_ORIGINS = [FRONTEND_URL] if FRONTEND_URL else ["*"]

# Configure MLflow authentication
if MLFLOW_TRACKING_USERNAME:
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME

if MLFLOW_TRACKING_PASSWORD:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logging.basicConfig(level=logging.INFO)

# ==============================
# Global Model State
# ==============================

model = None
model_loaded = False
model_version = None

model_lock = threading.Lock()

# ==============================
# Model Loading
# ==============================


def load_model():
    """
    Safely loads the MLflow champion model.
    Does NOT crash the app if loading fails.
    """
    global model, model_loaded, model_version

    with model_lock:
        try:
            loaded_model = mlflow.pyfunc.load_model(
                f"models:/{MODEL_NAME}@{MODEL_PRODUCTION_ALIAS}"
            )

            model = loaded_model
            model_loaded = True

            try:
                model_version = loaded_model.metadata.run_id
            except Exception:
                model_version = "unknown"

            logging.info("Model loaded successfully.")

        except Exception as e:
            model = None
            model_loaded = False
            model_version = None

            logging.error(f"Model loading failed: {e}")


# ==============================
# FastAPI Lifecycle
# ==============================


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logging.info("Shutting down API...")


app = FastAPI(
    title="Housing Price Prediction API",
    lifespan=lifespan
)

# ==============================
# Middleware
# ==============================

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=ALLOWED_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"{request.method} {request.url}")
    response = await call_next(request)
    return response


# ==============================
# Enums & Schemas
# ==============================

class OceanProximity(str, Enum):
    NEAR_BAY = "NEAR BAY"
    INLAND = "INLAND"
    NEAR_OCEAN = "NEAR OCEAN"
    ISLAND = "ISLAND"
    LESS_THAN_1H_OCEAN = "<1H OCEAN"


class HousingRecord(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: OceanProximity


class HousingBatchRequest(BaseModel):
    data: List[HousingRecord]


# ==============================
# Utilities
# ==============================


def ensure_model_loaded():
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check MLflow server or reload."
        )


def _service_metadata() -> dict:
        endpoint_rows = []
        for route in app.routes:
                if isinstance(route, APIRoute):
                        endpoint_rows.append(
                                {
                                        "path": route.path,
                                        "methods": sorted(
                                                method for method in route.methods if method not in {"HEAD", "OPTIONS"}
                                        ),
                                }
                        )

        return {
                "service": "Housing Price Prediction API",
                "status": "ok",
                "model_loaded": model_loaded,
                "model_name": MODEL_NAME,
                "model_version": model_version,
                "docs_url": "/docs",
                "redoc_url": "/redoc",
                "endpoints": [row["path"] for row in endpoint_rows],
                "endpoint_rows": endpoint_rows,
        }


def _render_homepage_html(meta: dict) -> str:
        status_class = "up" if meta["model_loaded"] else "down"
        status_label = "Loaded" if meta["model_loaded"] else "Not Loaded"

        endpoint_items = "\n".join(
                (
                        f"<li><span class='method'>{'/'.join(row['methods'])}</span>"
                        f"<a href='{row['path']}'>{row['path']}</a></li>"
                )
                for row in meta["endpoint_rows"]
        )

        return f"""<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>{meta['service']}</title>
    <style>
        :root {{
            --bg1: #f4f6fb;
            --bg2: #e7ecf7;
            --ink: #1f2a37;
            --muted: #5b6575;
            --card: #ffffff;
            --line: #d7deea;
            --accent: #0f766e;
            --up: #1d7a47;
            --down: #b42318;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: "Segoe UI", "Noto Sans", sans-serif;
            color: var(--ink);
            background: radial-gradient(circle at 10% 10%, var(--bg2), var(--bg1) 40%);
            min-height: 100vh;
        }}
        .wrap {{ max-width: 960px; margin: 0 auto; padding: 28px 20px 40px; }}
        .hero {{
            background: linear-gradient(120deg, #133b5c 0%, #1f6f8b 60%, #35a29f 100%);
            color: #fff;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 14px 30px rgba(18, 38, 63, 0.2);
        }}
        h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
        .sub {{ margin: 0; opacity: 0.95; }}
        .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 16px; }}
        .card {{
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 4px 12px rgba(32, 45, 65, 0.06);
        }}
        .label {{ font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
        .value {{ margin-top: 6px; font-size: 1.05rem; font-weight: 600; word-break: break-word; }}
        .status.up {{ color: var(--up); }}
        .status.down {{ color: var(--down); }}
        .links {{ margin-top: 16px; display: flex; gap: 12px; flex-wrap: wrap; }}
        .btn {{
            text-decoration: none;
            color: #083b5c;
            background: #d9f3ff;
            border: 1px solid #b9e6fb;
            border-radius: 999px;
            padding: 8px 14px;
            font-weight: 600;
        }}
        .panel {{ margin-top: 18px; }}
        ul {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }}
        li {{
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 10px 12px;
            display: flex;
            gap: 10px;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
        }}
        .method {{
            font-size: 0.72rem;
            color: #0b5cab;
            border: 1px solid #bcd8f5;
            background: #eaf4ff;
            border-radius: 6px;
            padding: 2px 8px;
            font-weight: 700;
        }}
        a {{ color: #0d4a85; text-decoration: none; font-weight: 600; }}
        @media (max-width: 680px) {{
            .hero {{ padding: 18px; }}
            h1 {{ font-size: 1.35rem; }}
        }}
    </style>
</head>
<body>
    <main class='wrap'>
        <section class='hero'>
            <h1>{meta['service']}</h1>
            <p class='sub'>FastAPI inference service for California housing predictions.</p>
            <div class='grid'>
                <div class='card'>
                    <div class='label'>Service Status</div>
                    <div class='value'>{meta['status'].upper()}</div>
                </div>
                <div class='card'>
                    <div class='label'>Model</div>
                    <div class='value'>{meta['model_name']}</div>
                </div>
                <div class='card'>
                    <div class='label'>Model Loaded</div>
                    <div class='value status {status_class}'>{status_label}</div>
                </div>
                <div class='card'>
                    <div class='label'>Model Version</div>
                    <div class='value'>{meta['model_version'] or 'N/A'}</div>
                </div>
            </div>
            <div class='links'>
                <a class='btn' href='{meta['docs_url']}'>Open Swagger</a>
                <a class='btn' href='{meta['redoc_url']}'>Open ReDoc</a>
            </div>
        </section>
        <section class='panel'>
            <h2>Available Endpoints</h2>
            <ul>
                {endpoint_items}
            </ul>
        </section>
    </main>
</body>
</html>
"""


# ==============================
# Endpoints
# ==============================

@app.get("/")
def root(request: Request):
    """
    Root endpoint for API discovery.
    Returns HTML for browser requests and JSON for API clients.
    """
    meta = _service_metadata()
    accept_header = request.headers.get("accept", "")

    if "text/html" in accept_header:
        return HTMLResponse(content=_render_homepage_html(meta))

    return {
        "service": meta["service"],
        "status": meta["status"],
        "model_loaded": meta["model_loaded"],
        "model_name": meta["model_name"],
        "model_version": meta["model_version"],
        "docs_url": meta["docs_url"],
        "redoc_url": meta["redoc_url"],
        "endpoints": meta["endpoints"],
    }

@app.get("/health")
def health():
    """
    Health check endpoint.
    Useful for AWS load balancers and monitoring.
    """
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_name": MODEL_NAME,
        "model_version": model_version
    }


@app.get("/model_info")
def model_info():
    """
    Returns information about the currently loaded model.
    """
    return {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_PRODUCTION_ALIAS,
        "model_loaded": model_loaded,
        "model_version": model_version
    }


@app.post("/predict")
def predict_single(record: HousingRecord):
    """
    Single prediction endpoint.
    """
    ensure_model_loaded()

    df = pd.DataFrame([record.model_dump()])

    try:
        prediction = model.predict(df)  # type: ignore
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    return {
        "prediction": float(prediction[0]),
        "model_version": model_version
    }


@app.post("/predict_batch")
def predict_batch(request: HousingBatchRequest):
    """
    Batch prediction endpoint.
    """
    ensure_model_loaded()

    df = pd.DataFrame(
        [record.model_dump() for record in request.data]
    )

    try:
        predictions = model.predict(df)  # type: ignore
    except Exception as e:
        logging.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

    return {
        "predictions": predictions.tolist(),
        "model_version": model_version
    }


@app.post("/reload_model")
def reload_model():
    """
    Reloads the champion model from MLflow.
    """
    load_model()

    if not model_loaded:
        raise HTTPException(
            status_code=500,
            detail="Model reload failed."
        )

    return {
        "message": "Model reloaded successfully.",
        "model_version": model_version
    }