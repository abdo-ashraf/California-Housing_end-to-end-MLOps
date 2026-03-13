import logging
import os
import threading
from typing import List
from enum import Enum
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
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


# ==============================
# Endpoints
# ==============================

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