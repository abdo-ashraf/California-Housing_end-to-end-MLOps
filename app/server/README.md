# Frontend (Streamlit)

Modern Streamlit frontend for the Housing Price Predictor backend.

## Setup

```bash
cd app/server
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Frontend runs on `http://localhost:8501`

## Environment Variables

Set the backend API URL (optional). `BACKEND_URL` is checked first, then `API_BASE_URL`:

```bash
BACKEND_URL=http://localhost:8000 streamlit run app.py
```

Defaults to `http://localhost:8000` if not set.

## Features

- ✨ Single housing record prediction
- 📊 Batch predictions with JSON input
- 🔄 Model reload capability
- 📈 Summary statistics for batch results
- 🧾 Model metadata display (name, alias, version)
- 🎨 Clean, modern Streamlit UI
- ⚡ Real-time API status checks
- 🧠 Session state management

## API Routes Used

- `GET /health` - Health check and model state
- `GET /model_info` - Loaded model metadata
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions (`{"data": [HousingRecord, ...]}`)
- `POST /reload_model` - Reload MLflow model
