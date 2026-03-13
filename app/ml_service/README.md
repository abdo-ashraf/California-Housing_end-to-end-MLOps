## docker build -t housing-ml-service .

## docker run --rm -p 8000:8000 --env-file .env housing-ml-service

## docker run --env=MODEL_NAME=HousingModel --env=MODEL_PRODUCTION_ALIAS=champion --env=MLFLOW_TRACKING_URI=https://dagshub.com/abdoashraff185/California-Housing_end-to-end-MLOps.mlflow --env=MLFLOW_TRACKING_USERNAME=abdoashraff185 --env=MLFLOW_TRACKING_PASSWORD=739f3151a65dacd613242e9409cba57ad66cd0c7 --env=FRONTEND_URL=http://localhost:8501 -p 8000:8000 -d housing-ml-service:latest