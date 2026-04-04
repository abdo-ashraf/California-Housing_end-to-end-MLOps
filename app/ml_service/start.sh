#!/bin/sh

WORKERS=$(python -c "import multiprocessing; print(multiprocessing.cpu_count())")

uvicorn app:app \
  --host 0.0.0.0 \
  --port 7860 \
  # --workers $WORKERS