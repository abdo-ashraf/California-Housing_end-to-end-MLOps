
# Base Image
FROM python:3.11-slim

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent python buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working Directory
WORKDIR /app

# Install Python Dependencies
# Copy requirements first (Docker layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy Application Code
COPY app.py start.sh ./

# Make script executable
RUN chmod +x start.sh

# Expose Port
EXPOSE 7860

# Run FastAPI
CMD ["./start.sh"]
