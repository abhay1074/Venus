# Venus AI API (CPU). Mount the trained checkpoints at /app/backend/weights.
#   docker build -t venus-api -f backend/Dockerfile .
#   docker run -p 8000:8000 -v $PWD/backend/weights:/app/backend/weights venus-api
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY samples samples
ENV TF_CPP_MIN_LOG_LEVEL=3 PYTHONIOENCODING=utf-8 VENUS_SWEEP_ON_START=true
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
