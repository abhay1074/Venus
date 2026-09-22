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
# 0.0.0.0 is required inside a container (a loopback bind is unreachable from
# the host); backend/serve.py refuses it unless this is set, so this line is
# the explicit opt-in. Publish the port only to a trusted network: the API
# has no authentication (docs/PRIVACY.md).
ENV TF_CPP_MIN_LOG_LEVEL=3 PYTHONIOENCODING=utf-8 VENUS_SWEEP_ON_START=true VENUS_BIND_ALL=true
EXPOSE 8000
CMD ["python", "-m", "backend.serve", "--port", "8000"]
