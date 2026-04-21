#
# Backend (FastAPI/Uvicorn) Dockerfile
#

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps commonly needed by opencv/scenedetect/ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY src /app/src

ENV PYTHONPATH=/app/src

EXPOSE 8001
CMD ["python", "-m", "uvicorn", "FastAPI_server:app", "--host", "0.0.0.0", "--port", "8001"]

