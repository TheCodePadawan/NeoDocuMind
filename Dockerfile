# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Serve the RAG API. Build the index at runtime (mount data + storage volumes)
# or bake it in by running `python -m scripts.ingest_sample` during your build.
CMD ["uvicorn", "documind.api:app", "--host", "0.0.0.0", "--port", "8000"]
