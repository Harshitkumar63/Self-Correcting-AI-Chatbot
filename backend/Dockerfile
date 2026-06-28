# ── Backend Dockerfile ─────────────────────────────────────
# Python 3.11 base, installs requirements, copies app code.
# Exposes port 8000 for FastAPI/Uvicorn.

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for torch + sentencepiece
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────
COPY . .

# ── Copy ML pipeline (mounted at project root level) ──────
# In docker-compose, ml_pipeline is mounted as a volume.
# For standalone builds, copy it manually if needed.

EXPOSE 8000

# ── Health check ───────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# ── Run ────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
