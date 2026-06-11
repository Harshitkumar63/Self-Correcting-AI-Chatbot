"""
FastAPI Main Application — Self-Improving LLM Pipeline

Entry point that wires together CORS, database lifecycle,
and all API routers.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path for ml_pipeline imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.database import init_db, close_db
from app.routers import chat, logs, tuning

# ── Logging ─────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    logger.info("🚀 Starting Self-Improving LLM Pipeline backend...")
    await init_db()
    logger.info("✅ Database initialized.")
    yield
    await close_db()
    logger.info("🛑 Backend shutdown complete.")


# ── App ─────────────────────────────────────────────────────

app = FastAPI(
    title="Self-Improving LLM Pipeline",
    description=(
        "Automated feedback loop system where LLM outputs are evaluated "
        "using semantic similarity and low-quality responses are collected "
        "for iterative LoRA fine-tuning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────

app.include_router(chat.router)
app.include_router(logs.router)
app.include_router(tuning.router)


# ── Health Check ────────────────────────────────────────────

@app.get("/", tags=["health"])
async def health_check():
    """Root endpoint — API health check."""
    return {
        "status": "healthy",
        "service": "Self-Improving LLM Pipeline",
        "version": "1.0.0",
    }


@app.get("/api/health", tags=["health"])
async def api_health():
    """Detailed API health check."""
    from app.services.ml_service import get_ml_service

    ml = get_ml_service()
    feedback = ml.get_feedback_status()
    training = ml.get_training_status()

    return {
        "status": "healthy",
        "pipeline": {
            "feedback_queue": feedback,
            "training": training,
        },
    }
