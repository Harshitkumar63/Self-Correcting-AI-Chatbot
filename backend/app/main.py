"""
FastAPI Main Application — Self-Improving LLM Pipeline

Entry point that wires together CORS, database lifecycle,
background monitor, and all API routers.
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
from app.routers import chat, logs, tuning, curation
from app.services.dataset_monitor import get_dataset_monitor

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

    # Start the background dataset monitor
    monitor = get_dataset_monitor()
    await monitor.start()
    logger.info("✅ Dataset monitor started.")

    yield

    # Shutdown
    await monitor.stop()
    logger.info("🛑 Dataset monitor stopped.")
    await close_db()
    logger.info("🛑 Backend shutdown complete.")


# ── App ─────────────────────────────────────────────────────

app = FastAPI(
    title="Self-Improving LLM Pipeline",
    description=(
        "Production-grade self-improving pipeline with hybrid evaluation "
        "(cosine + LLM-as-a-Judge), teacher-correction pattern, "
        "human-in-the-loop curation, and automated LoRA fine-tuning."
    ),
    version="2.0.0",
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
app.include_router(curation.router)


# ── Health Check ────────────────────────────────────────────

@app.get("/", tags=["health"])
async def health_check():
    """Root endpoint — API health check."""
    return {
        "status": "healthy",
        "service": "Self-Improving LLM Pipeline",
        "version": "2.0.0",
    }


@app.get("/api/health", tags=["health"])
async def api_health():
    """Detailed API health check."""
    from app.services.ml_service import get_ml_service

    ml = get_ml_service()
    feedback = ml.get_feedback_status()
    training = ml.get_training_status()
    curation_stats = ml.get_curation_stats()
    monitor = get_dataset_monitor()

    return {
        "status": "healthy",
        "pipeline": {
            "feedback_queue": feedback,
            "training": training,
            "curation": curation_stats,
            "dataset_monitor": monitor.get_status(),
        },
    }
