"""
FastAPI Main Application — Self-Improving LLM Pipeline v3.0

Entry point that wires together CORS, database lifecycle,
background monitor, authentication, rate limiting, and all API routers.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Load environment variables from .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Ensure project root is on path for ml_pipeline imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.database import init_db, close_db, AsyncSessionLocal
from app.routers import auth, chat, conversations, curation, ground_truth, logs, tuning
from app.services.auth_service import create_default_admin
from app.services.dataset_monitor import get_dataset_monitor

# ── Logging ─────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Rate Limiter ────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    logger.info("🚀 Starting Self-Improving LLM Pipeline backend v3.0...")
    await init_db()
    logger.info("✅ Database initialized.")

    # Create default admin user
    async with AsyncSessionLocal() as session:
        await create_default_admin(session)

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
        "human-in-the-loop curation, JWT authentication, and automated LoRA fine-tuning."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# Attach limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ────────────────────────────────────────────────────

cors_origins = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Routers ─────────────────────────────────────────────────

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(logs.router)
app.include_router(tuning.router)
app.include_router(curation.router)
app.include_router(ground_truth.router)


# ── Health Check ────────────────────────────────────────────

@app.get("/", tags=["health"])
async def health_check():
    """Root endpoint — API health check."""
    return {
        "status": "healthy",
        "service": "Self-Improving LLM Pipeline",
        "version": "3.0.0",
    }


@app.get("/api/v1/health", tags=["health"])
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
        "version": "3.0.0",
        "pipeline": {
            "feedback_queue": feedback,
            "training": training,
            "curation": curation_stats,
            "dataset_monitor": monitor.get_status(),
        },
    }
