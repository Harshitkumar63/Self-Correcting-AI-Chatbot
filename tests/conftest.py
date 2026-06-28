"""
Shared test fixtures for the Self-Improving LLM Pipeline.

Provides:
- Async test database (in-memory SQLite)
- FastAPI test client with overridden DB dependency
- Mock ML service
- Temporary directory for test data
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Set test environment variables before importing app modules
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"
os.environ["DEFAULT_ADMIN_USERNAME"] = "admin"
os.environ["DEFAULT_ADMIN_PASSWORD"] = "admin123"

from app.database import Base, get_db
from app.main import app
from app.services.auth_service import create_access_token, hash_password


# ── Test Database ───────────────────────────────────────────

TEST_ENGINE = create_async_engine(
    "sqlite+aiosqlite://",
    connect_args={"check_same_thread": False},
)
TestSessionLocal = async_sessionmaker(
    bind=TEST_ENGINE, class_=AsyncSession, expire_on_commit=False
)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    """Create all tables before each test, drop them after."""
    async with TEST_ENGINE.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with TEST_ENGINE.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session():
    """Provide a clean database session for each test."""
    async with TestSessionLocal() as session:
        yield session


async def _override_get_db():
    """Override the get_db dependency to use the test database."""
    async with TestSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ── Mock ML Service ─────────────────────────────────────────

def _create_mock_ml_service():
    """Create a mock MLService with reasonable defaults."""
    mock = MagicMock()

    # process_query returns a realistic result
    mock.process_query = AsyncMock(return_value={
        "response": "Machine learning is a subset of artificial intelligence.",
        "similarity_score": 0.85,
        "evaluation_status": "Passed",
        "ground_truth": "Machine learning is a branch of AI.",
        "matched_query": "What is machine learning?",
        "feedback_status": None,
        "hybrid_score": 0.82,
        "llm_judge_score": 0.78,
        "hallucination_detected": False,
        "completeness": 4,
        "teacher_correction": None,
    })

    # Feedback status
    mock.get_feedback_status.return_value = {
        "sample_count": 5,
        "threshold": 50,
        "progress_pct": 10.0,
        "should_trigger": False,
    }

    # Training status
    mock.get_training_status.return_value = {
        "state": "idle",
        "message": "No training in progress.",
        "started_at": None,
        "completed_at": None,
        "error": None,
    }

    # Curation queue
    mock.get_curation_queue.return_value = {
        "items": [],
        "stats": {"total": 0, "pending": 0, "approved": 0, "rejected": 0},
    }
    mock.get_curation_all.return_value = {
        "items": [],
        "stats": {"total": 0, "pending": 0, "approved": 0, "rejected": 0},
    }
    mock.get_curation_stats.return_value = {
        "total": 0, "pending": 0, "approved": 0, "rejected": 0,
    }

    return mock


@pytest.fixture
def mock_ml_service():
    """Provide a mock ML service."""
    return _create_mock_ml_service()


# ── FastAPI Test Client ─────────────────────────────────────

@pytest_asyncio.fixture
async def client(mock_ml_service):
    """
    Async test client with overridden database and ML service.
    """
    from app.services.ml_service import get_ml_service

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_ml_service] = lambda: mock_ml_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Auth Helpers ────────────────────────────────────────────

@pytest.fixture
def admin_token():
    """Generate a valid admin JWT token for test requests."""
    return create_access_token(data={"sub": "admin", "role": "admin"})


@pytest.fixture
def user_token():
    """Generate a valid user JWT token for test requests."""
    return create_access_token(data={"sub": "testuser", "role": "user"})


@pytest.fixture
def admin_headers(admin_token):
    """Headers dict with admin Bearer token."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def user_headers(user_token):
    """Headers dict with user Bearer token."""
    return {"Authorization": f"Bearer {user_token}"}


@pytest_asyncio.fixture
async def seeded_admin(db_session):
    """Create an admin user in the test database."""
    from app.models import User
    user = User(
        username="admin",
        hashed_password=hash_password("admin123"),
        role="admin",
    )
    db_session.add(user)
    await db_session.commit()
    return user


@pytest_asyncio.fixture
async def seeded_user(db_session):
    """Create a regular user in the test database."""
    from app.models import User
    user = User(
        username="testuser",
        hashed_password=hash_password("testpass"),
        role="user",
    )
    db_session.add(user)
    await db_session.commit()
    return user


# ── Temp Directories ────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary directory for test data files."""
    return tmp_path
