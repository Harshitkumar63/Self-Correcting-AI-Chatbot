"""
Authentication Service — JWT token management and password hashing.

Provides:
- Password hashing with bcrypt
- JWT token creation and verification
- FastAPI dependencies for auth guards
- Default admin user creation
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRATION_MINUTES", "1440"))

# ── Password Hashing ───────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


# ── JWT Tokens ──────────────────────────────────────────────


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT token. Returns payload or None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ── FastAPI Dependencies ────────────────────────────────────


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """
    FastAPI dependency to get the current authenticated user.
    Returns None if no valid token is provided (for optional auth).
    """
    if credentials is None:
        return None

    payload = decode_token(credentials.credentials)
    if payload is None:
        return None

    username = payload.get("sub")
    if username is None:
        return None

    from app.models import User
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    return user


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """
    FastAPI dependency that requires authentication.
    Raises 401 if not authenticated.
    """
    user = await get_current_user(credentials, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """
    FastAPI dependency that requires admin role.
    Raises 401/403 if not authenticated or not admin.
    """
    user = await require_auth(credentials, db)
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


# ── Default Admin ───────────────────────────────────────────


async def create_default_admin(db: AsyncSession) -> None:
    """Create the default admin user if no users exist."""
    from app.models import User

    result = await db.execute(select(User).limit(1))
    if result.scalar_one_or_none() is not None:
        return  # Users already exist

    admin_username = os.environ.get("DEFAULT_ADMIN_USERNAME", "admin")
    admin_password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin123")

    admin = User(
        username=admin_username,
        hashed_password=hash_password(admin_password),
        role="admin",
    )
    db.add(admin)
    await db.commit()
    logger.info("✅ Default admin user created: %s", admin_username)
