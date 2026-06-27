"""
Auth Router — POST /api/v1/auth/login, /register, /me

Handles user authentication with JWT tokens.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import LoginRequest, TokenResponse, UserCreate, UserResponse
from app.services.auth_service import (
    create_access_token,
    hash_password,
    verify_password,
    require_auth,
    require_admin,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate user and return a JWT access token.
    """
    result = await db.execute(
        select(User).where(User.username == request.username)
    )
    user = result.scalar_one_or_none()

    if user is None or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = create_access_token(data={"sub": user.username, "role": user.role})

    logger.info("User logged in: %s (role=%s)", user.username, user.role)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        username=user.username,
    )


@router.post("/register", response_model=UserResponse)
async def register(
    request: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Create a new user account. Admin only.
    """
    # Check if username already exists
    result = await db.execute(
        select(User).where(User.username == request.username)
    )
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{request.username}' already exists",
        )

    user = User(
        username=request.username,
        hashed_password=hash_password(request.password),
        role=request.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info(
        "New user registered: %s (role=%s) by admin %s",
        user.username, user.role, current_user.username,
    )

    return UserResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        created_at=user.created_at.isoformat() if user.created_at else None,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(require_auth),
):
    """
    Get the current authenticated user's profile.
    """
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        role=current_user.role,
        created_at=current_user.created_at.isoformat() if current_user.created_at else None,
    )
