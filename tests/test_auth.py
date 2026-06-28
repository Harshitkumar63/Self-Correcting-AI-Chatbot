"""
Tests for the Authentication system.

Tests JWT token creation/verification, login endpoint,
role-based access control, and admin-only endpoints.
"""

import pytest

from app.services.auth_service import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)


# ── Password Hashing Tests ──────────────────────────────────

class TestPasswordHashing:
    """Test bcrypt password hashing and verification."""

    def test_hash_password(self):
        """Hashing should produce a different string."""
        password = "my_secure_password"
        hashed = hash_password(password)
        assert hashed != password
        assert len(hashed) > 20

    def test_verify_correct_password(self):
        """Correct password should verify True."""
        password = "test123"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_verify_wrong_password(self):
        """Wrong password should verify False."""
        hashed = hash_password("correct_password")
        assert verify_password("wrong_password", hashed) is False

    def test_different_hashes(self):
        """Same password should produce different hashes (salted)."""
        h1 = hash_password("same_password")
        h2 = hash_password("same_password")
        assert h1 != h2  # bcrypt uses random salt


# ── JWT Token Tests ──────────────────────────────────────────

class TestJWTTokens:
    """Test JWT token creation and decoding."""

    def test_create_and_decode_token(self):
        """Token should round-trip successfully."""
        data = {"sub": "admin", "role": "admin"}
        token = create_access_token(data)
        payload = decode_token(token)

        assert payload is not None
        assert payload["sub"] == "admin"
        assert payload["role"] == "admin"

    def test_token_contains_expiry(self):
        """Token payload should contain an 'exp' field."""
        token = create_access_token({"sub": "user1"})
        payload = decode_token(token)
        assert "exp" in payload

    def test_invalid_token_returns_none(self):
        """Invalid token should return None."""
        assert decode_token("invalid.token.string") is None

    def test_empty_token_returns_none(self):
        """Empty token should return None."""
        assert decode_token("") is None


# ── Login Endpoint Tests ─────────────────────────────────────

class TestLoginEndpoint:
    """Tests for POST /api/v1/auth/login."""

    @pytest.mark.asyncio
    async def test_login_success(self, client, seeded_admin):
        """Valid credentials should return a JWT token."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["role"] == "admin"
        assert data["username"] == "admin"

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client, seeded_admin):
        """Wrong password should return 401."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrongpass"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, client):
        """Non-existent username should return 401."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": "nobody", "password": "pass"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_empty_credentials(self, client):
        """Empty credentials should return 422."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": "", "password": ""},
        )
        assert response.status_code == 422


# ── /me Endpoint Tests ───────────────────────────────────────

class TestMeEndpoint:
    """Tests for GET /api/v1/auth/me."""

    @pytest.mark.asyncio
    async def test_me_with_valid_token(self, client, seeded_admin, admin_headers):
        """Should return user info with valid token."""
        response = await client.get("/api/v1/auth/me", headers=admin_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["username"] == "admin"
        assert data["role"] == "admin"

    @pytest.mark.asyncio
    async def test_me_without_token(self, client):
        """Should return 401 without token."""
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_me_with_invalid_token(self, client):
        """Should return 401 with invalid token."""
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401


# ── Role-Based Access Tests ──────────────────────────────────

class TestRoleBasedAccess:
    """Test that admin-only endpoints reject non-admin users."""

    @pytest.mark.asyncio
    async def test_tuning_requires_admin(self, client, seeded_user, user_headers):
        """POST /api/v1/trigger-tuning should require admin."""
        response = await client.post(
            "/api/v1/trigger-tuning",
            headers=user_headers,
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_curation_requires_admin(self, client, seeded_user, user_headers):
        """GET /api/v1/curation/queue should require admin."""
        response = await client.get(
            "/api/v1/curation/queue",
            headers=user_headers,
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_curation_stats_requires_admin(self, client, seeded_user, user_headers):
        """GET /api/v1/curation/stats should require admin."""
        response = await client.get(
            "/api/v1/curation/stats",
            headers=user_headers,
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_tuning_allowed_for_admin(self, client, seeded_admin, admin_headers):
        """POST /api/v1/trigger-tuning should work for admin."""
        response = await client.post(
            "/api/v1/trigger-tuning",
            headers=admin_headers,
        )
        # 200 = success (no samples), not 403
        assert response.status_code == 200


# ── Register Endpoint Tests ──────────────────────────────────

class TestRegisterEndpoint:
    """Tests for POST /api/v1/auth/register (admin only)."""

    @pytest.mark.asyncio
    async def test_register_as_admin(self, client, seeded_admin, admin_headers):
        """Admin should be able to create new users."""
        response = await client.post(
            "/api/v1/auth/register",
            json={"username": "newuser", "password": "newpass123", "role": "user"},
            headers=admin_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["username"] == "newuser"
        assert data["role"] == "user"

    @pytest.mark.asyncio
    async def test_register_duplicate_username(self, client, seeded_admin, admin_headers):
        """Duplicate username should return 409."""
        response = await client.post(
            "/api/v1/auth/register",
            json={"username": "admin", "password": "anotherpass", "role": "user"},
            headers=admin_headers,
        )
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_register_without_admin(self, client, seeded_user, user_headers):
        """Non-admin should not be able to register users."""
        response = await client.post(
            "/api/v1/auth/register",
            json={"username": "hack", "password": "hackpass", "role": "admin"},
            headers=user_headers,
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_register_without_auth(self, client):
        """Unauthenticated requests should be rejected."""
        response = await client.post(
            "/api/v1/auth/register",
            json={"username": "anon", "password": "anonpass", "role": "user"},
        )
        assert response.status_code == 401
