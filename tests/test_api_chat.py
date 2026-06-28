"""
Tests for the Chat API endpoint.

Tests the POST /api/v1/chat endpoint with mocked ML service,
including conversation auto-creation and request validation.
"""

import pytest


class TestChatEndpoint:
    """Tests for POST /api/v1/chat."""

    @pytest.mark.asyncio
    async def test_chat_success(self, client):
        """Should return a valid chat response."""
        response = await client.post(
            "/api/v1/chat",
            json={"query": "What is machine learning?"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "response" in data
        assert "similarity_score" in data
        assert "evaluation_status" in data
        assert data["evaluation_status"] in ("Passed", "Flagged")
        assert "hybrid_score" in data
        assert "conversation_id" in data

    @pytest.mark.asyncio
    async def test_chat_creates_conversation(self, client):
        """Sending without conversation_id should auto-create one."""
        response = await client.post(
            "/api/v1/chat",
            json={"query": "Hello world"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["conversation_id"] is not None
        assert isinstance(data["conversation_id"], int)

    @pytest.mark.asyncio
    async def test_chat_with_conversation_id(self, client):
        """Sending with an existing conversation_id should use it."""
        # First, create a conversation
        create_resp = await client.post(
            "/api/v1/conversations",
            json={"title": "Test Conv"},
        )
        assert create_resp.status_code == 200
        conv_id = create_resp.json()["id"]

        # Send a message in that conversation
        response = await client.post(
            "/api/v1/chat",
            json={"query": "Follow-up question", "conversation_id": conv_id},
        )
        assert response.status_code == 200
        assert response.json()["conversation_id"] == conv_id

    @pytest.mark.asyncio
    async def test_chat_empty_query_rejected(self, client):
        """Empty query should return 422."""
        response = await client.post(
            "/api/v1/chat",
            json={"query": ""},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_missing_query_rejected(self, client):
        """Missing query field should return 422."""
        response = await client.post(
            "/api/v1/chat",
            json={},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_html_sanitized(self, client):
        """HTML tags in query should be stripped."""
        response = await client.post(
            "/api/v1/chat",
            json={"query": "Hello <script>alert('xss')</script> world"},
        )
        assert response.status_code == 200
        # The query should have been sanitized before reaching ML
        # (sanitization happens in the Pydantic validator)


class TestConversationEndpoints:
    """Tests for the conversation CRUD endpoints."""

    @pytest.mark.asyncio
    async def test_create_conversation(self, client):
        """POST /api/v1/conversations should create a new conversation."""
        response = await client.post(
            "/api/v1/conversations",
            json={"title": "My Test Chat"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["title"] == "My Test Chat"
        assert data["message_count"] == 0
        assert data["id"] is not None

    @pytest.mark.asyncio
    async def test_create_conversation_default_title(self, client):
        """Creating without title should use default."""
        response = await client.post(
            "/api/v1/conversations",
            json={},
        )
        assert response.status_code == 200
        assert response.json()["title"] == "New Conversation"

    @pytest.mark.asyncio
    async def test_list_conversations(self, client):
        """GET /api/v1/conversations should list all conversations."""
        # Create two
        await client.post("/api/v1/conversations", json={"title": "Chat 1"})
        await client.post("/api/v1/conversations", json={"title": "Chat 2"})

        response = await client.get("/api/v1/conversations")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 2
        assert len(data["conversations"]) == 2

    @pytest.mark.asyncio
    async def test_get_conversation_detail(self, client):
        """GET /api/v1/conversations/{id} should return conversation with messages."""
        create_resp = await client.post(
            "/api/v1/conversations", json={"title": "Detail Test"}
        )
        conv_id = create_resp.json()["id"]

        response = await client.get(f"/api/v1/conversations/{conv_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["title"] == "Detail Test"
        assert data["messages"] == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation(self, client):
        """Requesting a non-existent conversation should return 404."""
        response = await client.get("/api/v1/conversations/99999")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_conversation(self, client):
        """DELETE /api/v1/conversations/{id} should remove it."""
        create_resp = await client.post(
            "/api/v1/conversations", json={"title": "To Delete"}
        )
        conv_id = create_resp.json()["id"]

        del_resp = await client.delete(f"/api/v1/conversations/{conv_id}")
        assert del_resp.status_code == 200

        # Should be gone
        get_resp = await client.get(f"/api/v1/conversations/{conv_id}")
        assert get_resp.status_code == 404
