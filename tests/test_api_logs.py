"""
Tests for the Logs API endpoints.

Tests pagination, stats, chart data, and export functionality.
"""

import pytest


class TestLogsEndpoint:
    """Tests for GET /api/v1/logs."""

    @pytest.mark.asyncio
    async def test_logs_empty(self, client):
        """Should return empty logs when no interactions exist."""
        response = await client.get("/api/v1/logs")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 0
        assert data["logs"] == []
        assert data["page"] == 1

    @pytest.mark.asyncio
    async def test_logs_after_chat(self, client):
        """Should return logs after sending chat messages."""
        # Send a chat message
        await client.post("/api/v1/chat", json={"query": "Test query"})

        response = await client.get("/api/v1/logs")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 1
        assert len(data["logs"]) == 1
        assert data["logs"][0]["user_query"] == "Test query"

    @pytest.mark.asyncio
    async def test_logs_pagination(self, client):
        """Pagination should work correctly."""
        # Send multiple messages
        for i in range(5):
            await client.post("/api/v1/chat", json={"query": f"Query {i}"})

        # Page 1 with per_page=2
        response = await client.get("/api/v1/logs?page=1&per_page=2")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 5
        assert len(data["logs"]) == 2
        assert data["total_pages"] == 3

    @pytest.mark.asyncio
    async def test_logs_invalid_page(self, client):
        """Invalid page numbers should return 422."""
        response = await client.get("/api/v1/logs?page=0")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_logs_per_page_limit(self, client):
        """per_page exceeding 100 should return 422."""
        response = await client.get("/api/v1/logs?per_page=200")
        assert response.status_code == 422


class TestStatsEndpoint:
    """Tests for GET /api/v1/logs/stats."""

    @pytest.mark.asyncio
    async def test_stats_empty(self, client):
        """Stats should return zeros when empty."""
        response = await client.get("/api/v1/logs/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_queries"] == 0
        assert data["average_score"] == 0.0
        assert data["flagged_count"] == 0
        assert data["passed_count"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_chat(self, client):
        """Stats should update after chat interactions."""
        await client.post("/api/v1/chat", json={"query": "Test"})

        response = await client.get("/api/v1/logs/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_queries"] == 1


class TestChartDataEndpoint:
    """Tests for GET /api/v1/logs/chart-data."""

    @pytest.mark.asyncio
    async def test_chart_data_empty(self, client):
        """Chart data should handle empty state."""
        response = await client.get("/api/v1/logs/chart-data")
        assert response.status_code == 200

        data = response.json()
        assert "score_distribution" in data
        assert "trend_data" in data
        assert "status_breakdown" in data

    @pytest.mark.asyncio
    async def test_chart_data_structure(self, client):
        """Chart data should have the right structure after interactions."""
        await client.post("/api/v1/chat", json={"query": "Test"})

        response = await client.get("/api/v1/logs/chart-data")
        data = response.json()

        assert len(data["score_distribution"]) == 10  # 10 buckets
        assert all("range" in b and "count" in b for b in data["score_distribution"])
        assert "pass_rate" in data["status_breakdown"]


class TestExportEndpoint:
    """Tests for GET /api/v1/logs/export."""

    @pytest.mark.asyncio
    async def test_export_json(self, client):
        """Should export logs as JSON."""
        await client.post("/api/v1/chat", json={"query": "Test"})

        response = await client.get("/api/v1/logs/export?format=json")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_export_csv(self, client):
        """Should export logs as CSV."""
        await client.post("/api/v1/chat", json={"query": "Test"})

        response = await client.get("/api/v1/logs/export?format=csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_export_csv_headers(self, client):
        """CSV export should have proper headers."""
        await client.post("/api/v1/chat", json={"query": "CSV test"})

        response = await client.get("/api/v1/logs/export?format=csv")
        content = response.text
        assert "id" in content
        assert "user_query" in content
        assert "similarity_score" in content
