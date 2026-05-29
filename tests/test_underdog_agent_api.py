"""Unit and integration tests for Underdog Stock Agent and endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.agents.underdog_agent import UnderdogAgent, calculate_fingerprint
from maverick_mcp.data.models import Article, Stock
from maverick_mcp.database.base import Base
from maverick_mcp.providers.cost_tracking import get_global_cost_accumulator


@pytest.fixture
def test_db():
    """Create a temporary in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def mock_llm_response():
    """Fixture to mock LangChain LLM responses."""

    class MockAIMessage:
        def __init__(self, content, prompt_tokens=100, completion_tokens=50):
            self.content = content
            self.usage_metadata = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

    return MockAIMessage


def test_calculate_fingerprint():
    """Test MD5 fingerprint cache key generation."""
    h1 = calculate_fingerprint(
        "AAPL", "2026-05-29T12:00:00", "2026-05-29", "2026-05-28", "gpt-4o-mini"
    )
    h2 = calculate_fingerprint(
        "AAPL", "2026-05-29T12:00:00", "2026-05-29", "2026-05-28", "gpt-4o-mini"
    )
    h3 = calculate_fingerprint(
        "AAPL", "2026-05-29T12:00:01", "2026-05-29", "2026-05-28", "gpt-4o-mini"
    )

    assert h1 == h2
    assert h1 != h3


@patch("maverick_mcp.agents.underdog_agent.get_llm")
def test_underdog_agent_workflow(mock_get_llm, test_db, mock_llm_response):
    """Test the 3-node agent workflow: sentiment, risk, and synthesis."""
    # Seed db with a stock and data
    stock = Stock.get_or_create(
        test_db,
        "MSFT",
        company_name="Microsoft Corp",
        sector="Technology",
        pe_ratio=Decimal("25.0"),
        analyst_count=1,
        fcf_growth_yoy=Decimal("0.10"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        description="Software behemoth expanding into AI.",
    )
    test_db.commit()

    # Create mock articles
    art = Article(
        stock_id=stock.stock_id,
        ticker="MSFT",
        title="Microsoft Launches New AI Copilot",
        summary="Microsoft has announced a revolutionary new AI agent.",
        link="https://example.com/msft-ai",
        published_date=datetime.now(UTC),
        source="Seeking Alpha",
    )
    test_db.add(art)
    test_db.commit()

    # Mock the LLM to return JSON responses for the three nodes sequentially
    llm_instance = MagicMock()
    mock_get_llm.return_value = llm_instance

    sentiment_json = json.dumps(
        {
            "sentiment_score": 0.8,
            "sentiment_summary": "Highly positive news on AI launches.",
        }
    )
    risk_json = json.dumps(
        {"risk_level": "LOW", "risk_summary": "Stable cash flows reduce overall risk."}
    )
    synthesis_json = json.dumps(
        {
            "recommendation": "BUY",
            "confidence": 90,
            "reasoning": "Excellent fundamentals matched with dominant AI sentiment and low risk factors.",
        }
    )

    llm_instance.invoke.side_effect = [
        mock_llm_response(sentiment_json),
        mock_llm_response(risk_json),
        mock_llm_response(synthesis_json),
    ]

    agent = UnderdogAgent(test_db)
    result = agent.analyze_ticker("MSFT")

    # Verifications
    assert result["ticker"] == "MSFT"
    assert result["recommendation"] == "BUY"
    assert result["confidence"] == 90
    assert result["sentiment_score"] == 0.8
    assert result["risk_level"] == "LOW"
    assert result["stale_data"] is False
    assert not result["flags"]
    assert llm_instance.invoke.call_count == 3


@patch("maverick_mcp.agents.underdog_agent.get_llm")
def test_underdog_agent_staleness_guard(mock_get_llm, test_db, mock_llm_response):
    """Test that the Staleness Guard drops confidence by 50% when data is older than 24 hours."""
    # Seed old stock data (> 24 hours old)
    stock = Stock.get_or_create(
        test_db,
        "AAPL",
        company_name="Apple Inc",
        sector="Technology",
        pe_ratio=Decimal("28.0"),
        analyst_count=1,
        fcf_growth_yoy=Decimal("0.05"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        description="Consumer electronics giant.",
    )
    # Manually override updated_at to be 48 hours ago
    stock.updated_at = datetime.now(UTC) - timedelta(hours=48)
    test_db.commit()

    # Create mock article for AAPL so sentiment node executes
    art = Article(
        stock_id=stock.stock_id,
        ticker="AAPL",
        title="Apple Launches M4 Macs",
        summary="Apple has announced new high-performance Mac models.",
        link="https://example.com/aapl-mac",
        published_date=datetime.now(UTC),
        source="Seeking Alpha",
    )
    test_db.add(art)
    test_db.commit()

    llm_instance = MagicMock()
    mock_get_llm.return_value = llm_instance

    sentiment_json = json.dumps(
        {"sentiment_score": 0.2, "sentiment_summary": "Neutral sentiment."}
    )
    risk_json = json.dumps(
        {"risk_level": "MED", "risk_summary": "Medium operational risk."}
    )
    synthesis_json = json.dumps(
        {"recommendation": "BUY", "confidence": 80, "reasoning": "Strong fundamentals."}
    )

    llm_instance.invoke.side_effect = [
        mock_llm_response(sentiment_json),
        mock_llm_response(risk_json),
        mock_llm_response(synthesis_json),
    ]

    agent = UnderdogAgent(test_db)
    result = agent.analyze_ticker("AAPL")

    # Verification: Staleness triggers and confidence drops from 80 to 40
    assert result["stale_data"] is True
    assert "STALE_DATA" in result["flags"]
    assert result["confidence"] == 40
    assert "STALE_DATA" in result["reasoning"]


@pytest.fixture
def api_client():
    """Create a TestClient for testing FastAPI routes."""
    from fastapi import FastAPI

    from maverick_mcp.api.routers.underdog import router as underdog_router

    app = FastAPI()
    app.include_router(underdog_router)
    return TestClient(app)


def test_api_health_endpoint(api_client):
    """Test GET /api/underdog/health."""
    response = api_client.get("/api/underdog/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database_health" in data
    assert "cache_health" in data
    assert "active_llm_provider" in data


def test_api_cost_endpoint(api_client):
    """Test GET /api/underdog/cost."""
    # Let's clear records first
    accumulator = get_global_cost_accumulator()
    accumulator._records.clear()
    accumulator._daily_total = 0.0

    response = api_client.get("/api/underdog/cost")
    assert response.status_code == 200
    data = response.json()
    assert data["llm_call_count"] == 0
    assert data["input_tokens"] == 0
    assert data["output_tokens"] == 0
    assert data["usd_spend"] == 0.0


@patch("maverick_mcp.api.routers.underdog.UnderdogScreener")
@patch("maverick_mcp.api.routers.underdog.UnderdogAgent")
def test_api_screen_endpoint(mock_agent_class, mock_screener_class, api_client):
    """Test POST /api/underdog/screen with mocked pre-screener and agent reasoning."""
    # Mock pre-screening to return 1 candidate
    mock_screener = MagicMock()
    mock_screener_class.return_value = mock_screener
    mock_screener.screen_stocks.return_value = [
        {
            "symbol": "TSLA",
            "ticker": "TSLA",
            "company_name": "Tesla Inc",
            "sector": "Automotive",
            "pe_ratio": 45.0,
            "sector_median_pe": 50.0,
            "analyst_count": 2,
            "fcf_growth_yoy": 0.15,
            "short_interest_declining": True,
            "insider_buying_6m": True,
            "close_price": 200.0,
        }
    ]

    # Mock agent reasoning output
    mock_agent = MagicMock()
    mock_agent_class.return_value = mock_agent
    mock_agent.analyze_ticker.return_value = {
        "ticker": "TSLA",
        "company_name": "Tesla Inc",
        "recommendation": "BUY",
        "confidence": 85,
        "reasoning": "Superb electric vehicle outlook and cheap relative PE.",
        "sentiment_score": 0.6,
        "sentiment_summary": "Highly positive news.",
        "risk_level": "MED",
        "risk_summary": "Competition risks present.",
        "flags": [],
        "stale_data": False,
        "model_used": "gpt-4o-mini",
        "timestamp": "2026-05-29T12:00:00Z",
    }

    response = api_client.post("/api/underdog/screen")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["count"] == 1
    assert data["candidates"][0]["ticker"] == "TSLA"
    assert data["candidates"][0]["recommendation"] == "BUY"
    assert data["candidates"][0]["confidence"] == 85
    assert data["candidates"][0]["pe_ratio"] == 45.0
