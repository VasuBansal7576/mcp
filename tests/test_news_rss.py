"""Unit tests for the RSS news ingestion service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.data.models import Article, Stock
from maverick_mcp.database.base import Base
from maverick_mcp.services.news_rss import fetch_rss_news_for_ticker


@pytest.fixture
def db_session():
    """Create in-memory SQLite database and return a session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    _Session = sessionmaker(bind=engine)
    session = _Session()
    yield session
    session.close()


@patch("urllib.request.urlopen")
def test_fetch_rss_news_success(mock_urlopen, db_session):
    """Test successfully fetching and parsing RSS feeds."""
    # Mock stock entry
    Stock.get_or_create(db_session, "AAPL", company_name="Apple Inc.")
    db_session.commit()

    # RSS XML content matching feedparser structure
    rss_xml = """<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0">
        <channel>
            <title>Test Feed</title>
            <link>http://test.com</link>
            <description>Test feed description</description>
            <item>
                <title>Apple Inc. AAPL reports record Q2 earnings</title>
                <link>http://test.com/aapl-earnings</link>
                <description>Apple (AAPL) did extremely well this quarter.</description>
                <pubDate>Fri, 29 May 2026 12:00:00 GMT</pubDate>
            </item>
        </channel>
    </rss>
    """
    mock_response = MagicMock()
    mock_response.__enter__.return_value.read.return_value = bytes(
        rss_xml, encoding="utf-8"
    )
    mock_urlopen.return_value = mock_response

    # Invoke
    fetch_rss_news_for_ticker("AAPL", db_session, limit=5)

    # Verify db state
    db_articles = db_session.query(Article).all()
    assert len(db_articles) == 1  # Deduplicated to 1 item
    assert db_articles[0].ticker == "AAPL"
    assert "AAPL reports record Q2" in db_articles[0].title
    assert db_articles[0].source in ["Seeking Alpha", "MarketWatch", "Benzinga"]
