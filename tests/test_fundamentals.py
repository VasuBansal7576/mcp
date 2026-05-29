"""Unit tests for the stock fundamentals sync service."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.data.models import Stock
from maverick_mcp.database.base import Base
from maverick_mcp.services.fundamentals import sync_fundamentals_for_ticker


@pytest.fixture
def db_session():
    """Create in-memory SQLite database and return a session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    _Session = sessionmaker(bind=engine)
    session = _Session()
    yield session
    session.close()


@patch("yfinance.Ticker")
def test_sync_fundamentals_success(mock_ticker, db_session):
    """Test successfully syncing fundamentals using yfinance."""
    # Mock yfinance response
    mock_instance = MagicMock()
    mock_instance.info = {
        "trailingPE": 15.5,
        "numberOfAnalystOpinions": 3,
        "freeCashflow": 250000000,
        "freeCashflowGrowth": 0.12,
        "sharesShort": 150000,
        "shortPercentOfFloat": 0.02,
    }
    mock_ticker.return_value = mock_instance

    # Invoke
    result = sync_fundamentals_for_ticker("AAPL", db_session)

    # Verify return dict
    assert result["ticker"] == "AAPL"
    assert result["pe_ratio"] == 15.5
    assert result["analyst_count"] == 3
    assert result["fcf_growth_yoy"] == 0.12

    # Verify db state
    stock = db_session.query(Stock).filter_by(ticker_symbol="AAPL").first()
    assert stock is not None
    assert stock.pe_ratio == Decimal("15.5")
    assert stock.analyst_count == 3
    assert stock.fcf_growth_yoy == Decimal("0.12")
    assert stock.short_interest == Decimal("150000")
