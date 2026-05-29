"""Unit tests for the Underdog Stock Screener."""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.data.models import Stock
from maverick_mcp.database.base import Base
from maverick_mcp.services.screening.underdog_screener import UnderdogScreener


@pytest.fixture
def db_session():
    """Create in-memory SQLite database and return a session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    _Session = sessionmaker(bind=engine)
    session = _Session()
    yield session
    session.close()


def test_underdog_screener_filters_correctly(db_session):
    """Test that the UnderdogScreener correctly filters stocks based on the five rules."""
    # Sector median P/E will be calculated from the database
    # Let's populate the database with several stocks in the 'Technology' sector

    # 1. Sector Reference Stocks (Median P/E will be 20.0)
    Stock.get_or_create(
        db_session,
        "REF1",
        sector="Technology",
        pe_ratio=Decimal("15.0"),
        analyst_count=10,
        is_active=True,
    )
    Stock.get_or_create(
        db_session,
        "REF2",
        sector="Technology",
        pe_ratio=Decimal("20.0"),
        analyst_count=10,
        is_active=True,
    )
    Stock.get_or_create(
        db_session,
        "REF3",
        sector="Technology",
        pe_ratio=Decimal("25.0"),
        analyst_count=10,
        is_active=True,
    )

    # 2. Perfect Underdog Candidate (Matches all 5 criteria)
    Stock.get_or_create(
        db_session,
        "UNDG",
        company_name="Underdog Tech Inc.",
        sector="Technology",
        pe_ratio=Decimal("10.0"),  # 10.0 < Median (17.5)
        analyst_count=3,  # < 5
        fcf_growth_yoy=Decimal("0.12"),  # > 0
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
    )

    # 3. Fails analyst count (6 >= 5)
    Stock.get_or_create(
        db_session,
        "FAIL1",
        sector="Other",
        pe_ratio=Decimal("10.0"),
        analyst_count=6,
        fcf_growth_yoy=Decimal("0.10"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
    )

    # 4. Fails P/E ratio (30.0 >= Median 17.5)
    Stock.get_or_create(
        db_session,
        "FAIL2",
        sector="Technology",
        pe_ratio=Decimal("30.0"),
        analyst_count=2,
        fcf_growth_yoy=Decimal("0.05"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
    )

    # 5. Fails FCF growth (-0.02 <= 0)
    Stock.get_or_create(
        db_session,
        "FAIL3",
        sector="Other",
        pe_ratio=Decimal("10.0"),
        analyst_count=2,
        fcf_growth_yoy=Decimal("-0.02"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
    )

    # 6. Fails short interest declining (False)
    Stock.get_or_create(
        db_session,
        "FAIL4",
        sector="Other",
        pe_ratio=Decimal("10.0"),
        analyst_count=2,
        fcf_growth_yoy=Decimal("0.08"),
        short_interest_declining=False,
        insider_buying_6m=True,
        is_active=True,
    )

    # 7. Fails insider buying (False)
    Stock.get_or_create(
        db_session,
        "FAIL5",
        sector="Other",
        pe_ratio=Decimal("10.0"),
        analyst_count=2,
        fcf_growth_yoy=Decimal("0.08"),
        short_interest_declining=True,
        insider_buying_6m=False,
        is_active=True,
    )

    db_session.commit()

    # Instantiate screener and run
    screener = UnderdogScreener(db_session)
    results = screener.screen_stocks()

    # Assertions
    tickers = [r["ticker"] for r in results]
    assert "FAIL5" not in tickers
    assert len(results) == 1


def test_underdog_screener_graceful_degradation(db_session):
    """Test that the UnderdogScreener gracefully degrades and flags NO_INSIDER_DATA when SEC EDGAR is down."""
    from maverick_mcp.data.cache import clear_cache, save_to_cache

    # Sector median P/E will be calculated from the database
    Stock.get_or_create(
        db_session,
        "REF1",
        sector="Technology",
        pe_ratio=Decimal("15.0"),
        analyst_count=10,
        is_active=True,
    )
    Stock.get_or_create(
        db_session,
        "REF2",
        sector="Technology",
        pe_ratio=Decimal("20.0"),
        analyst_count=10,
        is_active=True,
    )

    # Candidate with insider buying
    Stock.get_or_create(
        db_session,
        "UNDG",
        company_name="Underdog Tech Inc.",
        sector="Technology",
        pe_ratio=Decimal("10.0"),
        analyst_count=3,
        fcf_growth_yoy=Decimal("0.12"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
    )

    # Candidate without insider buying (should only be matched if SEC EDGAR is down)
    Stock.get_or_create(
        db_session,
        "FAIL5",
        company_name="Fail Insider Inc.",
        sector="Technology",
        pe_ratio=Decimal("11.0"),
        analyst_count=2,
        fcf_growth_yoy=Decimal("0.08"),
        short_interest_declining=True,
        insider_buying_6m=False,
        is_active=True,
    )

    db_session.commit()

    # Trip the SEC EDGAR circuit breaker in cache
    save_to_cache("underdog:circuit_breaker:sec_edgar", True, ttl=300)

    screener = UnderdogScreener(db_session)
    results = screener.screen_stocks()

    # Clean up the cache key
    clear_cache("underdog:circuit_breaker:sec_edgar")

    # Both UNDG and FAIL5 should now match because the insider buying rule is skipped
    tickers = [r["ticker"] for r in results]
    assert "UNDG" in tickers
    assert "FAIL5" in tickers
    assert len(results) == 2

    # Verify that the NO_INSIDER_DATA flag is present for candidates
    fail_insider_cand = next(r for r in results if r["ticker"] == "FAIL5")
    assert "flags" in fail_insider_cand
    assert "NO_INSIDER_DATA" in fail_insider_cand["flags"]
