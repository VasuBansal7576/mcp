"""Unit tests for the production-ready, risk-controlled Underdog Screener Agent."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.api.routers.underdog import (
    dispatch_push_alert,
    get_underdog_portfolio_allocate,
    get_underdog_portfolio_monitor,
)
from maverick_mcp.data.models import PortfolioPosition, PriceCache, Stock, UserPortfolio
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


def test_underdog_production_screener_filters_correctly(db_session):
    """Test that the production UnderdogScreener filters by ROIC, F-Score, and high FCF growth floor."""
    # Tech sector peers to establish a high sector median (median will be 40.0)
    Stock.get_or_create(
        db_session,
        "REF1",
        sector="Technology",
        pe_ratio=Decimal("30.0"),
        analyst_count=10,
        is_active=True,
    )
    Stock.get_or_create(
        db_session,
        "REF2",
        sector="Technology",
        pe_ratio=Decimal("40.0"),
        analyst_count=10,
        is_active=True,
    )
    Stock.get_or_create(
        db_session,
        "REF3",
        sector="Technology",
        pe_ratio=Decimal("50.0"),
        analyst_count=10,
        is_active=True,
    )

    # 1. Perfect turnaround candidate matching ROIC and F-Score
    good_stock = Stock.get_or_create(
        db_session,
        "GOOD",
        company_name="Good Corp",
        sector="Technology",
        pe_ratio=Decimal("12.0"),  # 12.0 < 40.0 median
        analyst_count=3,
        fcf_growth_yoy=Decimal("0.10"),  # > 5%
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        roic=Decimal("15.0"),  # roic >= 10.0
        piotroski_f_score=7,  # f-score >= 6
    )
    # Add PriceCache with good close price and high volume
    db_session.add(
        PriceCache(
            stock_id=good_stock.stock_id,
            date=datetime.now(UTC).date(),
            open_price=Decimal("10.00"),
            high_price=Decimal("10.00"),
            low_price=Decimal("10.00"),
            close_price=Decimal("10.00"),
            volume=600000,  # > 500k volume
        )
    )

    # 2. Fails FCF growth floor (4% < 5%)
    fail_fcf = Stock.get_or_create(
        db_session,
        "FAIL_FCF",
        sector="Other",
        pe_ratio=Decimal("10.0"),
        analyst_count=2,
        fcf_growth_yoy=Decimal("0.04"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        roic=Decimal("12.0"),
        piotroski_f_score=7,
    )
    db_session.add(
        PriceCache(
            stock_id=fail_fcf.stock_id,
            date=datetime.now(UTC).date(),
            close_price=Decimal("10.00"),
            volume=600000,
        )
    )

    # 3. Fails ROIC filter (8.0% < 10.0%)
    fail_roic = Stock.get_or_create(
        db_session,
        "FAIL_ROIC",
        sector="Other",
        pe_ratio=Decimal("10.0"),
        analyst_count=2,
        fcf_growth_yoy=Decimal("0.08"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        roic=Decimal("8.0"),
        piotroski_f_score=7,
    )
    db_session.add(
        PriceCache(
            stock_id=fail_roic.stock_id,
            date=datetime.now(UTC).date(),
            close_price=Decimal("10.00"),
            volume=600000,
        )
    )

    # 4. Fails Price Floor ($4.00 < $5.00)
    fail_price = Stock.get_or_create(
        db_session,
        "FAIL_PRICE",
        sector="Other",
        pe_ratio=Decimal("10.0"),
        analyst_count=2,
        fcf_growth_yoy=Decimal("0.08"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        roic=Decimal("12.0"),
        piotroski_f_score=7,
    )
    db_session.add(
        PriceCache(
            stock_id=fail_price.stock_id,
            date=datetime.now(UTC).date(),
            close_price=Decimal("4.00"),  # Cheap penny stock
            volume=600000,
        )
    )

    # 5. Fails Volume Floor (100k < 500k)
    fail_vol = Stock.get_or_create(
        db_session,
        "FAIL_VOL",
        sector="Other",
        pe_ratio=Decimal("10.0"),
        analyst_count=2,
        fcf_growth_yoy=Decimal("0.08"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        roic=Decimal("12.0"),
        piotroski_f_score=7,
    )
    db_session.add(
        PriceCache(
            stock_id=fail_vol.stock_id,
            date=datetime.now(UTC).date(),
            close_price=Decimal("10.00"),
            volume=100000,  # Illiquid
        )
    )

    # 6. Fails Earnings Proximity Guard (< 14 days)
    fail_earnings = Stock.get_or_create(
        db_session,
        "FAIL_EARN",
        sector="Technology",
        pe_ratio=Decimal("10.0"),
        analyst_count=2,
        fcf_growth_yoy=Decimal("0.08"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        roic=Decimal("12.0"),
        piotroski_f_score=7,
        next_earnings_date=date.today() + timedelta(days=5),  # in 5 days
    )
    db_session.add(
        PriceCache(
            stock_id=fail_earnings.stock_id,
            date=datetime.now(UTC).date(),
            close_price=Decimal("10.00"),
            volume=600000,
        )
    )

    # 7. Fails Forensic Cash Flow Filter (operating_cash_flow <= 0.0)
    fail_forensic = Stock.get_or_create(
        db_session,
        "FAIL_FOREN",
        sector="Technology",
        pe_ratio=Decimal("10.0"),
        analyst_count=2,
        fcf_growth_yoy=Decimal("0.08"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        roic=Decimal("12.0"),
        piotroski_f_score=7,
        operating_cash_flow=Decimal("-50.00"),  # Negative operating cash flow
    )
    db_session.add(
        PriceCache(
            stock_id=fail_forensic.stock_id,
            date=datetime.now(UTC).date(),
            close_price=Decimal("10.00"),
            volume=600000,
        )
    )

    db_session.commit()

    screener = UnderdogScreener(db_session)
    candidates = screener.screen_stocks()

    assert len(candidates) == 1
    assert candidates[0]["symbol"] == "GOOD"
    assert candidates[0]["roic"] == 15.0
    assert candidates[0]["piotroski_f_score"] == 7


@pytest.mark.asyncio
async def test_underdog_allocation_logic_with_sector_cap(db_session):
    """Test that GET /portfolio/allocate enforces individual 10% and sector 15% diversification rules."""
    # Tech sector peers
    Stock.get_or_create(
        db_session,
        "REF1",
        sector="Technology",
        pe_ratio=Decimal("30.0"),
        analyst_count=10,
        is_active=True,
    )
    Stock.get_or_create(
        db_session,
        "REF2",
        sector="Technology",
        pe_ratio=Decimal("40.0"),
        analyst_count=10,
        is_active=True,
    )

    # Stock 1 in Tech
    s1 = Stock.get_or_create(
        db_session,
        "TECH1",
        company_name="Tech 1",
        sector="Technology",
        pe_ratio=Decimal("10.0"),
        analyst_count=3,
        fcf_growth_yoy=Decimal("0.10"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        roic=Decimal("15.0"),
        piotroski_f_score=7,
    )
    db_session.add(
        PriceCache(
            stock_id=s1.stock_id,
            date=datetime.now(UTC).date(),
            close_price=Decimal("10.00"),
            volume=600000,
        )
    )

    # Stock 2 in Tech
    s2 = Stock.get_or_create(
        db_session,
        "TECH2",
        company_name="Tech 2",
        sector="Technology",
        pe_ratio=Decimal("11.0"),
        analyst_count=3,
        fcf_growth_yoy=Decimal("0.10"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        roic=Decimal("15.0"),
        piotroski_f_score=7,
    )
    db_session.add(
        PriceCache(
            stock_id=s2.stock_id,
            date=datetime.now(UTC).date(),
            close_price=Decimal("10.00"),
            volume=600000,
        )
    )

    # Stock 3 in Tech
    s3 = Stock.get_or_create(
        db_session,
        "TECH3",
        company_name="Tech 3",
        sector="Technology",
        pe_ratio=Decimal("12.0"),
        analyst_count=3,
        fcf_growth_yoy=Decimal("0.10"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        roic=Decimal("15.0"),
        piotroski_f_score=7,
    )
    db_session.add(
        PriceCache(
            stock_id=s3.stock_id,
            date=datetime.now(UTC).date(),
            close_price=Decimal("10.00"),
            volume=600000,
        )
    )

    db_session.commit()

    # Total capital = $1000. Capped at 10% ($100) per stock, and 15% ($150) per sector!
    # Therefore, TECH1 gets $100 (10%). TECH2 can only get $50 (remaining of the $150 sector limit).
    # TECH3 gets $0 because Tech sector allocation has fully hit its $150 (15%) limit!
    response = await get_underdog_portfolio_allocate(capital=1000.0, db=db_session)
    assert response["status"] == "success"
    assert response["total_allocated"] == 1000.0

    allocations = response["portfolio_allocation"]
    tech1_alloc = next(item for item in allocations if item["ticker"] == "TECH1")
    tech2_alloc = next(item for item in allocations if item["ticker"] == "TECH2")
    # Verify TECH3 is not allocated because of the sector cap limit
    assert not any(item["ticker"] == "TECH3" for item in allocations)

    assert tech1_alloc["target_amount"] == 100.0
    assert tech1_alloc["allocation_pct"] == 10.0
    assert tech2_alloc["target_amount"] == 50.0
    assert tech2_alloc["allocation_pct"] == 5.0


@pytest.mark.asyncio
async def test_underdog_allocation_low_risk_swap(db_session):
    """Test that low_risk=True swaps standard index equity ETFs for safe Treasury Bond ETFs."""
    # Tech sector peers
    Stock.get_or_create(
        db_session,
        "REF1",
        sector="Technology",
        pe_ratio=Decimal("30.0"),
        analyst_count=10,
        is_active=True,
    )
    Stock.get_or_create(
        db_session,
        "REF2",
        sector="Technology",
        pe_ratio=Decimal("40.0"),
        analyst_count=10,
        is_active=True,
    )

    # 1. Turnaround Stock
    s1 = Stock.get_or_create(
        db_session,
        "TECH1",
        company_name="Tech 1",
        sector="Technology",
        pe_ratio=Decimal("10.0"),
        analyst_count=3,
        fcf_growth_yoy=Decimal("0.10"),
        short_interest_declining=True,
        insider_buying_6m=True,
        is_active=True,
        roic=Decimal("15.0"),
        piotroski_f_score=7,
    )
    db_session.add(
        PriceCache(
            stock_id=s1.stock_id,
            date=datetime.now(UTC).date(),
            close_price=Decimal("10.00"),
            volume=600000,
        )
    )
    db_session.commit()

    # Call allocation with low_risk=True
    response = await get_underdog_portfolio_allocate(
        capital=1000.0, low_risk=True, db=db_session
    )
    assert response["status"] == "success"

    allocations = response["portfolio_allocation"]

    # Core safety buffer should be allocated to BIL and SHV instead of SPY/QQQ
    bil_alloc = next(item for item in allocations if item["ticker"] == "BIL")
    shv_alloc = next(item for item in allocations if item["ticker"] == "SHV")

    assert bil_alloc["ticker"] == "BIL"
    assert bil_alloc["asset_class"] == "Ultra-Safe Treasury Cash Bond"
    assert shv_alloc["ticker"] == "SHV"
    assert shv_alloc["asset_class"] == "Ultra-Safe Treasury Cash Bond"

    # Confirm no SPY or QQQ allocations are present
    assert not any(item["ticker"] == "SPY" for item in allocations)
    assert not any(item["ticker"] == "QQQ" for item in allocations)


@pytest.mark.asyncio
async def test_underdog_portfolio_monitor_trailing_stop_loss(db_session):
    """Test that trailing stop-loss monitors peak price and triggers a red alert on 15%+ drawdown from peak."""
    portfolio = UserPortfolio(user_id="default", name="My Portfolio")
    db_session.add(portfolio)
    db_session.commit()

    stock = Stock.get_or_create(
        db_session,
        "TURN",
        company_name="Turnaround Corp",
        sector="Technology",
        is_active=True,
    )
    db_session.commit()

    # Initial position bought at $10.00, but has already peaked at $20.00
    pos = PortfolioPosition(
        portfolio_id=portfolio.id,
        ticker="TURN",
        shares=Decimal("10.0"),
        average_cost_basis=Decimal("10.00"),
        total_cost=Decimal("100.00"),
        purchase_date=datetime.now(UTC),
        peak_price=Decimal("20.00"),  # Tracked historic peak
    )
    db_session.add(pos)

    # Current price has dropped to $16.00
    # From cost basis ($10.00), it represents a profit (+60%).
    # From peak price ($20.00), it represents a -20% drawdown -> Trailing stop-loss should breach!
    db_session.add(
        PriceCache(
            stock_id=stock.stock_id,
            date=datetime.now(UTC).date(),
            open_price=Decimal("16.00"),
            high_price=Decimal("16.00"),
            low_price=Decimal("16.00"),
            close_price=Decimal("16.00"),
            volume=600000,
        )
    )
    db_session.commit()

    response = await get_underdog_portfolio_monitor(
        portfolio_name="My Portfolio", db=db_session
    )
    assert response["status"] == "success"
    assert response["alerts_triggered"] == 1
    assert response["alerts"][0]["ticker"] == "TURN"
    assert response["alerts"][0]["drawdown_pct"] == -20.0

    positions = response["positions"]
    turn_pos = next(item for item in positions if item["ticker"] == "TURN")
    assert turn_pos["status"] == "RED ALERT: STOP-LOSS BREACHED"
    assert turn_pos["peak_price"] == 20.0  # Kept historic peak
    assert turn_pos["pnl_pct"] == 60.0  # Overall P&L is still positive!


@pytest.mark.asyncio
async def test_dispatch_push_alert_mocked():
    """Test that dispatch_push_alert properly sends a POST request when a webhook is configured."""
    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200

        # Test with no webhook configured (should skip)
        with mock.patch.dict(os.environ, {}, clear=True):
            res = dispatch_push_alert("TURN", 16.0, 20.0, -20.0, "SELL")
            assert res is False
            mock_post.assert_not_called()

        # Test with webhook configured (should send post)
        with mock.patch.dict(
            os.environ, {"UNDERDOG_ALERT_WEBHOOK_URL": "http://mock-webhook"}
        ):
            res = dispatch_push_alert("TURN", 16.0, 20.0, -20.0, "SELL")
            assert res is True
            mock_post.assert_called_once()
