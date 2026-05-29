"""Unit tests for the SEC EDGAR Form 4 fetching tool."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.data.models import InsiderTrade, Stock
from maverick_mcp.database.base import Base
from tools.sec_edgar_form4 import (
    check_sec_edgar_circuit_breaker,
    fetch_insider_trades,
    get_cik_for_ticker,
    record_sec_edgar_failure,
    record_sec_edgar_success,
)


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
def test_get_cik_for_ticker(mock_urlopen):
    """Test CIK resolution for ticker."""
    mock_response = MagicMock()
    mock_response.__enter__.return_value.read.return_value = (
        b'{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}'
    )
    mock_urlopen.return_value = mock_response

    cik = get_cik_for_ticker("AAPL")
    assert cik == "0000320193"


@patch("urllib.request.urlopen")
def test_fetch_insider_trades_success(mock_urlopen, db_session):
    """Test fetching and parsing insider trades successfully."""
    # 1. Mock CIK lookup response
    mock_response_cik = MagicMock()
    mock_response_cik.__enter__.return_value.read.return_value = (
        b'{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}'
    )

    # 2. Mock submissions JSON response
    submissions_json = {
        "filings": {
            "recent": {
                "form": ["4", "10-Q"],
                "filingDate": ["2026-05-10", "2026-05-09"],
                "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002"],
                "primaryDocument": ["form4.xml", "10q.htm"],
            }
        }
    }
    mock_response_submissions = MagicMock()
    mock_response_submissions.__enter__.return_value.read.return_value = bytes(
        json.dumps(submissions_json), encoding="utf-8"
    )

    # 3. Mock Form 4 XML response
    form4_xml = """<?xml version="1.0"?>
    <ownershipDocument>
        <reportingOwner>
            <reportingOwnerId>
                <rptOwnerName>Cook Tim</rptOwnerName>
            </reportingOwnerId>
            <reportingOwnerRelationship>
                <isDirector>true</isDirector>
                <isOfficer>true</isOfficer>
                <officerTitle>CEO</officerTitle>
            </reportingOwnerRelationship>
        </reportingOwner>
        <nonDerivativeTable>
            <nonDerivativeTransaction>
                <transactionCoding>
                    <transactionCode>P</transactionCode>
                </transactionCoding>
                <transactionDate>
                    <value>2026-05-08</value>
                </transactionDate>
                <transactionAmounts>
                    <transactionShares>
                        <value>10000</value>
                    </transactionShares>
                    <transactionPricePerShare>
                        <value>150.00</value>
                    </transactionPricePerShare>
                    <transactionAcquiredDisposedCode>
                        <value>A</value>
                    </transactionAcquiredDisposedCode>
                </transactionAmounts>
            </nonDerivativeTransaction>
        </nonDerivativeTable>
    </ownershipDocument>
    """
    mock_response_xml = MagicMock()
    mock_response_xml.__enter__.return_value.read.return_value = bytes(
        form4_xml, encoding="utf-8"
    )

    # Wire side effects for urlopen
    mock_urlopen.side_effect = [
        mock_response_cik,  # mapping inside fetch_insider_trades -> get_cik_for_ticker
        mock_response_submissions,  # submissions json
        mock_response_xml,  # form4 xml
    ]

    # Invoke
    fetch_insider_trades("AAPL", db_session, limit_months=6)

    # Verify db state
    stock = db_session.query(Stock).filter_by(ticker_symbol="AAPL").first()
    assert stock is not None
    assert stock.insider_buying_6m is True

    db_trades = db_session.query(InsiderTrade).all()
    assert len(db_trades) == 1
    assert db_trades[0].filer_name == "Cook Tim"
    assert db_trades[0].relation == "Director, CEO"
    assert db_trades[0].transaction_type == "Buy"
    assert db_trades[0].shares == 10000
    assert db_trades[0].price == Decimal("150.00")
    assert db_trades[0].total_value == Decimal("1500000.0")


def test_sec_edgar_circuit_breaker(db_session):
    """Test that the SEC EDGAR circuit breaker trips and resets correctly."""
    # Reset count to ensure clean state
    record_sec_edgar_success()
    assert check_sec_edgar_circuit_breaker() is True

    # Record 3 failures
    record_sec_edgar_failure()
    record_sec_edgar_failure()
    record_sec_edgar_failure()

    # Circuit breaker should now be active
    assert check_sec_edgar_circuit_breaker() is False

    # fetch_insider_trades should fail immediately with empty list
    trades = fetch_insider_trades("MSFT", db_session)
    assert trades == []

    # Reset
    record_sec_edgar_success()
    assert check_sec_edgar_circuit_breaker() is True
