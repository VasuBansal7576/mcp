#!/usr/bin/env python3
"""
SEC EDGAR Form 4 tool.
Fetches insider buying from SEC EDGAR API (free, official) for the last 6 months
and stores it in the insider_trades table.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from decimal import Decimal

from sqlalchemy.orm import Session

from maverick_mcp.data.cache import get_from_cache, save_to_cache
from maverick_mcp.data.models import InsiderTrade, SessionLocal, Stock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sec_edgar_form4")

# Required User-Agent by SEC
HEADERS = {"User-Agent": "UnderdogStockScreener founder@underdog.com"}

SEC_EDGAR_FAILURES_KEY = "underdog:failures:sec_edgar"
SEC_EDGAR_BREAKER_KEY = "underdog:circuit_breaker:sec_edgar"


def check_sec_edgar_circuit_breaker() -> bool:
    """Check if the SEC EDGAR circuit breaker is active."""
    if get_from_cache(SEC_EDGAR_BREAKER_KEY) is True:
        logger.warning("SEC EDGAR circuit breaker is active. Skipping external calls.")
        return False
    return True


def record_sec_edgar_failure():
    """Record a failure for the SEC EDGAR circuit breaker."""
    failures = get_from_cache(SEC_EDGAR_FAILURES_KEY) or 0
    failures += 1
    logger.info(f"SEC EDGAR failure recorded. Total consecutive failures: {failures}/3")
    if failures >= 3:
        logger.warning("SEC EDGAR circuit breaker tripped! 5-minute cooldown active.")
        save_to_cache(SEC_EDGAR_BREAKER_KEY, True, ttl=300)
        save_to_cache(SEC_EDGAR_FAILURES_KEY, 0, ttl=300)
    else:
        save_to_cache(SEC_EDGAR_FAILURES_KEY, failures, ttl=300)


def record_sec_edgar_success():
    """Reset failure count and clear circuit breaker on success."""
    save_to_cache(SEC_EDGAR_FAILURES_KEY, 0, ttl=300)
    save_to_cache(SEC_EDGAR_BREAKER_KEY, False, ttl=300)


def get_cik_for_ticker(ticker: str) -> str | None:
    """Resolve stock ticker symbol to padded 10-digit SEC CIK."""
    if not check_sec_edgar_circuit_breaker():
        return None
    ticker = ticker.upper().strip()
    url = "https://www.sec.gov/files/company_tickers.json"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            for entry in data.values():
                if entry["ticker"] == ticker:
                    return str(entry["cik_str"]).zfill(10)
    except Exception as e:
        logger.error(f"Error fetching CIK mapping for {ticker}: {e}")
        record_sec_edgar_failure()
    return None


def fetch_and_parse_form4(cik: str, accession_no: str, doc_name: str) -> list[dict]:
    """Download and parse a Form 4 XML filing for insider transactions."""
    accession_clean = accession_no.replace("-", "")
    raw_doc = doc_name.split("/")[-1]
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{raw_doc}"
    req = urllib.request.Request(url, headers=HEADERS)

    try:
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)

        # 1. Parse Reporting Owner Details
        reporting_owner = root.find(".//reportingOwner")
        filer_name = "Unknown"
        relation = "Unknown"

        if reporting_owner is not None:
            name_node = reporting_owner.find(".//rptOwnerName")
            if name_node is not None:
                filer_name = name_node.text

            relationship = reporting_owner.find(".//reportingOwnerRelationship")
            if relationship is not None:
                relations = []
                if relationship.find("isDirector") is not None and relationship.find(
                    "isDirector"
                ).text in ("true", "1"):
                    relations.append("Director")
                if relationship.find("isOfficer") is not None and relationship.find(
                    "isOfficer"
                ).text in ("true", "1"):
                    title = relationship.find("officerTitle")
                    title_text = title.text if title is not None else "Officer"
                    relations.append(title_text)
                if relationship.find(
                    "isTenPercentOwner"
                ) is not None and relationship.find("isTenPercentOwner").text in (
                    "true",
                    "1",
                ):
                    relations.append("10% Owner")
                relation = ", ".join(relations) if relations else "Insider"

        # 2. Parse Transactions
        transactions = []
        # Form 4 XML stores transactions in nonDerivativeTransaction elements
        for trans_node in root.findall(".//nonDerivativeTransaction"):
            # Check transaction code
            trans_node.find(".//transactionCoding/transactionCode")

            # Check transaction date
            date_node = trans_node.find(".//transactionDate/value")
            trans_date = date_node.text if date_node is not None else None

            # Check shares
            shares_node = trans_node.find(
                ".//transactionAmounts/transactionShares/value"
            )
            shares = int(float(shares_node.text)) if shares_node is not None else 0

            # Check price per share
            price_node = trans_node.find(
                ".//transactionAmounts/transactionPricePerShare/value"
            )
            price = (
                Decimal(price_node.text) if price_node is not None else Decimal("0.0")
            )

            # Check acquired/disposed code: 'A' = Acquired (Buy), 'D' = Disposed (Sell)
            ad_node = trans_node.find(
                ".//transactionAmounts/transactionAcquiredDisposedCode/value"
            )
            ad_code = ad_node.text if ad_node is not None else "D"

            # We are interested in purchases (Acquired / Code P)
            # Standard code 'P' is Open Market Purchase
            transaction_type = "Buy" if ad_code == "A" else "Sell"

            if trans_date and shares > 0:
                total_value = Decimal(shares) * price
                transactions.append(
                    {
                        "filer_name": filer_name,
                        "relation": relation,
                        "transaction_date": datetime.strptime(
                            trans_date, "%Y-%m-%d"
                        ).date(),
                        "transaction_type": transaction_type,
                        "shares": shares,
                        "price": price,
                        "total_value": total_value,
                        "filing_url": url,
                    }
                )
        return transactions
    except Exception as e:
        logger.error(f"Error parsing Form 4 filing {accession_no}: {e}")
        return []


def fetch_insider_trades(
    ticker: str, db_session: Session, limit_months: int = 6
) -> list[dict]:
    """Fetch recent Form 4 insider transactions for a stock from SEC EDGAR."""
    if not check_sec_edgar_circuit_breaker():
        return []
    ticker = ticker.upper().strip()
    stock = Stock.get_or_create(db_session, ticker)

    cik = get_cik_for_ticker(ticker)
    if not cik:
        logger.warning(f"No CIK found for ticker {ticker}")
        return []

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    req = urllib.request.Request(url, headers=HEADERS)

    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())

        filings = data.get("filings", {}).get("recent", {})
        if not filings:
            logger.warning(f"No filings found for CIK {cik}")
            return []

        cutoff_date = date.today() - timedelta(days=limit_months * 30)

        # Lists of filing metadata
        forms = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])
        accession_numbers = filings.get("accessionNumber", [])
        primary_documents = filings.get("primaryDocument", [])

        insider_buys_count = 0
        has_insider_buys_6m = False

        for idx in range(len(forms)):
            if forms[idx] == "4":
                f_date = datetime.strptime(filing_dates[idx], "%Y-%m-%d").date()
                if f_date >= cutoff_date:
                    acc_no = accession_numbers[idx]
                    doc_name = primary_documents[idx]

                    # Parse filing
                    trades = fetch_and_parse_form4(cik, acc_no, doc_name)

                    for trade in trades:
                        # Store in database if not already exists
                        existing = (
                            db_session.query(InsiderTrade)
                            .filter_by(
                                stock_id=stock.stock_id,
                                filer_name=trade["filer_name"],
                                transaction_date=trade["transaction_date"],
                                shares=trade["shares"],
                                price=trade["price"],
                            )
                            .first()
                        )

                        if not existing:
                            insider_trade = InsiderTrade(
                                stock_id=stock.stock_id,
                                ticker=ticker,
                                filer_name=trade["filer_name"],
                                relation=trade["relation"],
                                transaction_date=trade["transaction_date"],
                                transaction_type=trade["transaction_type"],
                                shares=trade["shares"],
                                price=trade["price"],
                                total_value=trade["total_value"],
                                filing_url=trade["filing_url"],
                            )
                            db_session.add(insider_trade)

                        if trade["transaction_type"] == "Buy":
                            insider_buys_count += 1
                            has_insider_buys_6m = True

        # Update stock metadata
        stock.insider_buying_6m = has_insider_buys_6m
        db_session.commit()

        record_sec_edgar_success()
        logger.info(
            f"Filing fetch completed for {ticker}. Found {insider_buys_count} insider buys in last 6 months. insider_buying_6m={has_insider_buys_6m}"
        )

        # Return trades
        trades_query = (
            db_session.query(InsiderTrade)
            .filter(
                InsiderTrade.stock_id == stock.stock_id,
                InsiderTrade.transaction_date >= cutoff_date,
            )
            .order_by(InsiderTrade.transaction_date.desc())
            .all()
        )

        return [t.to_dict() for t in trades_query]

    except Exception as e:
        logger.error(f"Error fetching filings from SEC EDGAR for CIK {cik}: {e}")
        record_sec_edgar_failure()
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Form 4 insider trading from SEC EDGAR"
    )
    parser.add_argument("ticker", type=str, help="Stock ticker symbol")
    args = parser.parse_args()

    with SessionLocal() as session:
        results = fetch_insider_trades(args.ticker, session)
        print(json.dumps(results, indent=2, default=str))
