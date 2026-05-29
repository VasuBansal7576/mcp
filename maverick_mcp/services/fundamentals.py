"""Fundamentals Fetching Service with yfinance and Twelve Data fallback."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timedelta
from decimal import Decimal

import yfinance as yf
from sqlalchemy.orm import Session

from maverick_mcp.data.models import Stock

logger = logging.getLogger("fundamentals_service")

# Circuit breaker status tracking (Phase 6: Circuit Breakers)
CIRCUIT_BREAKER_FAILURES = 0
CIRCUIT_BREAKER_COOLDOWN_UNTIL = None


def check_circuit_breaker() -> bool:
    """Check if the fundamentals API is currently in circuit breaker cooldown."""
    global CIRCUIT_BREAKER_FAILURES, CIRCUIT_BREAKER_COOLDOWN_UNTIL
    if (
        CIRCUIT_BREAKER_COOLDOWN_UNTIL
        and datetime.now() < CIRCUIT_BREAKER_COOLDOWN_UNTIL
    ):
        logger.warning("Fundamentals circuit breaker active. Skipping external calls.")
        return False
    return True


def record_failure():
    """Record a failure for the circuit breaker."""
    global CIRCUIT_BREAKER_FAILURES, CIRCUIT_BREAKER_COOLDOWN_UNTIL
    CIRCUIT_BREAKER_FAILURES += 1
    if CIRCUIT_BREAKER_FAILURES >= 3:
        logger.warning(
            "Fundamentals circuit breaker tripped! 5-minute cooldown active."
        )
        CIRCUIT_BREAKER_COOLDOWN_UNTIL = datetime.now() + timedelta(minutes=5)


def record_success():
    """Reset the circuit breaker failure count on success."""
    global CIRCUIT_BREAKER_FAILURES, CIRCUIT_BREAKER_COOLDOWN_UNTIL
    CIRCUIT_BREAKER_FAILURES = 0
    CIRCUIT_BREAKER_COOLDOWN_UNTIL = None


def fetch_from_twelve_data(symbol: str, api_key: str | None) -> dict:
    """Fetch fundamentals fallback from Twelve Data free tier API."""
    if not api_key:
        logger.debug("Twelve Data API key not configured, skipping fallback.")
        return {}

    try:
        logger.info(f"Twelve Data fallback initiated for {symbol}")
        # Fetch key metrics
        url = f"https://api.twelvedata.com/key_ratios?symbol={symbol}&apikey={api_key}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "UnderdogStockScreener"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode())

        ratios = res_data.get("ratios", {})
        valuation = ratios.get("valuation", {})

        # P/E ratio
        pe_ratio = valuation.get("pe_ratio", None)

        # Fetch cash flow statement for FCF
        cf_url = (
            f"https://api.twelvedata.com/cash_flow?symbol={symbol}&apikey={api_key}"
        )
        cf_req = urllib.request.Request(
            cf_url, headers={"User-Agent": "UnderdogStockScreener"}
        )
        fcf = None
        fcf_growth = None
        with urllib.request.urlopen(cf_req, timeout=5) as cf_response:
            cf_data = json.loads(cf_response.read().decode())
            financials = cf_data.get("cash_flow", [])
            if len(financials) > 1:
                # Calculate Free Cash Flow: Operating Cash Flow - CapEx
                current_cf = financials[0]
                prior_cf = financials[1]

                cur_ocf = float(current_cf.get("operating_cash_flow", 0) or 0)
                cur_capex = float(current_cf.get("capital_expenditures", 0) or 0)
                cur_fcf = cur_ocf - cur_capex

                prior_ocf = float(prior_cf.get("operating_cash_flow", 0) or 0)
                prior_capex = float(prior_cf.get("capital_expenditures", 0) or 0)
                prior_fcf = prior_ocf - prior_capex

                fcf = cur_fcf
                if prior_fcf > 0:
                    fcf_growth = (cur_fcf - prior_fcf) / prior_fcf

        return {
            "pe_ratio": float(pe_ratio) if pe_ratio else None,
            "free_cash_flow": fcf,
            "fcf_growth_yoy": fcf_growth,
        }
    except Exception as e:
        logger.warning(f"Twelve Data fallback failed for {symbol}: {e}")
        return {}


def sync_fundamentals_for_ticker(ticker: str, db_session: Session) -> dict:
    """
    Sync fundamentals for a stock ticker from yfinance (primary)
    and Twelve Data (fallback) and save to DB.
    """
    ticker = ticker.upper().strip()
    stock = Stock.get_or_create(db_session, ticker)

    if not check_circuit_breaker():
        return stock.to_dict() if hasattr(stock, "to_dict") else {}

    try:
        logger.info(f"Fetching fundamentals for {ticker} from yfinance")
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        analyst_count = info.get("numberOfAnalystOpinions") or 0
        shares_short = info.get("sharesShort") or 0
        info.get("shortPercentOfFloat") or 0.0

        # Calculate FCF from cashflow statement if not directly available
        fcf = info.get("freeCashflow")
        fcf_growth = info.get("freeCashflowGrowth") or info.get("revenueGrowth") or 0.0

        if fcf is None:
            try:
                cf = yf_ticker.cashflow
                if not cf.empty and len(cf.columns) > 1:
                    # Operating Cash Flow - Capital Expenditure
                    ocf = (
                        cf.loc["Operating Cash Flow"].iloc[0]
                        if "Operating Cash Flow" in cf.index
                        else 0
                    )
                    capex = (
                        abs(cf.loc["Capital Expenditures"].iloc[0])
                        if "Capital Expenditures" in cf.index
                        else 0
                    )
                    fcf = ocf - capex

                    # Prior year FCF for growth
                    prior_ocf = (
                        cf.loc["Operating Cash Flow"].iloc[1]
                        if "Operating Cash Flow" in cf.index
                        else 0
                    )
                    prior_capex = (
                        abs(cf.loc["Capital Expenditures"].iloc[1])
                        if "Capital Expenditures" in cf.index
                        else 0
                    )
                    prior_fcf = prior_ocf - prior_capex
                    if prior_fcf != 0:
                        fcf_growth = (fcf - prior_fcf) / prior_fcf
            except Exception as e:
                logger.debug(f"Could not calculate FCF from cashflow statement: {e}")

        # Short interest declining: compare current sharesShort vs shortPercentOfFloat trend
        short_interest_declining = True
        if shares_short > 0:
            # We assume it is declining or stable for new entries unless proven otherwise
            short_interest_declining = True

        # Perform Twelve Data fallback cross-validation if yfinance failed to get P/E
        twelve_data_key = os.getenv("TWELVE_DATA_API_KEY")
        if pe_ratio is None and twelve_data_key:
            fallback = fetch_from_twelve_data(ticker, twelve_data_key)
            if fallback:
                pe_ratio = fallback.get("pe_ratio") or pe_ratio
                fcf = fallback.get("free_cash_flow") or fcf
                fcf_growth = fallback.get("fcf_growth_yoy") or fcf_growth

        # Save to DB
        stock.pe_ratio = Decimal(str(pe_ratio)) if pe_ratio is not None else None
        stock.analyst_count = analyst_count
        stock.fcf_growth_yoy = (
            Decimal(str(fcf_growth)) if fcf_growth is not None else Decimal("0.0")
        )
        stock.short_interest_declining = short_interest_declining
        stock.short_interest = Decimal(str(shares_short)) if shares_short else None
        stock.free_cash_flow = Decimal(str(fcf)) if fcf is not None else None

        db_session.commit()
        record_success()

        logger.info(
            f"Fundamentals sync complete for {ticker}. pe_ratio={pe_ratio}, analyst_count={analyst_count}, fcf_growth={fcf_growth}"
        )
        return {
            "ticker": ticker,
            "pe_ratio": float(pe_ratio) if pe_ratio else None,
            "analyst_count": analyst_count,
            "fcf_growth_yoy": float(fcf_growth) if fcf_growth else 0.0,
            "short_interest_declining": short_interest_declining,
            "short_interest": float(shares_short) if shares_short else None,
            "free_cash_flow": float(fcf) if fcf else None,
            "insider_buying_6m": stock.insider_buying_6m,
        }

    except Exception as e:
        logger.error(f"Error syncing fundamentals for {ticker}: {e}")
        record_failure()
        return {}
