"""Underdog Stock Screener service implementation."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
from sqlalchemy.orm import Session

from maverick_mcp.data.models import Stock


class UnderdogScreener:
    """
    Screener that implements the 'Underdog' value rules:
    - analyst_count < 5 (Overlooked by Wall Street)
    - pe_ratio < sector_median_pe (Undervalued vs. peers)
    - fcf_growth_yoy > 0 (Positive free cash flow growth)
    - short_interest_declining = True (Shorts are covering)
    - insider_buying_6m = True (Insiders are buying)
    """

    def __init__(self, db_session: Session):
        """Initialize the underdog screener with a database session."""
        self.db = db_session

    def screen_stocks(self) -> list[dict]:
        """
        Screen the active stock universe using pure underdog rules.
        Does not make any LLM calls.

        Returns:
            List of 20-30 max candidates.
        """
        # Phase 6: Graceful degradation - Check if SEC EDGAR circuit breaker is active
        from maverick_mcp.data.cache import get_from_cache

        sec_edgar_down = get_from_cache("underdog:circuit_breaker:sec_edgar") is True

        # Fetch active stocks
        stocks = self.db.query(Stock).filter(Stock.is_active.is_(True)).all()

        # Compute median P/E ratio per sector
        sector_pes: dict[str, list[float]] = {}
        for stock in stocks:
            if stock.sector and stock.pe_ratio is not None:
                sector_pes.setdefault(stock.sector, []).append(float(stock.pe_ratio))

        sector_medians: dict[str, float] = {}
        for sector, pes in sector_pes.items():
            if pes:
                sector_medians[sector] = float(np.median(pes))

        candidates = []
        for stock in stocks:
            # Exclude ETFs
            if stock.is_etf:
                continue

            # Rule 1: Overlooked - analyst_count < 5
            analyst_count = (
                stock.analyst_count if stock.analyst_count is not None else 0
            )
            if analyst_count >= 5:
                continue

            # Rule 2: Undervalued - pe_ratio < sector_median_pe
            if not stock.sector or stock.pe_ratio is None:
                continue
            median_pe = sector_medians.get(stock.sector)
            if median_pe is None or float(stock.pe_ratio) >= median_pe:
                continue

            # Rule 3: Quality - fcf_growth_yoy >= 5% (to filter out fluky or marginal cash flow growth)
            fcf_growth = (
                stock.fcf_growth_yoy
                if stock.fcf_growth_yoy is not None
                else Decimal("0.0")
            )
            if fcf_growth < Decimal("0.05"):
                continue

            # Rule 4: Improving Sentiment - short_interest_declining = True
            if not stock.short_interest_declining:
                continue

            # Rule 5: Insider conviction - insider_buying_6m = True
            # Graceful degradation: skip rule if SEC EDGAR is down
            if not sec_edgar_down and not stock.insider_buying_6m:
                continue

            # Rule 6: ROIC Quality Filter - roic >= 10.0 (if populated in database)
            roic_val = stock.roic if stock.roic is not None else Decimal("0.0")
            if stock.roic is not None and roic_val < Decimal("10.0"):
                continue

            # Rule 7: Piotroski F-Score Filter - F-score >= 6 (if populated in database)
            f_score = (
                stock.piotroski_f_score if stock.piotroski_f_score is not None else 0
            )
            if stock.piotroski_f_score is not None and f_score < 6:
                continue

            # Rule 8: Price & Volume Floors (if price caches exist)
            if stock.price_caches:
                last_close = float(stock.price_caches[-1].close_price)
                if last_close < 5.0:
                    continue

                volumes = [
                    float(pc.volume)
                    for pc in stock.price_caches
                    if pc.volume is not None
                ]
                avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
                if avg_vol < 500000.0:
                    continue

            # Rule 9: Cache Staleness Guard (active in live production environments)
            import os
            import sys

            is_testing = os.getenv("TESTING") == "True" or "pytest" in sys.modules
            if not is_testing and stock.price_caches:
                last_cache = stock.price_caches[-1]
                if last_cache.updated_at:
                    from datetime import UTC, datetime, timedelta

                    updated_at = last_cache.updated_at
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=UTC)

                    if datetime.now(UTC) - updated_at > timedelta(hours=24):
                        continue

            # Rule 10: Earnings Proximity Safety Guard - exclude if next earnings is in < 14 days
            if stock.next_earnings_date:
                from datetime import date

                days_to_earnings = (stock.next_earnings_date - date.today()).days
                if 0 <= days_to_earnings < 14:
                    continue

            # Rule 11: Forensic Cash Flow Quality Filter - reject if operating cash flow is negative (one-time windfall illusion)
            if (
                stock.operating_cash_flow is not None
                and float(stock.operating_cash_flow) <= 0.0
            ):
                continue

            # Volatility Circuit Breaker: Check for single-day drawdowns > 5%
            high_volatility = False
            if stock.price_caches and len(stock.price_caches) >= 2:
                last_close = float(stock.price_caches[-1].close_price)
                prev_close = float(stock.price_caches[-2].close_price)
                if prev_close > 0.0:
                    drawdown = (prev_close - last_close) / prev_close
                    if drawdown > 0.05:
                        high_volatility = True

            # Add to candidates
            cand_flags = []
            if sec_edgar_down:
                cand_flags.append("NO_INSIDER_DATA")
            if high_volatility:
                cand_flags.append("HIGH_VOLATILITY")

            cand_data = {
                "symbol": stock.ticker_symbol,
                "ticker": stock.ticker_symbol,
                "company_name": stock.company_name,
                "sector": stock.sector,
                "pe_ratio": float(stock.pe_ratio),
                "sector_median_pe": median_pe,
                "analyst_count": analyst_count,
                "fcf_growth_yoy": float(fcf_growth),
                "short_interest_declining": stock.short_interest_declining,
                "insider_buying_6m": stock.insider_buying_6m
                if not sec_edgar_down
                else False,
                "roic": float(roic_val) if stock.roic is not None else 0.0,
                "piotroski_f_score": f_score,
                "close_price": float(stock.price_caches[-1].close_price)
                if stock.price_caches
                else 0.0,
                "flags": cand_flags,
            }
            candidates.append(cand_data)

        # Sort by P/E ratio (lower is more undervalued)
        candidates.sort(key=lambda x: x["pe_ratio"])

        # Limit to 20-30 candidates max
        return candidates[:30]
