"""Underdog Stock Screener Agent.

Implements a provider-agnostic, cached, 3-node reasoning agent:
Node 1: News Sentiment - summarizes last 5 articles and scores from -1.0 to 1.0.
Node 2: Filing Risk - scans latest risk indicators and flags LOW/MED/HIGH.
Node 3: Synthesizer - combines signals into a final recommendation and confidence.
"""

import hashlib
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy.orm import Session

from maverick_mcp.data.cache import get_from_cache, save_to_cache
from maverick_mcp.data.models import Article, InsiderTrade, Stock
from maverick_mcp.providers.cost_tracking import get_global_cost_accumulator
from maverick_mcp.providers.llm_factory import get_llm

logger = logging.getLogger("maverick_mcp.agents.underdog_agent")


def get_active_provider_rates() -> tuple[float, float, str]:
    """Returns (cost_per_million_input, cost_per_million_output, model_name)."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    model = os.getenv("LLM_MODEL", "").lower().strip()

    if provider == "deepseek":
        return 0.27, 1.10, model or "deepseek-chat"
    elif provider == "openai":
        if "gpt-4o" in model and "mini" not in model:
            return 2.50, 10.00, model or "gpt-4o"
        return 0.15, 0.60, model or "gpt-4o-mini"
    elif provider == "anthropic":
        if "sonnet" in model:
            return 3.00, 15.00, model or "claude-3-sonnet"
        elif "opus" in model:
            return 15.00, 75.00, model or "claude-3-opus"
        return 0.25, 1.25, model or "claude-3-haiku"
    elif provider == "openrouter":
        if "deepseek" in model:
            return 0.27, 1.10, model
        elif "gemini" in model:
            return 0.075, 0.30, model
        return 0.15, 0.60, model or "google/gemini-2.5-flash"
    else:
        return 0.15, 0.60, model or "gpt-4o-mini"


def extract_token_usage(response: Any) -> tuple[int, int]:
    """Extract input and output tokens from a LangChain message response."""
    input_tokens = 0
    output_tokens = 0

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = response.usage_metadata.get("input_tokens", 0)
        output_tokens = response.usage_metadata.get("output_tokens", 0)
    elif hasattr(response, "response_metadata") and response.response_metadata:
        token_usage = response.response_metadata.get("token_usage", {})
        if token_usage:
            input_tokens = token_usage.get(
                "prompt_tokens", token_usage.get("input_tokens", 0)
            )
            output_tokens = token_usage.get(
                "completion_tokens", token_usage.get("output_tokens", 0)
            )

    # Fallback/mock support for testing
    if output_tokens == 0 and hasattr(response, "content") and response.content:
        output_tokens = len(response.content) // 4

    return input_tokens, output_tokens


def record_llm_call(model_name: str, input_tokens: int, output_tokens: int) -> None:
    """Record LLM call cost to global CostAccumulator."""
    in_rate, out_rate, _ = get_active_provider_rates()
    accumulator = get_global_cost_accumulator()

    import asyncio

    try:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                accumulator.record_cost(
                    model_id=model_name,
                    cost_per_million_input=in_rate,
                    cost_per_million_output=out_rate,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
        except RuntimeError:
            # No running event loop
            asyncio.run(
                accumulator.record_cost(
                    model_id=model_name,
                    cost_per_million_input=in_rate,
                    cost_per_million_output=out_rate,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
    except Exception as e:
        logger.warning(f"Failed to record LLM cost: {e}")


def calculate_fingerprint(
    ticker: str,
    last_price_update: str,
    last_article_date: str,
    last_filing_date: str,
    model_name: str,
) -> str:
    """Calculate MD5 fingerprint hash for caching."""
    raw = f"{ticker.upper()}:{last_price_update}:{last_article_date}:{last_filing_date}:{model_name}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class UnderdogAgent:
    """Orchestrates the 3-node agent workflow for Underdog Stocks."""

    def __init__(self, db: Session):
        self.db = db
        self.llm = get_llm()

    def run_sentiment_node(
        self, ticker: str, articles: list[Article]
    ) -> tuple[float, str]:
        """Node 1: Summarize articles and score sentiment [-1.0, 1.0]."""
        if not articles:
            return 0.0, "No recent news articles available for analysis."

        articles_text = "\n\n".join(
            f"Title: {a.title}\nSource: {a.source}\nDate: {a.published_date}\nSummary: {a.summary or 'N/A'}"
            for a in articles[:5]
        )

        prompt = f"""You are an elite financial news analyst. Analyze the following articles for {ticker} and score the overall sentiment.
Return a valid JSON object with the keys: "sentiment_score" (a float strictly between -1.0 and 1.0) and "sentiment_summary" (a 1-2 sentence explanation).

Articles:
{articles_text}

JSON response:"""

        try:
            _, _, model_name = get_active_provider_rates()
            response = self.llm.invoke(prompt)
            in_t, out_t = extract_token_usage(response)
            record_llm_call(model_name, in_t, out_t)

            content = response.content.strip()
            # Clean possible markdown block formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)
            score = float(data.get("sentiment_score", 0.0))
            summary = str(data.get("sentiment_summary", "Analyzed recent news."))
            return max(-1.0, min(1.0, score)), summary
        except Exception as e:
            logger.error(f"Error in sentiment node for {ticker}: {e}")
            return 0.0, f"Error analyzing news sentiment: {str(e)}"

    def run_risk_node(self, ticker: str, stock: Stock) -> tuple[str, str]:
        """Node 2: Evaluate filing risk as LOW/MED/HIGH."""
        # Retrieve filing/description info
        desc = stock.description or "No company description available."
        prompt = f"""You are a risk management specialist. Evaluate the operational and filing risk for {ticker} based on this company overview.
Analyze potential warning flags and assign a risk rating of "LOW", "MED", or "HIGH".
Return a valid JSON object with keys: "risk_level" (string: "LOW", "MED", or "HIGH") and "risk_summary" (a 1-2 sentence explanation of main risks).

Company Overview:
{desc}

JSON response:"""

        try:
            _, _, model_name = get_active_provider_rates()
            response = self.llm.invoke(prompt)
            in_t, out_t = extract_token_usage(response)
            record_llm_call(model_name, in_t, out_t)

            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)
            risk_level = str(data.get("risk_level", "MED")).upper().strip()
            if risk_level not in ["LOW", "MED", "HIGH"]:
                risk_level = "MED"
            summary = str(data.get("risk_summary", "Filing risk evaluated."))
            return risk_level, summary
        except Exception as e:
            logger.error(f"Error in risk node for {ticker}: {e}")
            return "MED", f"Error analyzing filing risk: {str(e)}"

    def run_synthesizer_node(
        self,
        stock: Stock,
        sentiment_score: float,
        sentiment_summary: str,
        risk_level: str,
        risk_summary: str,
    ) -> tuple[str, int, str]:
        """Node 3: Synthesize sentiment, risk, and fundamentals into a recommendation."""
        mcap = f"${stock.market_cap:,}" if stock.market_cap is not None else "N/A"
        pe = f"{float(stock.pe_ratio):.2f}" if stock.pe_ratio is not None else "N/A"
        fcf_growth = (
            f"{float(stock.fcf_growth_yoy):.2%}"
            if stock.fcf_growth_yoy is not None
            else "N/A"
        )
        si = (
            f"{float(stock.short_interest):,}"
            if stock.short_interest is not None
            else "N/A"
        )
        roic_txt = f"{float(stock.roic):.2f}%" if stock.roic is not None else "N/A"
        f_score_txt = (
            str(stock.piotroski_f_score)
            if stock.piotroski_f_score is not None
            else "N/A"
        )

        fundamentals_text = f"""Ticker: {stock.ticker_symbol}
Company Name: {stock.company_name}
Sector: {stock.sector} / Industry: {stock.industry}
Market Cap: {mcap}
P/E Ratio: {pe}
FCF Growth YoY: {fcf_growth}
ROIC: {roic_txt}
Piotroski F-Score: {f_score_txt}
Short Interest: {si}
Insider Buying (last 6 months): {"Yes" if stock.insider_buying_6m else "No"}"""

        prompt = f"""You are the Chief Investment Officer. Combine the fundamental metrics, sentiment indicators, and filing risks to make a final stock recommendation.
Return a valid JSON object with keys:
"recommendation" (string: "BUY", "HOLD", or "SELL")
"confidence" (integer strictly between 0 and 100)
"reasoning" (detailed reasoning trace listing specific signal sources)

Stock Fundamentals:
{fundamentals_text}

News Sentiment Score: {sentiment_score} (Explanation: {sentiment_summary})
Filing Risk Level: {risk_level} (Explanation: {risk_summary})

JSON response:"""

        try:
            _, _, model_name = get_active_provider_rates()
            response = self.llm.invoke(prompt)
            in_t, out_t = extract_token_usage(response)
            record_llm_call(model_name, in_t, out_t)

            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)
            rec = str(data.get("recommendation", "HOLD")).upper().strip()
            if rec not in ["BUY", "HOLD", "SELL"]:
                rec = "HOLD"
            conf = int(data.get("confidence", 50))
            reasoning = str(data.get("reasoning", "Synthesized stock data."))
            return rec, max(0, min(100, conf)), reasoning
        except Exception as e:
            logger.error(f"Error in synthesizer node for {stock.ticker_symbol}: {e}")
            return "HOLD", 50, f"Error synthesizing recommendation: {str(e)}"

    def analyze_ticker(self, ticker: str) -> dict[str, Any]:
        """Runs the complete 3-node agent workflow with cache checking & staleness guards."""
        ticker = ticker.upper().strip()
        stock = self.db.query(Stock).filter(Stock.ticker_symbol == ticker).first()
        if not stock:
            return {
                "ticker": ticker,
                "status": "error",
                "message": "Stock not found in database.",
            }

        # 1. Fetch metadata for fingerprint cache
        last_price_update = stock.updated_at.isoformat() if stock.updated_at else "N/A"

        # Fetch last 5 articles
        articles = (
            self.db.query(Article)
            .filter(Article.ticker == ticker)
            .order_by(Article.published_date.desc())
            .limit(5)
            .all()
        )
        last_article_date = (
            articles[0].published_date.isoformat() if articles else "no_articles"
        )

        # Fetch latest insider trade
        latest_insider = (
            self.db.query(InsiderTrade)
            .filter(InsiderTrade.ticker == ticker)
            .order_by(InsiderTrade.transaction_date.desc())
            .first()
        )
        last_filing_date = (
            latest_insider.transaction_date.isoformat()
            if latest_insider
            else "no_insider_trades"
        )

        _, _, model_name = get_active_provider_rates()

        # Generate cache key fingerprint
        fingerprint = calculate_fingerprint(
            ticker, last_price_update, last_article_date, last_filing_date, model_name
        )
        cache_key = f"underdog:cache:{fingerprint}"

        # Try to retrieve from Redis
        cached_result = get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Redis Fingerprint Cache Hit for {ticker}!")
            return cast(dict[str, Any], cached_result)

        logger.info(f"Redis Fingerprint Cache Miss for {ticker}. Running LLM Agent...")

        # 2. Check staleness (> 24 hours)
        is_stale = False
        staleness_reasons = []

        now = datetime.now(UTC)
        if stock.updated_at and (
            now - stock.updated_at.replace(tzinfo=UTC)
        ) > timedelta(hours=24):
            is_stale = True
            staleness_reasons.append("Fundamentals updated > 24 hours ago")

        # 3. Execute 3-Node Workflow
        sentiment_score, sentiment_summary = self.run_sentiment_node(ticker, articles)
        risk_level, risk_summary = self.run_risk_node(ticker, stock)
        recommendation, confidence, reasoning = self.run_synthesizer_node(
            stock, sentiment_score, sentiment_summary, risk_level, risk_summary
        )

        # 4. Apply Staleness Guard and Graceful Degradation
        flags = []
        if is_stale:
            confidence = int(confidence * 0.5)
            flags.append("STALE_DATA")
            reasoning = f"[STALE_DATA: {', '.join(staleness_reasons)}] " + reasoning

        # Graceful degradation flag if SEC EDGAR is down
        sec_edgar_down = get_from_cache("underdog:circuit_breaker:sec_edgar") is True
        if sec_edgar_down:
            flags.append("NO_INSIDER_DATA")
            reasoning = "[NO_INSIDER_DATA: SEC EDGAR is down] " + reasoning

        result = {
            "ticker": ticker,
            "company_name": stock.company_name,
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": reasoning,
            "sentiment_score": sentiment_score,
            "sentiment_summary": sentiment_summary,
            "risk_level": risk_level,
            "risk_summary": risk_summary,
            "flags": flags,
            "stale_data": is_stale,
            "model_used": model_name,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # 5. Save to Cache (24 hours TTL)
        save_to_cache(cache_key, result, ttl=86400)

        return result
