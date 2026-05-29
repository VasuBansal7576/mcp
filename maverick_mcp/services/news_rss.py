"""RSS News Ingestion Service using feedparser."""

from __future__ import annotations

import email.utils
import logging
import time
from datetime import UTC, datetime

import feedparser
from sqlalchemy.orm import Session

from maverick_mcp.data.models import Article, Stock

logger = logging.getLogger("news_rss")


def parse_rss_date(date_str: str | None) -> datetime:
    """Safely parse RSS publish date into UTC datetime."""
    if not date_str:
        return datetime.now(UTC)
    try:
        # feedparser parsed time tuple
        return datetime.fromtimestamp(
            time.mktime(email.utils.parsedate(date_str)), tz=UTC
        )
    except Exception:
        try:
            # Fallback standard ISO formats
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return datetime.now(UTC)


def fetch_rss_news_for_ticker(
    ticker: str, db_session: Session, limit: int = 5
) -> list[dict]:
    """
    Fetch news from free RSS feeds (Seeking Alpha, MarketWatch, Benzinga)
    and ingest them into the mcp_articles table.

    No scraping, no API keys, zero cost.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)
        db_session: Database session
        limit: Max articles to return
    """
    ticker = ticker.upper().strip()
    stock = db_session.query(Stock).filter_by(ticker_symbol=ticker).first()
    stock_id = stock.stock_id if stock else None

    # Construct ticker-specific and general RSS endpoints
    feeds = {
        "Seeking Alpha": f"https://seekingalpha.com/api/v3/symbols/{ticker}/rss.xml",
        "MarketWatch": "https://feeds.a.dj.com/rss/RSSMarketwatch.xml",
        "Benzinga": "https://www.benzinga.com/rss.xml",
    }

    headers = {"User-Agent": "UnderdogStockScreener founder@underdog.com"}
    articles_added = 0

    for source_name, feed_url in feeds.items():
        try:
            logger.info(f"Fetching RSS feed from {source_name} for {ticker}")

            # Using feedparser with custom headers or simply parser
            # In feedparser, we can pass agent or download using urllib first
            # Since feedparser.parse can sometimes get 403, downloading first is safer!
            import urllib.request

            req = urllib.request.Request(feed_url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                feed_content = response.read()

            parsed = feedparser.parse(feed_content)

            for entry in parsed.entries[:10]:
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                link = entry.get("link", "")
                pub_date_str = entry.get("published", entry.get("pubDate", None))
                published_date = parse_rss_date(pub_date_str)

                # Filter: for ticker-specific feed, accept all.
                # For general feeds, verify ticker is mentioned in title or summary.
                is_relevant = True
                if "symbols" not in feed_url:  # General feeds
                    # Search for "$AAPL", "AAPL", or stock company name
                    keywords = [f"${ticker}", f" {ticker} ", f"({ticker})", ticker]
                    if stock and stock.company_name:
                        keywords.append(stock.company_name)
                    is_relevant = any(
                        kw.lower() in title.lower() or kw.lower() in summary.lower()
                        for kw in keywords
                    )

                if is_relevant:
                    # Check if article already exists in DB (to prevent duplicates)
                    existing = (
                        db_session.query(Article)
                        .filter_by(
                            ticker=ticker,
                            title=title[:400],  # Compare first part of title
                        )
                        .first()
                    )

                    if not existing:
                        article = Article(
                            stock_id=stock_id,
                            ticker=ticker,
                            title=title,
                            summary=summary,
                            link=link,
                            published_date=published_date,
                            source=source_name,
                        )
                        db_session.add(article)
                        articles_added += 1

            db_session.commit()
        except Exception as e:
            logger.warning(
                f"Error fetching RSS news from {source_name} for {ticker}: {e}"
            )

    # Return top parsed articles from DB
    articles = (
        db_session.query(Article)
        .filter_by(ticker=ticker)
        .order_by(Article.published_date.desc())
        .limit(limit)
        .all()
    )

    return [a.to_dict() for a in articles]
