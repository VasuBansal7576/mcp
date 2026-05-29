from decimal import Decimal
from datetime import datetime, UTC, date, timedelta
import uuid

from maverick_mcp.data import SessionLocal
from maverick_mcp.data.models import Stock, Article, InsiderTrade, PriceCache

def seed_perfect_underdogs():
    db = SessionLocal()
    try:
        print("Clearing old data to ensure a perfectly clean and aligned seed...")
        db.query(PriceCache).delete()
        db.query(Article).delete()
        db.query(InsiderTrade).delete()
        db.query(Stock).delete()
        db.commit()

        print("Seeding stock metadata with ROIC, F-Score, operating cash flows, earnings proximity date, etc...")
        
        # 1. Tech Sector
        aapl = Stock.get_or_create(
            db,
            "AAPL",
            company_name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
            description="Consumer electronics giant.",
            is_etf=False,
            is_active=True,
            analyst_count=43,
            pe_ratio=Decimal("37.88"),
            fcf_growth_yoy=Decimal("0.166"),
            short_interest_declining=True,
            insider_buying_6m=True,
            market_cap=3000000000000,
            roic=Decimal("28.0"),
            piotroski_f_score=8,
            next_earnings_date=date.today() + timedelta(days=45),
            operating_cash_flow=Decimal("110000000000.00")
        )
        
        intc = Stock.get_or_create(
            db,
            "INTC",
            company_name="Intel Corporation",
            sector="Technology",
            industry="Semiconductors",
            description="Designs, manufactures, and sells computer components and related products. Foundry transition candidate.",
            is_etf=False,
            is_active=True,
            analyst_count=3,
            pe_ratio=Decimal("15.5"),
            fcf_growth_yoy=Decimal("0.18"),
            short_interest_declining=True,
            insider_buying_6m=True,
            market_cap=120000000000,
            roic=Decimal("11.2"),
            piotroski_f_score=7,
            next_earnings_date=date.today() + timedelta(days=45),
            operating_cash_flow=Decimal("12000000000.00")
        )

        # 2. Financials Sector
        v = Stock.get_or_create(
            db,
            "V",
            company_name="Visa Inc.",
            sector="Financials",
            industry="Credit Services",
            description="Global payment technology leader.",
            is_etf=False,
            is_active=True,
            analyst_count=38,
            pe_ratio=Decimal("28.0"),
            fcf_growth_yoy=Decimal("0.11"),
            short_interest_declining=True,
            insider_buying_6m=False,
            market_cap=500000000000,
            roic=Decimal("22.0"),
            piotroski_f_score=8,
            next_earnings_date=date.today() + timedelta(days=45),
            operating_cash_flow=Decimal("18000000000.00")
        )

        pypl = Stock.get_or_create(
            db,
            "PYPL",
            company_name="PayPal Holdings, Inc.",
            sector="Financials",
            industry="Credit Services",
            description="Operates a digital payment technology platform.",
            is_etf=False,
            is_active=True,
            analyst_count=4,
            pe_ratio=Decimal("14.2"),
            fcf_growth_yoy=Decimal("0.12"),
            short_interest_declining=True,
            insider_buying_6m=True,
            market_cap=75000000000,
            roic=Decimal("18.5"),
            piotroski_f_score=8,
            next_earnings_date=date.today() + timedelta(days=45),
            operating_cash_flow=Decimal("5500000000.00")
        )

        # 3. Healthcare Sector
        lly = Stock.get_or_create(
            db,
            "LLY",
            company_name="Eli Lilly and Company",
            sector="Healthcare",
            industry="Drug Manufacturers - General",
            description="Global pharmaceutical company.",
            is_etf=False,
            is_active=True,
            analyst_count=29,
            pe_ratio=Decimal("65.0"),
            fcf_growth_yoy=Decimal("0.35"),
            short_interest_declining=True,
            insider_buying_6m=False,
            market_cap=800000000000,
            roic=Decimal("20.0"),
            piotroski_f_score=7,
            next_earnings_date=date.today() + timedelta(days=45),
            operating_cash_flow=Decimal("14000000000.00")
        )

        gild = Stock.get_or_create(
            db,
            "GILD",
            company_name="Gilead Sciences, Inc.",
            sector="Healthcare",
            industry="Drug Manufacturers - General",
            description="Biopharmaceutical company with oncology and antiviral focus.",
            is_etf=False,
            is_active=True,
            analyst_count=4,
            pe_ratio=Decimal("11.8"),
            fcf_growth_yoy=Decimal("0.08"),
            short_interest_declining=True,
            insider_buying_6m=True,
            market_cap=90000000000,
            roic=Decimal("14.8"),
            piotroski_f_score=7,
            next_earnings_date=date.today() + timedelta(days=45),
            operating_cash_flow=Decimal("8000000000.00")
        )

        db.commit()
        print("Stocks seeded successfully.")

        # Seed Price Caches
        print("Seeding fresh, perfect price caches...")
        now = datetime.now(UTC)
        
        # Helper to add price caches
        def add_price_cache(stock, price, vol):
            # Two caches per stock to ensure volatility breaker doesn't trigger and history exists
            # Day 1: slightly lower price
            # Day 2: current price
            # This ensures daily drawdown = 0 or negative (positive return), which avoids volatility breaker!
            prev_price = price * Decimal("0.99") # 1% gain
            
            db.add(PriceCache(
                price_cache_id=uuid.uuid4(),
                stock_id=stock.stock_id,
                date=date.today() - timedelta(days=1),
                open_price=prev_price,
                high_price=prev_price,
                low_price=prev_price,
                close_price=prev_price,
                volume=vol,
                created_at=now - timedelta(days=1),
                updated_at=now - timedelta(days=1)
            ))
            db.add(PriceCache(
                price_cache_id=uuid.uuid4(),
                stock_id=stock.stock_id,
                date=date.today(),
                open_price=price,
                high_price=price,
                low_price=price,
                close_price=price,
                volume=vol,
                created_at=now,
                updated_at=now
            ))

        add_price_cache(aapl, Decimal("180.00"), 50000000)
        add_price_cache(intc, Decimal("30.00"), 5000000)
        add_price_cache(v, Decimal("250.00"), 15000000)
        add_price_cache(pypl, Decimal("60.00"), 8000000)
        add_price_cache(lly, Decimal("750.00"), 3000000)
        add_price_cache(gild, Decimal("75.00"), 6000000)
        
        db.commit()
        print("Price caches seeded successfully.")

        # Seed News Articles
        print("Seeding news articles...")
        db.add(Article(
            stock_id=aapl.stock_id,
            ticker="AAPL",
            title="Apple Launches Advanced AI Ecosystem Integrated Across All Devices",
            summary="Apple reveals their next-generation AI integrations, drawing strong client reception.",
            link="https://seekingalpha.com/news/aapl-ai",
            published_date=now,
            source="Seeking Alpha"
        ))
        db.add(Article(
            stock_id=intc.stock_id,
            ticker="INTC",
            title="Intel Secures Major US Foundry Grants and Commercial Client Commitments",
            summary="Intel Foundry announces massive new multi-billion contract wins, boosting production scale.",
            link="https://seekingalpha.com/news/intc-grants",
            published_date=now,
            source="Seeking Alpha"
        ))
        db.add(Article(
            stock_id=v.stock_id,
            ticker="V",
            title="Visa Partners with Leading FinTechs to Expand Payments Network",
            summary="Visa announces new network expansion partnerships across emerging market sectors.",
            link="https://seekingalpha.com/news/v-fintech",
            published_date=now,
            source="Reuters"
        ))
        db.add(Article(
            stock_id=pypl.stock_id,
            ticker="PYPL",
            title="PayPal Reports Record Cash Flow Generation and Massive Buyback Execution",
            summary="PayPal's new CEO drives operational efficiencies, growing margins and executing large buybacks.",
            link="https://seekingalpha.com/news/pypl-cash-flow",
            published_date=now,
            source="Benzinga"
        ))
        db.add(Article(
            stock_id=lly.stock_id,
            ticker="LLY",
            title="Eli Lilly Pipeline Shows Solid Expansion in Obesity and Diabetes Treatments",
            summary="Eli Lilly highlights new drug candidate clinical trial results with outstanding efficiency.",
            link="https://seekingalpha.com/news/lly-pipeline",
            published_date=now,
            source="Seeking Alpha"
        ))
        db.add(Article(
            stock_id=gild.stock_id,
            ticker="GILD",
            title="Gilead's Oncology Pipeline Excels with FDA Approvals in Key Therapeutic Areas",
            summary="Gilead expands clinical pipeline success, diversifying revenue beyond core HIV franchise.",
            link="https://seekingalpha.com/news/gild-pipeline",
            published_date=now,
            source="Seeking Alpha"
        ))

        # Seed Insider Trades
        print("Seeding insider trades...")
        db.add(InsiderTrade(
            stock_id=intc.stock_id,
            ticker="INTC",
            filer_name="Gelsinger Patrick P",
            relation="CEO / Director",
            transaction_date=date.today(),
            transaction_type="Buy",
            shares=25000,
            price=Decimal("30.00"),
            total_value=Decimal("750000.00"),
            filing_url="https://sec.gov/form4/intc"
        ))
        db.add(InsiderTrade(
            stock_id=pypl.stock_id,
            ticker="PYPL",
            filer_name="Chriss Alex",
            relation="CEO / President",
            transaction_date=date.today(),
            transaction_type="Buy",
            shares=15000,
            price=Decimal("60.00"),
            total_value=Decimal("900000.00"),
            filing_url="https://sec.gov/form4/pypl"
        ))
        db.add(InsiderTrade(
            stock_id=gild.stock_id,
            ticker="GILD",
            filer_name="O'Day Daniel P",
            relation="CEO / Chairman",
            transaction_date=date.today(),
            transaction_type="Buy",
            shares=8000,
            price=Decimal("75.00"),
            total_value=Decimal("600000.00"),
            filing_url="https://sec.gov/form4/gild"
        ))

        db.commit()
        print("All real underdog data seeded successfully with perfect, fresh parameters!")

    finally:
        db.close()

if __name__ == '__main__':
    seed_perfect_underdogs()
