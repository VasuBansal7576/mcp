import json
import asyncio

from maverick_mcp.data import SessionLocal
from maverick_mcp.services.screening.underdog_screener import UnderdogScreener
from maverick_mcp.api.routers.underdog import get_underdog_portfolio_allocate

async def run_live_pipeline():
    print("==============================================================")
    print("🚀 RUNNING LOCAL rule-based screener & asset allocation 🚀")
    print("==============================================================")
    
    db = SessionLocal()
    try:
        # 1. Run UnderdogScreener
        print("\nStep 1: Running UnderdogScreener to find candidates...")
        screener = UnderdogScreener(db)
        candidates = screener.screen_stocks()
        print(f"✓ Found {len(candidates)} candidates matching all 11 strict underdog rules.")
        
        for cand in candidates:
            print(f"  • {cand['symbol']} ({cand['company_name']})")
            print(f"    - Sector: {cand['sector']} | P/E: {cand['pe_ratio']} (Sector Median: {cand['sector_median_pe']:.2f})")
            print(f"    - ROIC: {cand['roic']}% | F-Score: {cand['piotroski_f_score']} | FCF Growth: {cand['fcf_growth_yoy']:.1%}")
            print(f"    - Close Price: ${cand['close_price']:.2f} | Insider Align: {cand['insider_buying_6m']}")
            print(f"    - Flags: {cand['flags']}")
            
        # 2. Run Portfolio Allocation for $1000 capital (Normal Risk / Equity ETFs safety net)
        print("\nStep 2: Generating Portfolio Allocation for $1,000 (Standard Equity Buffer)...")
        alloc_normal = await get_underdog_portfolio_allocate(capital=1000.0, low_risk=False, db=db)
        print(json.dumps(alloc_normal, indent=2))
        
        # 3. Run Portfolio Allocation for $1000 capital (Low Risk / Treasury Cash safety net)
        print("\nStep 3: Generating Portfolio Allocation for $1,000 (Low-Risk Treasury Buffer)...")
        alloc_low_risk = await get_underdog_portfolio_allocate(capital=1000.0, low_risk=True, db=db)
        print(json.dumps(alloc_low_risk, indent=2))
        
    finally:
        db.close()

if __name__ == '__main__':
    asyncio.run(run_live_pipeline())
