# 📊 MaverickMCP - Underdog Stock Screener & Asset Allocation Agent

**MaverickMCP** is a production-grade, highly cost-effective multi-agent system and quantitative asset allocation engine designed to identify overlooked, undervalued U.S. equities and structure them into risk-controlled portfolios. 

Built from first principles for personal-use financial analysis, it implements strict institutional-grade financial quality filters (like Piotroski F-score and ROIC), enforces a strict diversification safety net, and dynamically monitors positions using an automated 10-minute trailing stop-loss scheduler.

---

## 🎯 Core Features & Design Principles

* **11 Strict Financial Quality Gates**: Eliminates "Value Traps" (cheap but deteriorating companies) using strict constraints on analyst coverage, relative P/E, Return on Invested Capital (ROIC $\ge 10.0\%$), Piotroski F-Score ($\ge 6$), positive operating cash flow, institutional liquidity, volatility, and earnings proximity.
* **Risk-Capped Asset Allocation**: Caps turnaround candidates at a **strict 10% ($100 max) individual limit** and **15% ($150 max) sector concentration limit** of your total capital.
* **Dynamic Safety Buffer**: Allocates the remaining **70%+ of your capital** to safe, highly liquid assets. Choose between:
  * **Option A (Standard Growth)**: Core equity index ETFs (`SPY` and `QQQ`).
  * **Option B (Low-Risk Preservation)**: Ultra-safe, short-duration U.S. Treasury Bill/Bond ETFs (`BIL` and `SHV`).
* **Active Trailing Stop-Loss Monitor**: An automated background job runs every 10 minutes to track position prices. If a stock falls **15% or more from its dynamic peak price since purchase**, the system triggers a **RED ALERT** and immediately dispatches a **SELL** warning via Slack/Discord webhooks.

---

## 🏗️ System Architecture

MaverickMCP implements a **2-Tier Hybrid Architecture** to achieve high speed and eliminate unnecessary LLM costs:

```
                      +--------------------------------------------+
                      |           FastAPI Web API Server           |
                      |        (:8003/api/underdog/...)            |
                      +---------------------+----------------------+
                                            |
                                            v
                      +--------------------------------------------+
                      |        Underdog Pre-Screening Engine       |
                      |  (Zero LLM Cost · Pure rule-based SQL/DB)  |
                      +---------------------+----------------------+
                                            |
                         Overlooked? Yes    |  (Top 3 Turnarounds)
                                            v
                      +--------------------------------------------+
                      |          Redis Fingerprint Cache           |
                      |   (Skip LLM if ticker, price, articles,    |
                      |        or filings remain unchanged)        |
                      +---------------------+----------------------+
                                            |
                                            | Cache Miss
                                            v
                      +--------------------------------------------+
                      |       3-Node LLM Agent Reasoning Flow      |
                      |                                            |
                      |  Node 1: News Sentiment (Score -1.0 to 1.0)|
                      |  Node 2: SEC Filing Risk (LOW/MED/HIGH)    |
                      |  Node 3: CIO Decision (BUY/HOLD/SELL, Conf)|
                      +--------------------------------------------+
```

### The 2-Tier Operational Model:
1. **Tier 1: High-Speed Pre-Screener ($0.00 cost)**: SQL/DB rules narrow the entire market universe to candidates that satisfy all 11 financial criteria. The asset allocation engine calculates weights mathematically under strict caps.
2. **Tier 2: Qualitative 3-Node AI Agent (Configurable cost)**: Connects to OpenAI or OpenRouter to parse news sentiment, read corporate descriptions for operational filing risks, and output a high-conviction decision.

---

## 💻 Tech Stack

* **Runtime**: Python 3.12+ (Lightning-fast dependency resolution via `uv`)
* **API Framework**: FastAPI & FastMCP (seamless integration with Claude Desktop and MCP clients)
* **Database & Caching**: SQLite (equipped with dynamic dynamic migration on boot) + Redis (pooling + msgpack serialization, with graceful in-memory fallbacks)
* **Orchestration**: LangChain / LangGraph provider-agnostic integration
* **Ingestion**: `yfinance` market feed + free SEC EDGAR API (Form 4 Insider Transactions) + `feedparser` (Seeking Alpha, Benzinga, and MarketWatch news RSS feeds)

---

## 📈 Model Cost Analysis & Tracking

The system features a **Global Token Cost Accumulator** that tracks dynamic spending in real-time. If Live LLM mode is activated, the estimated operating cost is **under $0.50/day** due to our 80%+ Redis Cache hit rate:

| Model Name | Provider | Cost (Input/Output per 1M) | Cache Hit Rate | Est. Cost / Day | Conviction ROI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek Chat** | DeepSeek | `$0.27 / $1.10` | 75% - 85% | **~$0.28 / day** | High |
| **GPT-4o Mini** | OpenAI | `$0.15 / $0.60` | 75% - 85% | **~$0.18 / day** | Balanced |
| **Claude 3 Haiku** | Anthropic | `$0.25 / $1.25` | 75% - 85% | **~$0.35 / day** | High |
| **Gemini 2.5 Flash**| OpenRouter | `$0.075 / $0.30` | 75% - 85% | **~$0.09 / day** | Cost-Minimal |

---

## 🚀 Quick Start & Local Testing Guide

Your codebase includes two self-contained scripts directly in the `scripts/` directory to allow you or your client to test and verify the entire system locally with **zero cost or API key friction**.

### 1. Installation & Environment Setup

Ensure you have [Python 3.12](https://www.python.org/downloads/) installed. We recommend [uv](https://docs.astral.sh/uv/) for instant package setup.

```bash
# Clone the repository
git clone https://github.com/VasuBansal7576/mcp.git
cd mcp

# Install dependencies and create virtual environment automatically
uv sync

# Copy env template
cp .env.example .env
```

### 2. Seed Deterministic Turnaround Candidates (Step A)
Reset your local database and seed it with three perfect, realistic turnaround stocks (`INTC`, `PYPL`, `GILD`) along with safe price caches, executive insider filings, and news articles:
```bash
uv run python scripts/seed_perfect_underdogs.py
```

### 3. Run the Quantitative Screener & Allocation Plans (Step B)
Execute the screener and generate both the **Standard Growth** and **Low-Risk** `$1,000` capital asset allocation plans:
```bash
uv run python scripts/run_live_screening_allocation.py
```

### 4. Start the HTTP API Server (Step C)
Boot up the FastAPI web and MCP server locally:
```bash
make dev
```

### 5. Query Endpoints via Command Line (Step D)
Open a separate terminal window and test the live JSON endpoints:
* **Query Standard Growth $1,000 Portfolio Weights**:
  ```bash
  curl -X GET "http://localhost:8003/api/underdog/portfolio/allocate?capital=1000.0&low_risk=false"
  ```
* **Query Low-Risk Treasury-Hedged $1,000 Portfolio Weights**:
  ```bash
  curl -X GET "http://localhost:8003/api/underdog/portfolio/allocate?capital=1000.0&low_risk=true"
  ```
* **Trigger the Trailing Stop-Loss Monitor & Webhook Dispatcher**:
  ```bash
  curl -X GET "http://localhost:8003/api/underdog/portfolio/monitor?portfolio_name=My%20Portfolio"
  ```

---

## 🧪 Automated Testing & Code Quality

Verify that all quantitative rules, volatility breakers, stop-losses, and edge cases pass flawlessly:
```bash
# Run the complete test suite (918 tests)
make test

# Run static linting and strict typecheck analysis (0 errors)
make check
```

---

## ⚖️ Fiduciary Disclosure & Professional Disclaimer

> [!WARNING]
> **Educational & Informational Use Only**
> This project is designed solely as a software development demonstrator for personal-use quantitative analysis. It does **not** constitute financial, tax, or trading advice. 
> 
> * **Market Risk**: Equity markets are inherently volatile. Turnaround stocks carry extreme specific risks. You can lose all of your allocated capital.
> * **No Execution Automation**: This software does **not** execute trades or manage real brokerage funds. Any investment must be manually processed by the user through a licensed broker-dealer.
> * **No Fiduciary Duty**: The developer of this software holds zero liability for any financial decisions, profits, or losses incurred as a result of using this application. Please perform your own due diligence.
