"""
Single Agent (benchmark) — receives raw market data AND news articles in one
call and directly produces portfolio trading actions, bypassing the MAS pipeline.

This is a single-agent baseline for comparison against the multi-agent system.

Input context:
    {
        "week": "2025-W03",
        "assets": [                      # same as CryptoAgent input
            {
                "symbol": "BTC",
                "dates":       ["2025-01-01", ...],
                "close":       [93507.86, ...],
                "volume":      [3.2e10, ...],
                "market_cap":  [1.85e12, ...],
            },
            ...
        ],
        "articles": [                    # same as NewsAgent input
            {"title": ..., "published_at": ..., "text": ...},
            ...
        ],
        "portfolio": {                   # same as TradingAgent input
            "cash": float,
            "total_value": float,
            "pnl_usd": float,
            "pnl_pct": float,
            "holdings": {
                "BTC": {
                    "value_usd": float,
                    "cost_basis_usd": float,
                    "pnl_usd": float,
                    "pnl_pct": float,
                },
                ...
            }
        }
    }

Output schema (identical to TradingAgent):
    [
        {
            "symbol": "BTC",
            "action": float,   # > 0: buy fraction of post-sell cash, < 0: sell fraction of holdings, 0: hold
            "rationale": str,
        },
        ...
    ]
"""

import json
import logging

from .base import BaseAgent

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a cryptocurrency portfolio manager.

You will receive:
1. Raw market data (price, volume, market cap) for a set of assets over the past N days.
2. A set of recent news articles from the cryptocurrency space.
3. The current portfolio state: cash balance, per-asset holdings with unrealised P&L.

Your task is to analyse all of this information and produce executable trading actions for
every asset in the universe.

Trading action semantics (IMPORTANT — read carefully):
  - Positive value (0, 1]:   BUY — allocate that fraction of the POST-SELL CASH POOL to purchase this asset.
  - Negative value [-1, 0):  SELL — liquidate that fraction of CURRENT HOLDINGS of this asset.
  - Zero:                    HOLD — no change.

Execution model:
  ALL sell actions execute first, converting holdings to cash.
  ALL buy actions then execute simultaneously from the resulting cash pool.
  Each positive action value is a fraction of that post-sell cash total.
  If the sum of buy fractions exceeds 1.0, they are scaled down proportionally.

Constraints:
  - Do not sell more than 100% of a holding (action ≥ -1.0).
  - Avoid concentrating too much in a single asset.
  - Be conservative when signals are conflicting or confidence is low.

Asset universe: BTC, ETH, BNB, XRP, SOL, TRX, ADA, BCH, HYPE, XMR, ZEC, LTC, SUI, AVAX, HBAR

Produce a JSON array with one object per asset (include ALL assets in the universe):
  "symbol"    – ticker string
  "action"    – float in [-1.0, 1.0]  (0.0 = hold)
  "rationale" – one-sentence justification

Return ONLY valid JSON, no prose outside the JSON.\
"""

_SKILL_DESCRIPTIONS = """\
- trend_analysis(prices): Identify direction and strength of the price trend.
- volume_analysis(volumes): Assess whether volume confirms or contradicts the price move.
- sentiment_classifier(text): Classify financial text as bullish, neutral, or bearish.
- event_detector(text): Identify major events (regulatory news, hacks, upgrades, listings).
- risk_budget_allocator(signals, cash, holdings): Compute safe allocation fractions.
"""


class SingleAgent(BaseAgent):
    """Single-agent baseline that processes raw market data and news in one call."""

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    def _skill_descriptions(self) -> str:
        return _SKILL_DESCRIPTIONS

    def build_user_message(self, context: dict) -> str:
        week = context.get("week", "unknown")
        assets = context.get("assets", [])
        articles = context.get("articles", [])
        portfolio = context.get("portfolio", {})

        # Truncate article texts to keep prompt size manageable (same as NewsAgent)
        trimmed_articles = []
        for art in articles:
            trimmed_articles.append({
                "title": art.get("title", ""),
                "published_at": art.get("published_at", ""),
                "text": art.get("text", "")[:800],
            })

        payload = {
            "week": week,
            "market_data": assets,
            "news_articles": trimmed_articles,
            "portfolio": portfolio,
        }

        return (
            f"Week: {week}\n\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Analyse the market data and news, then determine trading actions for all assets "
            "and return the JSON array."
        )

    def parse_response(self, content: str) -> list[dict]:
        actions = self._extract_json(content)
        assert isinstance(actions, list), f"Expected list, got {type(actions)}"
        # Clamp action values to [-1, 1] (same as TradingAgent)
        for item in actions:
            a = float(item.get("action", 0))
            item["action"] = max(-0.9999, min(1.0, a))
        return actions
