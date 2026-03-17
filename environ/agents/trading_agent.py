"""
Trading Agent — integrates Crypto Agent and News Agent outputs
to produce portfolio trading actions for each asset.

Output schema (list of dicts):
    [
        {
            "symbol": "BTC",
            "action": float,   # > 0: buy fraction of post-sell cash, < 0: sell fraction of holdings, 0: hold
            "rationale": str,
        },
        ...
    ]

action is in (-1, 1]:
  +0.3  → allocate 30% of post-sell cash pool to buy this asset
  -0.5  → sell 50% of current holdings of this asset
   0.0  → hold

Execution order: ALL sells execute first; then ALL buys are funded
from the resulting cash pool simultaneously (order-independent).
If total buy fractions exceed 1.0 they are scaled down proportionally.
"""

import json
import logging

from .base import BaseAgent

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are the Trading Agent in a multi-agent cryptocurrency portfolio management system.

You receive:
1. Crypto Agent signals: technical/fundamental scores for each asset (signal in [-1,1]).
2. News Agent signals: overall market sentiment and per-coin news signals.
3. Current portfolio state: cash balance, per-asset holdings with unrealised P&L, total value and overall P&L.

Your task is to produce executable trading actions for each asset in the universe.

Trading action semantics (IMPORTANT — read carefully):
  - Positive value (0, 1]:   BUY — allocate that fraction of the POST-SELL CASH POOL to purchase this asset.
  - Negative value [-1, 0):  SELL — liquidate that fraction of CURRENT HOLDINGS of this asset.
  - Zero:                    HOLD — no change.

Execution model:
  ALL sell actions execute first, converting holdings to cash.
  ALL buy actions then execute simultaneously from the resulting cash pool.
  Each positive action value is a fraction of that post-sell cash total.
  Example: actions BTC=0.4, ETH=0.3 with $100k post-sell cash → $40k to BTC, $30k to ETH.
  If the sum of buy fractions exceeds 1.0, they are scaled down proportionally — you do NOT need to
  manually ensure they sum to ≤1, but be intentional about the relative sizes.

Portfolio context you receive per asset:
  - value_usd:      current market value of your holding
  - cost_basis_usd: total USD paid to acquire the current position
  - pnl_usd:        unrealised profit/loss (value_usd − cost_basis_usd)
  - pnl_pct:        unrealised return in percent

Use the per-asset P&L to inform decisions (e.g. consider taking profits on large winners,
cutting losses on underperformers) alongside the incoming signals.

Constraints:
  - Do not sell more than 100% of a holding (action ≥ -1.0).
  - Avoid concentrating too much in a single asset.
  - Be conservative when signals are conflicting or confidence is low.

Produce a JSON array with one object per asset (include ALL assets in the universe):
  "symbol"    – ticker string
  "action"    – float in [-1.0, 1.0]  (0.0 = hold)
  "rationale" – one-sentence justification referencing signals and/or P&L

Return ONLY valid JSON, no prose outside the JSON.\
"""

_SKILL_DESCRIPTIONS = """\
- signal_aggregator(crypto_signal, news_signal, weights): Compute weighted composite signal.
- risk_budget_allocator(signals, cash, holdings, max_position): Compute safe allocation fractions.
- conflict_detector(crypto_signal, news_signal): Flag assets where signals disagree.
"""

UNIVERSE = ["BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "ADA", "BCH",
            "HYPE", "XMR", "ZEC", "LTC", "SUI", "AVAX", "HBAR"]


class TradingAgent(BaseAgent):
    """Converts agent signals into portfolio trading actions."""

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    def _skill_descriptions(self) -> str:
        return _SKILL_DESCRIPTIONS

    def build_user_message(self, context: dict) -> str:
        """
        Args:
            context: {
                "week": "2025-W03",
                "crypto_signals": [...],     # from CryptoAgent
                "news_output": {...},        # from NewsAgent
                "portfolio": {
                    "cash": float,
                    "total_value": float,
                    "pnl_usd": float,        # overall unrealised P&L vs initial $100k
                    "pnl_pct": float,
                    "holdings": {
                        "BTC": {
                            "value_usd": float,
                            "cost_basis_usd": float,
                            "pnl_usd": float,
                            "pnl_pct": float,
                        },
                        ...  # only assets currently held
                    }
                }
            }
        """
        week = context.get("week", "unknown")
        crypto_signals = context.get("crypto_signals", [])
        news_output = context.get("news_output", {})
        portfolio = context.get("portfolio", {})

        # Build a combined signal table for clarity
        news_coin_map = {
            s["symbol"]: s
            for s in news_output.get("coin_signals", [])
        }

        combined = []
        for sig in crypto_signals:
            sym = sig["symbol"]
            news_sig = news_coin_map.get(sym, {})
            combined.append({
                "symbol": sym,
                "crypto_signal": sig.get("signal", 0),
                "crypto_confidence": sig.get("confidence", 0),
                "news_signal": news_sig.get("signal", news_output.get("overall_sentiment", 0)),
                "news_confidence": news_sig.get("confidence", 0.5),
            })

        payload = {
            "week": week,
            "overall_news_sentiment": news_output.get("overall_sentiment", 0),
            "overall_news_rationale": news_output.get("overall_rationale", ""),
            "asset_signals": combined,
            "portfolio": portfolio,
        }

        return (
            f"Week: {week}\n\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Determine trading actions for all assets and return the JSON array."
        )

    def parse_response(self, content: str) -> list[dict]:
        actions = self._extract_json(content)
        assert isinstance(actions, list), f"Expected list, got {type(actions)}"
        # Clamp action values to [-1, 1]
        for item in actions:
            a = float(item.get("action", 0))
            item["action"] = max(-0.9999, min(1.0, a))
        return actions

