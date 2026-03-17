"""
Crypto Agent — analyses recent price, volume, and market cap data
from CoinGecko and produces a structured signal for each asset.

Output schema (list of dicts, one per asset):
    [
        {
            "symbol": "BTC",
            "signal": float,      # in [-1, 1]; negative=bearish, positive=bullish
            "confidence": float,  # in [0, 1]
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
You are the Crypto Agent in a multi-agent cryptocurrency portfolio management system.

Your role is to analyse recent market data for a set of cryptocurrencies and produce \
a directional signal for each one.

For each asset you will receive the last N days of:
  - close  : daily closing price (USD)
  - volume : daily trading volume (USD)
  - market_cap : daily market capitalisation (USD)

Dates are in ascending order (oldest first, most recent last).

Produce a JSON array with one object per asset:
  "symbol"     – ticker string
  "signal"     – float in [-1.0, 1.0]  (−1 = strong bearish, 0 = neutral, 1 = strong bullish)
  "confidence" – float in [0.0, 1.0]
  "rationale"  – one-sentence explanation

Return ONLY valid JSON, no prose outside the JSON.\
"""

_SKILL_DESCRIPTIONS = """\
- trend_analysis(prices): Identify direction and strength of the price trend.
- volume_analysis(volumes): Assess whether volume confirms or contradicts the price move.
- market_cap_analysis(market_caps): Evaluate relative size and any significant changes.
"""


class CryptoAgent(BaseAgent):
    """Analyses raw market data and returns per-asset signals."""

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
                "assets": [
                    {
                        "symbol": "BTC",
                        "dates":       ["2025-01-01", ...],
                        "close":       [93507.86, ...],
                        "volume":      [3.2e10, ...],
                        "market_cap":  [1.85e12, ...],
                    },
                    ...
                ]
            }
        """
        week = context.get("week", "unknown")
        assets = context.get("assets", [])
        return (
            f"Week: {week}\n\n"
            f"Asset market data:\n{json.dumps(assets, indent=2)}\n\n"
            "Analyse each asset and return the JSON signal array."
        )

    def parse_response(self, content: str) -> list[dict]:
        signals = self._extract_json(content)
        assert isinstance(signals, list), f"Expected list, got {type(signals)}"
        return signals

