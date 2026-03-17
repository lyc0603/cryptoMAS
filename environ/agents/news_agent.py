"""
News Agent — analyses Cointelegraph articles for a given week and
produces an overall market sentiment signal plus per-coin signals
where articles are coin-specific.

Output schema:
    {
        "week": "2025-W03",
        "overall_sentiment": float,   # [-1, 1]
        "overall_rationale": str,
        "coin_signals": [
            {
                "symbol": "BTC",
                "signal": float,      # [-1, 1]
                "confidence": float,  # [0, 1]
                "rationale": str,
            },
            ...
        ]
    }
"""

import json
import logging

from .base import BaseAgent

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are the News Agent in a multi-agent cryptocurrency portfolio management system.

Your role is to read a set of Cointelegraph news articles from a single week and:
1. Assess the overall market sentiment for the cryptocurrency market.
2. Extract coin-specific signals where articles mention specific cryptocurrencies.

Each article is provided as:
  - title
  - published_at (UTC ISO timestamp)
  - text (article body)

Produce a JSON object with:
  "week"                – ISO week string
  "overall_sentiment"   – float in [-1.0, 1.0] (−1 = very bearish, 1 = very bullish)
  "overall_rationale"   – one-sentence summary of the news environment
  "coin_signals"        – array of objects, one per mentioned coin:
      "symbol"          – ticker (BTC, ETH, SOL, etc.)
      "signal"          – float in [-1.0, 1.0]
      "confidence"      – float in [0.0, 1.0]
      "rationale"       – one-sentence explanation

Only include coins from this universe: BTC, ETH, BNB, XRP, SOL, TRX, ADA, BCH,
HYPE, XMR, ZEC, LTC, SUI, AVAX, HBAR.
If a coin is not mentioned in the news, omit it from coin_signals.

Return ONLY valid JSON, no prose outside the JSON.\
"""

_SKILL_DESCRIPTIONS = """\
- sentiment_classifier(text): Classify financial text as bullish, neutral, or bearish.
- event_detector(text): Identify major events (regulatory news, hacks, upgrades, listings).
- entity_recognizer(text): Extract mentioned cryptocurrency tickers from article text.
"""


class NewsAgent(BaseAgent):
    """Analyses news articles and returns market sentiment signals."""

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
                "articles": [
                    {"title": ..., "published_at": ..., "text": ...},
                    ...
                ]
            }
        """
        week = context.get("week", "unknown")
        articles = context.get("articles", [])

        # Truncate article texts to keep prompt size manageable
        trimmed = []
        for art in articles:
            trimmed.append({
                "title": art.get("title", ""),
                "published_at": art.get("published_at", ""),
                "text": art.get("text", "")[:800],   # first ~800 chars
            })

        articles_json = json.dumps(trimmed, indent=2, ensure_ascii=False)
        return (
            f"Week: {week}\n\n"
            f"News articles ({len(trimmed)} total):\n{articles_json}\n\n"
            "Analyse the news and return the JSON sentiment object."
        )

    def parse_response(self, content: str) -> dict:
        result = self._extract_json(content)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        return result

