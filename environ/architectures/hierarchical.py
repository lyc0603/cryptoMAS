"""
Hierarchical Architecture.

The Crypto Agent and News Agent are analytical sub-agents.
The Trading Agent acts as a supervisory agent that integrates
their reports and makes all final decisions.

Flow:
    CryptoAgent  ──► report ──┐
                              ├──► TradingAgent (supervisor) ──► actions
    NewsAgent    ──► report ──┘
"""

import logging

from environ.agents import CryptoAgent, NewsAgent, TradingAgent
from environ.agents.rag_crypto_agent import RAGCryptoAgent
from environ.agents.skill_crypto_agent import SkillCryptoAgent
from .base_arch import _build_rag_store, _collect_rag_examples, _llm_capability

logger = logging.getLogger(__name__)


class HierarchicalMAS:
    """
    Hierarchical multi-agent system.

    The Trading Agent's prompt frames it as a supervisor integrating
    analyst reports, rather than a peer reading a shared board.
    """

    SUPERVISOR_PREFIX = (
        "You are the supervisory Trading Agent receiving structured analytical "
        "reports from two subordinate analyst agents. Your role is to integrate "
        "their findings and produce the final portfolio decisions.\n\n"
    )

    def __init__(
        self,
        capability: str = "zero_shot",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        memory_window: int = 4,
    ):
        llm_cap = _llm_capability(capability)
        if capability == "rag":
            crypto_cls = RAGCryptoAgent
        elif capability == "skill":
            crypto_cls = SkillCryptoAgent
        else:
            crypto_cls = CryptoAgent
        self.crypto_agent  = crypto_cls(     capability=llm_cap, model=model, temperature=temperature, memory_window=memory_window)
        self.news_agent    = NewsAgent(      capability=llm_cap, model=model, temperature=temperature, memory_window=memory_window)
        self.trading_agent = _SupervisorTradingAgent(capability=llm_cap, model=model, temperature=temperature, memory_window=memory_window)
        self._rag_store    = _build_rag_store(capability)

    def run(
        self,
        week: str,
        indicators: list[dict],
        articles: list[dict],
        portfolio: dict,
    ) -> list[dict]:
        logger.info("[Hierarchical] Week %s — CryptoAgent report", week)
        crypto_signals = self.crypto_agent.run({
            "week":         week,
            "assets":       indicators,
            "rag_examples": _collect_rag_examples(self._rag_store, indicators, week=week),
        })

        logger.info("[Hierarchical] Week %s — NewsAgent report", week)
        news_output = self.news_agent.run({"week": week, "articles": articles})

        logger.info("[Hierarchical] Week %s — TradingAgent (supervisor)", week)
        actions = self.trading_agent.run({
            "week":           week,
            "crypto_signals": crypto_signals,
            "news_output":    news_output,
            "portfolio":      portfolio,
        })

        return actions

    # ------------------------------------------------------------------
    # Memory persistence
    # ------------------------------------------------------------------

    def get_memory_state(self) -> dict:
        return {
            "crypto_agent":  self.crypto_agent.get_memory_state(),
            "news_agent":    self.news_agent.get_memory_state(),
            "trading_agent": self.trading_agent.get_memory_state(),
        }

    def set_memory_state(self, state: dict) -> None:
        self.crypto_agent.set_memory_state( state.get("crypto_agent",  []))
        self.news_agent.set_memory_state(   state.get("news_agent",    []))
        self.trading_agent.set_memory_state(state.get("trading_agent", []))


class _SupervisorTradingAgent(TradingAgent):
    """TradingAgent with supervisor framing prepended to its system prompt."""

    @property
    def system_prompt(self) -> str:
        return HierarchicalMAS.SUPERVISOR_PREFIX + super().system_prompt
