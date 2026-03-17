"""
Collaborative Architecture.

Agents iteratively exchange information and refine their outputs
before the Trading Agent produces the final portfolio action.

Round 0 (independent):
    CryptoAgent and NewsAgent each produce initial signals.

Refinement rounds:
    CryptoAgent sees the NewsAgent's output and revises its signals.
    NewsAgent sees the CryptoAgent's output and revises its signals.
    (memory is NOT updated for intermediate rounds)

Final:
    TradingAgent integrates the refined signals.
    All three agents commit their final outputs to memory.
"""

import logging

from environ.agents import CryptoAgent, NewsAgent, TradingAgent
from environ.agents.rag_crypto_agent import RAGCryptoAgent
from environ.agents.skill_crypto_agent import SkillCryptoAgent
from .base_arch import _build_rag_store, _collect_rag_examples, _inject_peer_context, _llm_capability

logger = logging.getLogger(__name__)


class CollaborativeMAS:
    """
    Collaborative multi-agent system with iterative cross-agent refinement.

    Args:
        refinement_rounds: Number of cross-agent refinement iterations (default 1).
        memory_window:     Past weekly outputs each agent sees in its prompt.
    """

    def __init__(
        self,
        capability: str = "zero_shot",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        memory_window: int = 4,
        refinement_rounds: int = 1,
    ):
        llm_cap = _llm_capability(capability)
        if capability == "rag":
            crypto_cls = RAGCryptoAgent
        elif capability == "skill":
            crypto_cls = SkillCryptoAgent
        else:
            crypto_cls = CryptoAgent
        self.crypto_agent      = crypto_cls(  capability=llm_cap, model=model, temperature=temperature, memory_window=memory_window)
        self.news_agent        = NewsAgent(   capability=llm_cap, model=model, temperature=temperature, memory_window=memory_window)
        self.trading_agent     = TradingAgent(capability=llm_cap, model=model, temperature=temperature, memory_window=memory_window)
        self.refinement_rounds = refinement_rounds
        self._rag_store        = _build_rag_store(capability)

    def run(
        self,
        week: str,
        indicators: list[dict],
        articles: list[dict],
        portfolio: dict,
    ) -> list[dict]:
        rag_examples = _collect_rag_examples(self._rag_store, indicators, week=week)

        # --- Round 0: independent initial pass (no memory update yet) ---
        logger.info("[Collaborative] Week %s — Round 0: CryptoAgent", week)
        crypto_signals = self.crypto_agent.run(
            {"week": week, "assets": indicators, "rag_examples": rag_examples},
            memorize=False,
        )

        logger.info("[Collaborative] Week %s — Round 0: NewsAgent", week)
        news_output = self.news_agent.run(
            {"week": week, "articles": articles}, memorize=False
        )

        # --- Refinement rounds (no memory update for intermediate results) ---
        for r in range(self.refinement_rounds):
            logger.info("[Collaborative] Week %s — Refinement round %d", week, r + 1)

            crypto_ctx = _inject_peer_context(
                base_context={"week": week, "assets": indicators, "rag_examples": rag_examples},
                peer_name="News Agent",
                peer_output=news_output,
            )
            crypto_signals = self.crypto_agent.run(crypto_ctx, memorize=False)

            news_ctx = _inject_peer_context(
                base_context={"week": week, "articles": articles},
                peer_name="Crypto Agent",
                peer_output={"signals": crypto_signals},
            )
            news_output = self.news_agent.run(news_ctx, memorize=False)

        # --- Commit final refined outputs to each agent's memory ---
        self.crypto_agent._store_memory(week, crypto_signals)
        self.news_agent._store_memory(week, news_output)

        # --- Trading Agent produces final actions (memory updated normally) ---
        logger.info("[Collaborative] Week %s — TradingAgent (final)", week)
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
