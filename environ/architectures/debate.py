"""
Debate Architecture.

CryptoAgent and NewsAgent hold independent initial positions, then
engage in structured debate: each round they explicitly challenge or
defend the opponent's argument before a judge (TradingAgent) issues
the final portfolio decision based on the full debate transcript.

Flow:
    Round 0 (independent):
        CryptoAgent  ──► crypto_pos_0
        NewsAgent    ──► news_pos_0

    Debate rounds 1..N (no memory update):
        CryptoAgent sees news_pos_{r-1} + own crypto_pos_{r-1} ──► crypto_pos_r
        NewsAgent   sees crypto_pos_{r-1} + own news_pos_{r-1} ──► news_pos_r

    Final:
        TradingAgent receives full debate transcript
        (all rounds of both agents' evolving positions) ──► actions
        All three agents commit their final outputs to memory.

Key difference from Collaborative:
    Agents are explicitly asked to *challenge* the opponent and justify
    any agreement or disagreement, not merely refine in light of new info.
    The TradingAgent sees how positions evolved, not just final signals.
"""

import json
import logging

from environ.agents import CryptoAgent, NewsAgent, TradingAgent
from environ.agents.rag_crypto_agent import RAGCryptoAgent
from environ.agents.skill_crypto_agent import SkillCryptoAgent
from .base_arch import _build_rag_store, _collect_rag_examples, _inject_debate_context, _llm_capability

logger = logging.getLogger(__name__)


class _DebateTradingAgent(TradingAgent):
    """
    TradingAgent variant that receives a full debate transcript instead
    of a single pair of final signals.
    """

    JUDGE_PREFIX = (
        "You are the Trading Agent acting as a JUDGE in a structured debate "
        "between two analyst agents (Crypto Agent and News Agent). "
        "You will receive the full transcript of their debate — their initial "
        "positions and how those positions evolved round by round. "
        "Your role is to weigh the strength of each side's arguments, identify "
        "where they converged or remained in disagreement, and produce the final "
        "portfolio decisions.\n\n"
    )

    @property
    def system_prompt(self) -> str:
        return self.JUDGE_PREFIX + super().system_prompt

    def build_user_message(self, context: dict) -> str:
        week              = context.get("week", "unknown")
        transcript        = context.get("debate_transcript", [])
        portfolio         = context.get("portfolio", {})

        # Extract final-round signals for the standard signal table
        final_round       = transcript[-1] if transcript else {}
        crypto_signals    = final_round.get("crypto", [])
        news_output       = final_round.get("news", {})

        # Build combined signal table (same as parent)
        news_coin_map = {
            s["symbol"]: s
            for s in news_output.get("coin_signals", [])
        }
        combined = []
        for sig in crypto_signals:
            sym      = sig["symbol"]
            news_sig = news_coin_map.get(sym, {})
            combined.append({
                "symbol":           sym,
                "crypto_signal":    sig.get("signal", 0),
                "crypto_confidence": sig.get("confidence", 0),
                "news_signal":      news_sig.get("signal", news_output.get("overall_sentiment", 0)),
                "news_confidence":  news_sig.get("confidence", 0.5),
            })

        payload = {
            "week":                    week,
            "overall_news_sentiment":  news_output.get("overall_sentiment", 0),
            "overall_news_rationale":  news_output.get("overall_rationale", ""),
            "final_asset_signals":     combined,
            "portfolio":               portfolio,
            "debate_transcript":       transcript,
        }

        return (
            f"Week: {week}\n\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Based on the full debate transcript and the final signals, "
            "determine trading actions for all assets and return the JSON array."
        )

class DebateMAS:
    """
    Debate multi-agent system with structured adversarial argumentation.

    Args:
        debate_rounds: Number of back-and-forth debate rounds (default 2).
        memory_window: Past weekly outputs each agent sees in its prompt.
    """

    def __init__(
        self,
        capability: str = "zero_shot",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        memory_window: int = 4,
        debate_rounds: int = 2,
    ):
        llm_cap = _llm_capability(capability)
        if capability == "rag":
            crypto_cls = RAGCryptoAgent
        elif capability == "skill":
            crypto_cls = SkillCryptoAgent
        else:
            crypto_cls = CryptoAgent
        self.crypto_agent  = crypto_cls(        capability=llm_cap, model=model, temperature=temperature, memory_window=memory_window)
        self.news_agent    = NewsAgent(         capability=llm_cap, model=model, temperature=temperature, memory_window=memory_window)
        self.trading_agent = _DebateTradingAgent(capability=llm_cap, model=model, temperature=temperature, memory_window=memory_window)
        self.debate_rounds = debate_rounds
        self._rag_store    = _build_rag_store(capability)

    def run(
        self,
        week: str,
        indicators: list[dict],
        articles: list[dict],
        portfolio: dict,
    ) -> list[dict]:
        transcript: list[dict] = []
        rag_examples = _collect_rag_examples(self._rag_store, indicators, week=week)

        # ── Round 0: independent initial positions ────────────────────────
        logger.info("[Debate] Week %s — Round 0: CryptoAgent initial position", week)
        crypto_pos = self.crypto_agent.run(
            {"week": week, "assets": indicators, "rag_examples": rag_examples},
            memorize=False,
        )

        logger.info("[Debate] Week %s — Round 0: NewsAgent initial position", week)
        news_pos = self.news_agent.run(
            {"week": week, "articles": articles}, memorize=False
        )

        transcript.append({"round": 0, "crypto": crypto_pos, "news": news_pos})

        # ── Debate rounds ─────────────────────────────────────────────────
        for r in range(1, self.debate_rounds + 1):
            logger.info("[Debate] Week %s — Round %d: CryptoAgent rebuttal", week, r)
            crypto_ctx = _inject_debate_context(
                base_context={"week": week, "assets": indicators, "rag_examples": rag_examples},
                opponent_name="News Agent",
                opponent_output=news_pos,
                own_previous_output=crypto_pos,
                round_number=r,
            )
            crypto_pos = self.crypto_agent.run(crypto_ctx, memorize=False)

            logger.info("[Debate] Week %s — Round %d: NewsAgent rebuttal", week, r)
            news_ctx = _inject_debate_context(
                base_context={"week": week, "articles": articles},
                opponent_name="Crypto Agent",
                opponent_output=crypto_pos,
                own_previous_output=news_pos,
                round_number=r,
            )
            news_pos = self.news_agent.run(news_ctx, memorize=False)

            transcript.append({"round": r, "crypto": crypto_pos, "news": news_pos})

        # ── Commit final positions to agent memory ────────────────────────
        self.crypto_agent._store_memory(week, crypto_pos)
        self.news_agent._store_memory(week, news_pos)

        # ── TradingAgent judges the full debate ───────────────────────────
        logger.info("[Debate] Week %s — TradingAgent judging %d-round debate",
                    week, self.debate_rounds)
        actions = self.trading_agent.run({
            "week":              week,
            "debate_transcript": transcript,
            "portfolio":         portfolio,
        })

        return actions

    # ── Memory persistence ────────────────────────────────────────────────

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
