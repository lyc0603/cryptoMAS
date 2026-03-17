"""
Shared utilities for MAS architectures.
"""

import json
import logging

logger = logging.getLogger(__name__)


def _llm_capability(capability: str) -> str:
    """Normalise portfolio-level capabilities to their base LLM capability.

    'rag'   — retrieval-augmented generation; base LLM is zero_shot.
    'skill' — Liu et al. (2022) factor-augmented; base LLM is zero_shot.
    Individual agents receive the underlying LLM capability.
    """
    if capability in ("rag", "skill"):
        return "zero_shot"
    return capability


def _build_rag_store(capability: str, rag_end_date: str = "2026-01-01"):
    """Return a built RAGStore when capability is 'rag', else None."""
    if capability != "rag":
        return None
    from environ.data.rag_store import RAGStore
    logger.info("Building RAG store (history before %s)…", rag_end_date)
    store = RAGStore()
    store.build(end_date=rag_end_date)
    logger.info("RAG store ready.")
    return store


def _collect_rag_examples(
    rag_store,
    indicators: list[dict],
    top_k: int = 3,
    week: str | None = None,
) -> dict[str, str]:
    """Query the RAG store for each asset and return {sym: formatted_text}.

    `week` is the current ISO week string ("YYYY-Www"). When provided, only
    store entries strictly before this week are eligible (no lookahead).
    """
    if not rag_store:
        return {}
    result = {}
    for snap in indicators:
        sym  = snap["symbol"]
        text = rag_store.format_examples(sym, snap, top_k=top_k, as_of_week=week)
        if text:
            result[sym] = text
            logger.debug("RAG [%s]:\n%s", sym, text)
        else:
            store = rag_store._stores.get(sym)
            if store is None or len(store["entries"]) == 0:
                reason = "no historical entries in store"
            else:
                reason = "current snapshot has insufficient data for feature extraction"
            logger.warning("RAG [%s]: skipped — %s", sym, reason)
    logger.info("RAG retrieved analogues for %d/%d assets", len(result), len(indicators))
    return result


def _inject_peer_context(base_context: dict, peer_name: str, peer_output: dict) -> dict:
    """
    Add a peer agent's output into a context dict so the receiving agent
    can refine its analysis.

    A 'peer_context' key is added containing a serialised summary.
    """
    ctx = dict(base_context)
    ctx["peer_context"] = {
        "from": peer_name,
        "output": peer_output,
        "instruction": (
            f"The {peer_name} has already produced the above output. "
            "Review it and, if relevant, revise your own analysis accordingly. "
            "Your final output must still follow the same JSON schema as before."
        ),
    }
    return ctx


def _inject_debate_context(
    base_context: dict,
    opponent_name: str,
    opponent_output,
    own_previous_output,
    round_number: int,
) -> dict:
    """
    Build a debate-round context for an agent.

    The agent sees its opponent's latest argument and its own previous position,
    and is asked to explicitly challenge or defend.
    """
    ctx = dict(base_context)
    ctx["debate"] = {
        "round": round_number,
        "your_previous_position": own_previous_output,
        "opponent": {
            "name": opponent_name,
            "argument": opponent_output,
        },
        "instruction": (
            f"Debate round {round_number}: The {opponent_name} has presented the argument above. "
            "Your task:\n"
            "1. Critically evaluate their position — identify specific points you agree with "
            "and explicitly challenge points you disagree with, citing evidence from the data.\n"
            "2. Produce your updated signal. You may reinforce or revise your previous stance, "
            "but you MUST justify any change or maintained disagreement in the rationale field.\n"
            "Your output must still follow the same JSON schema as before."
        ),
    }
    return ctx
