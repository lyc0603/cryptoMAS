"""
Ablation architectures for Experiment A.

All three variants are based on Hierarchical ZS as the reference.
Each removes exactly one system component to isolate its contribution.

Variants
--------
AblationNoNews      — CryptoAgent + TradingAgent only (NewsAgent removed)
AblationNoCrypto    — NewsAgent + TradingAgent only (CryptoAgent removed)
AblationNoMemory    — Full Hierarchical ZS with memory_window=0
"""

import logging

from environ.agents import CryptoAgent, NewsAgent, TradingAgent
from environ.agents.trading_agent import UNIVERSE
from .base_arch import _build_rag_store, _collect_rag_examples, _llm_capability
from .hierarchical import _SupervisorTradingAgent

logger = logging.getLogger(__name__)

# Placeholder outputs used when an agent is disabled
_NULL_NEWS = {
    "overall_sentiment": 0.0,
    "overall_rationale": "",
    "coin_signals": [],
}

_NULL_CRYPTO = [
    {
        "symbol": sym,
        "signal": 0.0,
        "confidence": 0.0,
        "rationale": "",
    }
    for sym in UNIVERSE
]


class AblationNoNews:
    """
    Hierarchical ZS without NewsAgent.
    TradingAgent receives only CryptoAgent signals; news input is zeroed out.
    """

    def __init__(
        self,
        capability: str = "zero_shot",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        memory_window: int = 4,
    ):
        llm_cap = _llm_capability(capability)
        self.crypto_agent  = CryptoAgent(capability=llm_cap, model=model,
                                         temperature=temperature, memory_window=memory_window)
        self.trading_agent = _SupervisorTradingAgent(capability=llm_cap, model=model,
                                                     temperature=temperature, memory_window=memory_window)

    def run(self, week: str, indicators: list[dict],
            articles: list[dict], portfolio: dict) -> list[dict]:
        logger.info("[Ablation:NoNews] Week %s — CryptoAgent", week)
        crypto_signals = self.crypto_agent.run({
            "week": week, "assets": indicators, "rag_examples": {},
        })

        logger.info("[Ablation:NoNews] Week %s — TradingAgent (no news)", week)
        return self.trading_agent.run({
            "week": week,
            "crypto_signals": crypto_signals,
            "news_output": _NULL_NEWS,
            "portfolio": portfolio,
        })

    def get_memory_state(self) -> dict:
        return {
            "crypto_agent":  self.crypto_agent.get_memory_state(),
            "trading_agent": self.trading_agent.get_memory_state(),
        }

    def set_memory_state(self, state: dict) -> None:
        self.crypto_agent.set_memory_state( state.get("crypto_agent",  []))
        self.trading_agent.set_memory_state(state.get("trading_agent", []))


class AblationNoCrypto:
    """
    Hierarchical ZS without CryptoAgent.
    TradingAgent receives only NewsAgent signals; crypto input is zeroed out.
    """

    def __init__(
        self,
        capability: str = "zero_shot",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        memory_window: int = 4,
    ):
        llm_cap = _llm_capability(capability)
        self.news_agent    = NewsAgent(capability=llm_cap, model=model,
                                       temperature=temperature, memory_window=memory_window)
        self.trading_agent = _SupervisorTradingAgent(capability=llm_cap, model=model,
                                                     temperature=temperature, memory_window=memory_window)

    def run(self, week: str, indicators: list[dict],
            articles: list[dict], portfolio: dict) -> list[dict]:
        logger.info("[Ablation:NoCrypto] Week %s — NewsAgent", week)
        news_output = self.news_agent.run({"week": week, "articles": articles})

        logger.info("[Ablation:NoCrypto] Week %s — TradingAgent (no crypto)", week)
        return self.trading_agent.run({
            "week": week,
            "crypto_signals": _NULL_CRYPTO,
            "news_output": news_output,
            "portfolio": portfolio,
        })

    def get_memory_state(self) -> dict:
        return {
            "news_agent":    self.news_agent.get_memory_state(),
            "trading_agent": self.trading_agent.get_memory_state(),
        }

    def set_memory_state(self, state: dict) -> None:
        self.news_agent.set_memory_state(   state.get("news_agent",    []))
        self.trading_agent.set_memory_state(state.get("trading_agent", []))


class AblationNoMemory:
    """
    Full Hierarchical ZS with memory_window=0 (agents have no access to past outputs).
    """

    def __init__(
        self,
        capability: str = "zero_shot",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        memory_window: int = 4,   # accepted but ignored — always 0
    ):
        llm_cap = _llm_capability(capability)
        self.crypto_agent  = CryptoAgent(capability=llm_cap, model=model,
                                         temperature=temperature, memory_window=0)
        self.news_agent    = NewsAgent(capability=llm_cap, model=model,
                                       temperature=temperature, memory_window=0)
        self.trading_agent = _SupervisorTradingAgent(capability=llm_cap, model=model,
                                                     temperature=temperature, memory_window=0)
        self._rag_store    = _build_rag_store(capability)

    def run(self, week: str, indicators: list[dict],
            articles: list[dict], portfolio: dict) -> list[dict]:
        logger.info("[Ablation:NoMemory] Week %s — CryptoAgent", week)
        crypto_signals = self.crypto_agent.run({
            "week": week,
            "assets": indicators,
            "rag_examples": _collect_rag_examples(self._rag_store, indicators, week=week),
        })

        logger.info("[Ablation:NoMemory] Week %s — NewsAgent", week)
        news_output = self.news_agent.run({"week": week, "articles": articles})

        logger.info("[Ablation:NoMemory] Week %s — TradingAgent", week)
        return self.trading_agent.run({
            "week": week,
            "crypto_signals": crypto_signals,
            "news_output": news_output,
            "portfolio": portfolio,
        })

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
