"""
Estimate OpenAI API token cost for one week of the MAS experiment.

For each architecture × capability combination the script builds the exact
prompts each agent would receive (using real market + news data) and counts
tokens with tiktoken.  Output tokens are estimated from the expected JSON
schema sizes; chain-of-thought variants add a reasoning-block estimate.

Usage:
    python scripts/estimate_cost.py
    python scripts/estimate_cost.py --week 2025-W04
    python scripts/estimate_cost.py --week 2025-W04 --memory-weeks 4
"""

import argparse
import json
import textwrap
from pathlib import Path

import tiktoken

from environ.architectures.base_arch import _inject_peer_context, _inject_debate_context
from environ.architectures.debate import _DebateTradingAgent
from environ.agents import CryptoAgent, NewsAgent, TradingAgent
from environ.data.coingecko import get_raw_snapshots_all, SYMBOL_TO_ID
from environ.data.cointelegraph import CointelegraphFetcher

# ── GPT-4o pricing (USD per 1M tokens, as of 2025) ──────────────────────────
INPUT_PRICE_PER_M  = 2.50
OUTPUT_PRICE_PER_M = 10.00

# ── Estimated output token counts (conservative) ─────────────────────────────
# Based on expected JSON schema × number of assets / fields.
OUTPUT_TOKENS = {
    "crypto":  650,   # 15 assets × ~43 tokens (symbol + signal + confidence + rationale)
    "news":    450,   # overall fields + ~10 coin signals
    "trading": 600,   # 15 assets × ~40 tokens (symbol + action + rationale)
}
# chain-of-thought adds a <reasoning>...</reasoning> block before the JSON
COT_REASONING_TOKENS = 700

UNIVERSE = list(SYMBOL_TO_ID.keys())
NEWS_DIR = Path("data/news")

LOOKBACK_DAYS = 30

# ── Realistic placeholder outputs (used as peer context in Collaborative) ────

_PLACEHOLDER_CRYPTO_SIGNALS = [
    {"symbol": s, "signal": 0.1, "confidence": 0.6,
     "rationale": "Mild bullish momentum with stable volume."}
    for s in UNIVERSE
]

_PLACEHOLDER_NEWS_OUTPUT = {
    "week": "placeholder",
    "overall_sentiment": 0.2,
    "overall_rationale": "Market broadly positive with no major negative events.",
    "coin_signals": [
        {"symbol": s, "signal": 0.15, "confidence": 0.55,
         "rationale": "Positive mentions in recent articles."}
        for s in ["BTC", "ETH", "SOL", "XRP"]
    ],
}

_PLACEHOLDER_PORTFOLIO = {
    "cash": 80_000.0,
    "total_value": 115_000.0,
    "pnl_usd": 15_000.0,
    "pnl_pct": 15.0,
    "holdings": {
        "BTC": {"value_usd": 20_000.0, "cost_basis_usd": 17_000.0,
                "pnl_usd": 3_000.0,   "pnl_pct": 17.65},
        "ETH": {"value_usd": 15_000.0, "cost_basis_usd": 14_000.0,
                "pnl_usd": 1_000.0,   "pnl_pct": 7.14},
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _week_sunday_ts(week_str: str):
    import pandas as pd
    year, w = week_str.split("-W")
    monday = pd.Timestamp.fromisocalendar(int(year), int(w), 1).tz_localize("UTC")
    return monday + pd.Timedelta(days=6)


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def build_memory_entries(agent_type: str, n_weeks: int) -> list[dict]:
    """Create n_weeks of synthetic memory entries for the agent."""
    if n_weeks == 0:
        return []
    if agent_type == "crypto":
        output = _PLACEHOLDER_CRYPTO_SIGNALS
    elif agent_type == "news":
        output = _PLACEHOLDER_NEWS_OUTPUT
    else:
        output = [{"symbol": s, "action": 0.0, "rationale": "hold"} for s in UNIVERSE]
    return [{"week": f"2025-W{i+1:02d}", "output": output} for i in range(n_weeks)]


def prompt_messages(agent, context: dict) -> tuple[str, str]:
    """Return (system, user) strings as the agent would send them."""
    system   = agent._decorate_system_prompt()
    user_msg = agent.build_user_message(context)
    if agent.memory and agent.memory_window > 0:
        user_msg = agent._format_memory() + "\n\n---\n\n" + user_msg
    return system, user_msg


# ── Per-architecture call breakdown ──────────────────────────────────────────

def calls_blackboard(cap, assets_data, articles, memory_weeks):
    """3 calls: Crypto → News → Trading."""
    crypto  = CryptoAgent( capability=cap, memory_window=memory_weeks)
    news    = NewsAgent(   capability=cap, memory_window=memory_weeks)
    trading = TradingAgent(capability=cap, memory_window=memory_weeks)

    crypto.memory  = build_memory_entries("crypto",  memory_weeks)
    news.memory    = build_memory_entries("news",     memory_weeks)
    trading.memory = build_memory_entries("trading",  memory_weeks)

    results = []
    results.append(("CryptoAgent",  cap,
                    prompt_messages(crypto,  {"week": "W", "assets": assets_data}),
                    "crypto"))
    results.append(("NewsAgent",    cap,
                    prompt_messages(news,    {"week": "W", "articles": articles}),
                    "news"))
    results.append(("TradingAgent", cap,
                    prompt_messages(trading, {
                        "week": "W",
                        "crypto_signals": _PLACEHOLDER_CRYPTO_SIGNALS,
                        "news_output":    _PLACEHOLDER_NEWS_OUTPUT,
                        "portfolio":      _PLACEHOLDER_PORTFOLIO,
                    }),
                    "trading"))
    return results


def calls_hierarchical(cap, assets_data, articles, memory_weeks):
    """Same 3 calls as Blackboard (supervisor framing only changes system prompt)."""
    from environ.architectures.hierarchical import _SupervisorTradingAgent
    crypto  = CryptoAgent(            capability=cap, memory_window=memory_weeks)
    news    = NewsAgent(              capability=cap, memory_window=memory_weeks)
    trading = _SupervisorTradingAgent(capability=cap, memory_window=memory_weeks)

    crypto.memory  = build_memory_entries("crypto",  memory_weeks)
    news.memory    = build_memory_entries("news",     memory_weeks)
    trading.memory = build_memory_entries("trading",  memory_weeks)

    results = []
    results.append(("CryptoAgent",           cap,
                    prompt_messages(crypto,  {"week": "W", "assets": assets_data}),
                    "crypto"))
    results.append(("NewsAgent",             cap,
                    prompt_messages(news,    {"week": "W", "articles": articles}),
                    "news"))
    results.append(("TradingAgent(sup)",     cap,
                    prompt_messages(trading, {
                        "week": "W",
                        "crypto_signals": _PLACEHOLDER_CRYPTO_SIGNALS,
                        "news_output":    _PLACEHOLDER_NEWS_OUTPUT,
                        "portfolio":      _PLACEHOLDER_PORTFOLIO,
                    }),
                    "trading"))
    return results


def calls_collaborative(cap, assets_data, articles, memory_weeks, refinement_rounds=1):
    """5 calls (1 refinement round): Crypto0, News0, CryptoR1, NewsR1, Trading."""
    crypto  = CryptoAgent( capability=cap, memory_window=memory_weeks)
    news    = NewsAgent(   capability=cap, memory_window=memory_weeks)
    trading = TradingAgent(capability=cap, memory_window=memory_weeks)

    crypto.memory  = build_memory_entries("crypto",  memory_weeks)
    news.memory    = build_memory_entries("news",     memory_weeks)
    trading.memory = build_memory_entries("trading",  memory_weeks)

    results = []

    # Round 0
    results.append(("CryptoAgent(r0)", cap,
                    prompt_messages(crypto, {"week": "W", "assets": assets_data}),
                    "crypto"))
    results.append(("NewsAgent(r0)",   cap,
                    prompt_messages(news,   {"week": "W", "articles": articles}),
                    "news"))

    # Refinement rounds
    for r in range(refinement_rounds):
        label = f"r{r+1}"
        crypto_ctx = _inject_peer_context(
            {"week": "W", "assets": assets_data},
            "News Agent", _PLACEHOLDER_NEWS_OUTPUT,
        )
        news_ctx = _inject_peer_context(
            {"week": "W", "articles": articles},
            "Crypto Agent", {"signals": _PLACEHOLDER_CRYPTO_SIGNALS},
        )
        results.append((f"CryptoAgent({label})", cap,
                        prompt_messages(crypto, crypto_ctx), "crypto"))
        results.append((f"NewsAgent({label})",   cap,
                        prompt_messages(news,   news_ctx),   "news"))

    # Final trading
    results.append(("TradingAgent", cap,
                    prompt_messages(trading, {
                        "week": "W",
                        "crypto_signals": _PLACEHOLDER_CRYPTO_SIGNALS,
                        "news_output":    _PLACEHOLDER_NEWS_OUTPUT,
                        "portfolio":      _PLACEHOLDER_PORTFOLIO,
                    }),
                    "trading"))
    return results


def calls_debate(cap, assets_data, articles, memory_weeks, debate_rounds=2):
    """5+ calls: Crypto0, News0, [CryptoRr, NewsRr] × rounds, DebateTradingAgent."""
    crypto  = CryptoAgent(        capability=cap, memory_window=memory_weeks)
    news    = NewsAgent(          capability=cap, memory_window=memory_weeks)
    trading = _DebateTradingAgent(capability=cap, memory_window=memory_weeks)

    crypto.memory  = build_memory_entries("crypto",  memory_weeks)
    news.memory    = build_memory_entries("news",     memory_weeks)
    trading.memory = build_memory_entries("trading",  memory_weeks)

    results = []

    crypto_pos = _PLACEHOLDER_CRYPTO_SIGNALS
    news_pos   = _PLACEHOLDER_NEWS_OUTPUT

    # Round 0
    results.append(("CryptoAgent(r0)", cap,
                    prompt_messages(crypto, {"week": "W", "assets": assets_data}),
                    "crypto"))
    results.append(("NewsAgent(r0)",   cap,
                    prompt_messages(news,   {"week": "W", "articles": articles}),
                    "news"))

    # Debate rounds
    for r in range(1, debate_rounds + 1):
        label = f"r{r}"
        crypto_ctx = _inject_debate_context(
            {"week": "W", "assets": assets_data},
            opponent_name="News Agent",
            opponent_output=news_pos,
            own_previous_output=crypto_pos,
            round_number=r,
        )
        news_ctx = _inject_debate_context(
            {"week": "W", "articles": articles},
            opponent_name="Crypto Agent",
            opponent_output=crypto_pos,
            own_previous_output=news_pos,
            round_number=r,
        )
        results.append((f"CryptoAgent({label})", cap,
                        prompt_messages(crypto, crypto_ctx), "crypto"))
        results.append((f"NewsAgent({label})",   cap,
                        prompt_messages(news,   news_ctx),   "news"))

    # Build a placeholder transcript for the trading agent
    transcript = [{"round": 0, "crypto": crypto_pos, "news": news_pos}]
    for r in range(1, debate_rounds + 1):
        transcript.append({"round": r, "crypto": crypto_pos, "news": news_pos})

    results.append(("TradingAgent(judge)", cap,
                    prompt_messages(trading, {
                        "week":             "W",
                        "debate_transcript": transcript,
                        "portfolio":        _PLACEHOLDER_PORTFOLIO,
                    }),
                    "trading"))
    return results


# ── Cost calculation ──────────────────────────────────────────────────────────

def estimate_call_cost(agent_label, cap, messages, output_type, enc):
    system, user = messages
    in_tokens = count_tokens(system, enc) + count_tokens(user, enc)
    out_tokens = OUTPUT_TOKENS[output_type]
    if cap == "chain_of_thought":
        out_tokens += COT_REASONING_TOKENS
    in_cost  = in_tokens  / 1_000_000 * INPUT_PRICE_PER_M
    out_cost = out_tokens / 1_000_000 * OUTPUT_PRICE_PER_M
    return {
        "agent":       agent_label,
        "capability":  cap,
        "in_tokens":   in_tokens,
        "out_tokens":  out_tokens,
        "in_cost":     in_cost,
        "out_cost":    out_cost,
        "total_cost":  in_cost + out_cost,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Estimate API token cost for one experiment week.")
    parser.add_argument("--week",         default="2025-W02",
                        help="ISO week to use for market data (default: 2025-W02)")
    parser.add_argument("--memory-weeks", type=int, default=0,
                        help="Simulated past memory entries per agent (default: 0 = first week)")
    args = parser.parse_args()

    enc = tiktoken.encoding_for_model("gpt-4o")

    # ── Load real data ────────────────────────────────────────────────────────
    print(f"Loading market data for week {args.week} …")
    sunday = _week_sunday_ts(args.week)
    try:
        assets_data = get_raw_snapshots_all(sunday, lookback_days=LOOKBACK_DAYS)
    except Exception as exc:
        print(f"  Warning: could not load market data ({exc}); using empty list.")
        assets_data = []

    news_fetcher = CointelegraphFetcher(output_dir=str(NEWS_DIR))
    articles = news_fetcher.load_week(args.week)
    if not articles:
        print(f"  Warning: no news articles found for {args.week}; using empty list.")

    CAPABILITIES = ["zero_shot", "chain_of_thought", "skill_augmented"]

    arch_calls = {
        "blackboard":    lambda cap: calls_blackboard(   cap, assets_data, articles, args.memory_weeks),
        "hierarchical":  lambda cap: calls_hierarchical( cap, assets_data, articles, args.memory_weeks),
        "collaborative": lambda cap: calls_collaborative(cap, assets_data, articles, args.memory_weeks),
        "debate":        lambda cap: calls_debate(       cap, assets_data, articles, args.memory_weeks),
    }

    # ── Estimate costs ────────────────────────────────────────────────────────
    all_rows = []
    arch_totals = {}

    for arch_name, call_fn in arch_calls.items():
        arch_cost = 0.0
        for cap in CAPABILITIES:
            calls = call_fn(cap)
            combo_cost = 0.0
            for agent_label, capability, messages, output_type in calls:
                row = estimate_call_cost(agent_label, capability, messages, output_type, enc)
                row["architecture"] = arch_name
                all_rows.append(row)
                combo_cost += row["total_cost"]
            arch_cost += combo_cost
        arch_totals[arch_name] = arch_cost

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"  Token cost estimate — week {args.week}  |  memory_weeks={args.memory_weeks}")
    print("=" * 90)

    # Per-combination summary
    print(f"\n{'Architecture':<16} {'Capability':<20} {'Calls':>5}  "
          f"{'In tok':>8}  {'Out tok':>8}  {'In $':>8}  {'Out $':>8}  {'Total $':>9}")
    print("-" * 90)

    for arch_name in arch_calls:
        for cap in CAPABILITIES:
            cap_rows = [r for r in all_rows
                        if r["architecture"] == arch_name and r["capability"] == cap]
            if not cap_rows:
                continue
            n_calls  = len(cap_rows)
            in_tok   = sum(r["in_tokens"]  for r in cap_rows)
            out_tok  = sum(r["out_tokens"] for r in cap_rows)
            in_cost  = sum(r["in_cost"]    for r in cap_rows)
            out_cost = sum(r["out_cost"]   for r in cap_rows)
            total    = in_cost + out_cost
            print(f"{arch_name:<16} {cap:<20} {n_calls:>5}  "
                  f"{in_tok:>8,}  {out_tok:>8,}  "
                  f"{in_cost:>8.4f}  {out_cost:>8.4f}  {total:>9.4f}")
        print()

    # Per-architecture subtotals
    print("-" * 90)
    grand_total = 0.0
    for arch_name, cost in arch_totals.items():
        print(f"  {arch_name:<30} subtotal:   ${cost:.4f}")
        grand_total += cost

    all_arches_per_week = grand_total
    print()
    print(f"  {'GRAND TOTAL (all 12 combinations × 1 week)':<44} ${grand_total:.4f}")
    print(f"  {'Estimated for full 52-week backtest':<44} ${grand_total * 52:.2f}")
    print()

    # Per-agent breakdown (averaged over capabilities)
    print("Per-agent average input token count (averaged over 3 capability variants):")
    for agent_key in ["CryptoAgent", "NewsAgent", "TradingAgent"]:
        rows = [r for r in all_rows if r["agent"].startswith(agent_key)]
        if not rows:
            continue
        avg_in = sum(r["in_tokens"] for r in rows) / len(rows)
        print(f"  {agent_key:<20} avg input tokens: {avg_in:,.0f}")

    print()
    print(f"GPT-4o pricing used: input=${INPUT_PRICE_PER_M}/M tokens, "
          f"output=${OUTPUT_PRICE_PER_M}/M tokens")
    print(f"Output token estimates: crypto={OUTPUT_TOKENS['crypto']}, "
          f"news={OUTPUT_TOKENS['news']}, trading={OUTPUT_TOKENS['trading']} "
          f"(+{COT_REASONING_TOKENS} for chain-of-thought reasoning)")
    print()


if __name__ == "__main__":
    main()
