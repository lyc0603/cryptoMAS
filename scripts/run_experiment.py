"""
Experiment script — runs all MAS architecture × capability combinations
over the 2025 backtest period and saves weekly decision records.

For each (architecture, capability) combination the script iterates over
every ISO week, calls the relevant MAS, and writes:

    processed_data/<architecture>_<capability>/
        <YYYY-WNN>.json      ← one file per week

Each weekly JSON contains:
    week, architecture, capability,
    portfolio_before, portfolio_after,
    execution_prices,
    crypto_signals, news_output, trading_actions

Resume support: weeks whose output file already exists are skipped.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --arch hierarchical --cap zero_shot
    python scripts/run_experiment.py --arch hierarchical --cap zero_shot --weeks 2025-W02 2025-W05
    python scripts/run_experiment.py --dry-run   # skips LLM calls, saves placeholder actions
"""

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

from environ.architectures import (
    HierarchicalMAS,
    CollaborativeMAS,
    DebateMAS,
)
from environ.agents import SingleAgent
from environ.data.coingecko import load_asset, get_raw_snapshots_all, SYMBOL_TO_ID
from environ.data.cointelegraph import CointelegraphFetcher

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_log_file = _LOG_DIR / f"experiment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),  # terminal
        logging.FileHandler(_log_file, encoding="utf-8"),  # file
    ],
)
logger = logging.getLogger(__name__)
logger.info("Log file: %s", _log_file)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

class _SingleAgentWrapper:
    """
    Adapts SingleAgent to the MAS run() interface used by run_combination():
        mas.run(week, indicators, articles, portfolio)
    """

    def __init__(self, capability: str, model: str, temperature: float):
        from environ.architectures.base_arch import _llm_capability
        self._agent = SingleAgent(
            capability=_llm_capability(capability),
            model=model,
            temperature=temperature,
        )

    def run(self, week: str, indicators: list, articles: list, portfolio: dict) -> list:
        context = {
            "week": week,
            "assets": indicators,
            "articles": articles,
            "portfolio": portfolio,
        }
        return self._agent.run(context)

    def get_memory_state(self) -> list:
        return self._agent.get_memory_state()

    def set_memory_state(self, state: list) -> None:
        self._agent.set_memory_state(state)


ARCHITECTURES = {
    "hierarchical": HierarchicalMAS,
    "collaborative": CollaborativeMAS,
    "debate": DebateMAS,
    "single_agent": _SingleAgentWrapper,
}

CAPABILITIES = ["zero_shot", "chain_of_thought", "rag", "skill"]

UNIVERSE = list(SYMBOL_TO_ID.keys())

BACKTEST_START = "2025-01-01"
BACKTEST_END = "2026-01-01"
INITIAL_CASH = 100_000.0
TRANSACTION_COST = 0.001  # 0.1% per trade (applied to both buys and sells)
LOOKBACK_DAYS = 30  # days of market data fed to the Crypto Agent
OUTPUT_DIR = Path("processed_data")
NEWS_DIR = Path("data/news")
OPENAI_MODEL = "gpt-4o"


# ------------------------------------------------------------------
# Portfolio
# ------------------------------------------------------------------


@dataclass
class Portfolio:
    cash: float = INITIAL_CASH
    holdings: dict = field(default_factory=dict)  # symbol → quantity (crypto units)
    cost_basis: dict = field(
        default_factory=dict
    )  # symbol → USD paid (for unrealised P&L)
    initial_value: float = INITIAL_CASH  # reference for overall P&L

    def total_value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(
            self.holdings.get(sym, 0.0) * prices.get(sym, 0.0) for sym in UNIVERSE
        )

    def _overall_pnl(self, prices: dict[str, float]) -> tuple[float, float]:
        tv = self.total_value(prices)
        abs_pnl = tv - self.initial_value
        pct_pnl = abs_pnl / self.initial_value * 100.0
        return round(abs_pnl, 2), round(pct_pnl, 4)

    def _per_asset_detail(self, prices: dict[str, float]) -> dict:
        """Per-asset breakdown: current value, cost basis, unrealised P&L."""
        detail = {}
        for sym in UNIVERSE:
            qty = self.holdings.get(sym, 0.0)
            price = prices.get(sym, 0.0)
            if qty <= 0:
                continue
            value = round(qty * price, 2)
            basis = round(self.cost_basis.get(sym, 0.0), 2)
            pnl = round(value - basis, 2)
            pct = round(pnl / basis * 100.0, 4) if basis > 0 else 0.0
            detail[sym] = {
                "value_usd": value,
                "cost_basis_usd": basis,
                "pnl_usd": pnl,
                "pnl_pct": pct,
            }
        return detail

    def to_prompt_dict(self, prices: dict[str, float]) -> dict:
        abs_pnl, pct_pnl = self._overall_pnl(prices)
        return {
            "cash": round(self.cash, 2),
            "total_value": round(self.total_value(prices), 2),
            "pnl_usd": abs_pnl,
            "pnl_pct": pct_pnl,
            "holdings": self._per_asset_detail(prices),
        }

    def to_record(self, prices: dict[str, float]) -> dict:
        abs_pnl, pct_pnl = self._overall_pnl(prices)
        return {
            "cash": round(self.cash, 2),
            "holdings_qty": {k: round(v, 8) for k, v in self.holdings.items()},
            "holdings_detail": self._per_asset_detail(prices),
            "total_value": round(self.total_value(prices), 2),
            "pnl_usd": abs_pnl,
            "pnl_pct": pct_pnl,
        }

    def apply_actions(self, actions: list[dict], prices: dict[str, float]) -> None:
        """
        Apply trading actions in-place.

        Execution order: ALL sells first → then ALL buys from the post-sell cash pool.

        action > 0: allocate that fraction of post-sell cash to buy this asset
        action < 0: sell that fraction of current holdings
        action = 0: hold

        Buy fractions use the same post-sell cash as the base, so the order of
        buy actions does not matter.  If total desired spend exceeds available
        cash the fractions are scaled down proportionally.
        """
        # ── Phase 1: execute all sells ────────────────────────────────────
        for item in actions:
            sym = item.get("symbol")
            action = float(item.get("action", 0.0))
            price = prices.get(sym)
            if price is None or price <= 0 or action >= 0:
                continue
            sell_qty = abs(action) * self.holdings.get(sym, 0.0)
            if sell_qty <= 0:
                continue
            gross_usd = sell_qty * price
            net_usd = gross_usd * (1 - TRANSACTION_COST)
            self.holdings[sym] = self.holdings.get(sym, 0.0) - sell_qty
            self.cash += net_usd
            # Reduce cost basis proportionally to the fraction sold
            if sym in self.cost_basis:
                self.cost_basis[sym] *= 1.0 - abs(action)

        # ── Phase 2: allocate buys from the post-sell cash pool ───────────
        post_sell_cash = self.cash
        buy_items = [
            (item, float(item.get("action", 0.0)))
            for item in actions
            if float(item.get("action", 0.0)) > 0
            and prices.get(item.get("symbol"), 0) > 0
        ]
        if not buy_items:
            return

        # Desired spend for each asset = fraction × post_sell_cash (same base)
        desired: dict[str, float] = {
            item["symbol"]: action * post_sell_cash for item, action in buy_items
        }
        total_desired = sum(desired.values())

        # Scale down proportionally if the agent over-allocates
        scale = min(1.0, post_sell_cash / total_desired) if total_desired > 0 else 0.0

        for item, _ in buy_items:
            sym = item["symbol"]
            price = prices[sym]
            spend_usd = desired[sym] * scale
            if spend_usd <= 0:
                continue
            net_usd = spend_usd * (1 - TRANSACTION_COST)
            quantity = net_usd / price
            self.cash -= spend_usd
            self.holdings[sym] = self.holdings.get(sym, 0.0) + quantity
            self.cost_basis[sym] = self.cost_basis.get(sym, 0.0) + spend_usd


# ------------------------------------------------------------------
# Week helpers
# ------------------------------------------------------------------


def generate_weeks(start: str, end: str) -> list[str]:
    """Return ISO week strings ('YYYY-WNN') for all Mondays in [start, end)."""
    mondays = pd.date_range(start, end, freq="W-MON")
    return [f"{d.isocalendar().year}-W{d.isocalendar().week:02d}" for d in mondays]


def week_monday(week_str: str) -> pd.Timestamp:
    """Return the Monday of an ISO week string as a UTC Timestamp."""
    year, w = week_str.split("-W")
    return pd.Timestamp.fromisocalendar(int(year), int(w), 1).tz_localize("UTC")


def week_sunday(week_str: str) -> pd.Timestamp:
    """Return the Sunday (last day) of an ISO week string as a UTC Timestamp."""
    return week_monday(week_str) + pd.Timedelta(days=6)


def get_execution_prices(as_of: pd.Timestamp) -> dict[str, float]:
    """
    Return the close price for each asset on `as_of` date.
    Falls back to the most recent available close if the exact date is missing.
    """
    prices = {}
    for sym in UNIVERSE:
        df = load_asset(sym)
        row = df.loc[:as_of]["close"].dropna()
        prices[sym] = float(row.iloc[-1]) if not row.empty else 0.0
    return prices


# ------------------------------------------------------------------
# Dry-run stub
# ------------------------------------------------------------------


def _dry_run_actions() -> tuple[list[dict], dict, dict]:
    """Return placeholder outputs without calling any LLM."""
    crypto_signals = [
        {"symbol": s, "signal": 0.0, "confidence": 0.5, "rationale": "dry-run"}
        for s in UNIVERSE
    ]
    news_output = {
        "week": None,
        "overall_sentiment": 0.0,
        "overall_rationale": "dry-run",
        "coin_signals": [],
    }
    trading_actions = [
        {"symbol": s, "action": 0.0, "rationale": "dry-run"} for s in UNIVERSE
    ]
    return crypto_signals, news_output, trading_actions


# ------------------------------------------------------------------
# Single-combination runner
# ------------------------------------------------------------------


def run_combination(
    arch_name: str,
    capability: str,
    weeks: list[str],
    dry_run: bool = False,
) -> None:
    combo_name = f"{arch_name}_{capability}"
    out_dir = OUTPUT_DIR / combo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Combination: %s", combo_name)
    logger.info("Weeks: %d  |  Output: %s", len(weeks), out_dir)
    logger.info("=" * 60)

    # Instantiate MAS (shared across weeks for this combination)
    if not dry_run:
        mas = ARCHITECTURES[arch_name](
            capability=capability,
            model=OPENAI_MODEL,
            temperature=0.0,
        )

    # News loader
    news_fetcher = CointelegraphFetcher(output_dir=str(NEWS_DIR))

    # Portfolio + memory (check for saved state to support resume)
    portfolio = Portfolio()
    state_path = out_dir / "_state.json"
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        portfolio.cash = saved["portfolio"]["cash"]
        portfolio.holdings = saved["portfolio"]["holdings"]
        portfolio.cost_basis = saved["portfolio"].get("cost_basis", {})
        portfolio.initial_value = saved["portfolio"].get("initial_value", INITIAL_CASH)
        if not dry_run and "memory" in saved:
            mas.set_memory_state(saved["memory"])
        logger.info("Resumed portfolio: cash=%.2f", portfolio.cash)

    for week in weeks:
        out_path = out_dir / f"{week}.json"
        if out_path.exists():
            logger.info("[%s] %s — already done, skipping", combo_name, week)
            # Restore portfolio from saved record so subsequent weeks are consistent
            record = json.loads(out_path.read_text())
            after = record["portfolio_after"]
            portfolio.cash = after["cash"]
            # support both old key ("holdings") and new key ("holdings_qty")
            portfolio.holdings = after.get("holdings_qty") or after.get("holdings", {})
            portfolio.cost_basis = after.get(
                "cost_basis",
                {
                    k: v["cost_basis_usd"]
                    for k, v in after.get("holdings_detail", {}).items()
                },
            )
            continue

        logger.info("[%s] %s — processing", combo_name, week)

        # --- Market data ---
        sunday = week_sunday(week)
        try:
            assets_data = get_raw_snapshots_all(sunday, lookback_days=LOOKBACK_DAYS)
        except Exception as exc:
            logger.error("Failed to load market data for %s: %s", week, exc)
            continue

        # --- Execution prices (Sunday close of this week) ---
        exec_prices = get_execution_prices(sunday)

        # --- News articles ---
        articles = news_fetcher.load_week(week)
        if not articles:
            logger.warning("[%s] %s — no news articles found", combo_name, week)

        # --- Portfolio snapshot before decisions ---
        portfolio_before = portfolio.to_record(exec_prices)

        # --- Run MAS ---
        if dry_run:
            crypto_signals, news_output, trading_actions = _dry_run_actions()
        else:
            try:
                trading_actions = mas.run(
                    week=week,
                    indicators=assets_data,  # raw price/vol/mcap — see CryptoAgent
                    articles=articles,
                    portfolio=portfolio.to_prompt_dict(exec_prices),
                )
                # For non-blackboard architectures the intermediate signals are
                # not directly returned by mas.run(); we re-run agents individually
                # for record-keeping only if this is Blackboard (has .blackboard attr)
                crypto_signals = getattr(mas, "blackboard", {}).get(
                    "crypto_signals", []
                )
                news_output = getattr(mas, "blackboard", {}).get("news_output", {})
            except Exception as exc:
                logger.critical(
                    "[%s] %s — MAS failed: %s\n"
                    "Terminating to prevent wrong results or contaminated memory.",
                    combo_name,
                    week,
                    exc,
                    exc_info=True,
                )
                raise SystemExit(1) from exc

        # --- Apply actions to portfolio ---
        portfolio.apply_actions(trading_actions, exec_prices)
        portfolio_after = portfolio.to_record(exec_prices)

        # --- Save record ---
        record = {
            "week": week,
            "architecture": arch_name,
            "capability": capability,
            "portfolio_before": portfolio_before,
            "execution_prices": {k: round(v, 2) for k, v in exec_prices.items()},
            "crypto_signals": crypto_signals,
            "news_output": news_output,
            "trading_actions": trading_actions,
            "portfolio_after": portfolio_after,
        }
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        logger.info(
            "[%s] %s — saved (portfolio value: $%.2f | P&L: $%.2f / %.2f%%)",
            combo_name,
            week,
            portfolio_after["total_value"],
            portfolio_after["pnl_usd"],
            portfolio_after["pnl_pct"],
        )

        # Persist portfolio + memory state for crash recovery
        checkpoint = {
            "portfolio": {
                "cash": portfolio.cash,
                "holdings": portfolio.holdings,
                "cost_basis": portfolio.cost_basis,
                "initial_value": portfolio.initial_value,
            },
            "memory": mas.get_memory_state() if not dry_run else {},
        }
        state_path.write_text(json.dumps(checkpoint, indent=2))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run MAS backtest experiments.")
    parser.add_argument(
        "--arch",
        choices=list(ARCHITECTURES),
        default=None,
        help="Run only this architecture (default: all MAS; use 'single_agent' for the LLM baseline).",
    )
    parser.add_argument(
        "--cap",
        choices=CAPABILITIES,
        default=None,
        help="Run only this capability (default: all).",
    )
    parser.add_argument(
        "--weeks",
        nargs=2,
        metavar=("START_WEEK", "END_WEEK"),
        default=None,
        help="Restrict to a range of ISO weeks, e.g. --weeks 2025-W02 2025-W10",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; write placeholder hold actions.",
    )
    args = parser.parse_args()

    all_weeks = generate_weeks(BACKTEST_START, BACKTEST_END)

    if args.weeks:
        start_w, end_w = args.weeks
        all_weeks = [w for w in all_weeks if start_w <= w <= end_w]
        logger.info(
            "Restricted to weeks %s – %s (%d weeks)", start_w, end_w, len(all_weeks)
        )

    arch_list = [args.arch] if args.arch else list(ARCHITECTURES)
    cap_list = [args.cap] if args.cap else CAPABILITIES

    total = len(arch_list) * len(cap_list)
    logger.info("Running %d combination(s) × %d weeks", total, len(all_weeks))

    for arch_name in arch_list:
        for capability in cap_list:
            run_combination(arch_name, capability, all_weeks, dry_run=args.dry_run)

    logger.info("All combinations complete.")


if __name__ == "__main__":
    main()
