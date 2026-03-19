"""
Ablation Study — Experiment A: Agent Component Ablation.

Runs three ablation variants of the Hierarchical ZS baseline, each removing
exactly one system component to isolate its contribution:

    ablation_no_news     — CryptoAgent + TradingAgent only (NewsAgent removed)
    ablation_no_crypto   — NewsAgent + TradingAgent only (CryptoAgent removed)
    ablation_no_memory   — Full Hierarchical ZS, but memory_window=0

Output is written to the same processed_data/ tree used by run_experiment.py,
so evaluate.py and all evaluation plots work without modification:

    processed_data/ablation_no_news_zero_shot/   <YYYY-WNN>.json  ...
    processed_data/ablation_no_crypto_zero_shot/ <YYYY-WNN>.json  ...
    processed_data/ablation_no_memory_zero_shot/ <YYYY-WNN>.json  ...

Usage:
    python scripts/run_ablation.py                        # all three variants
    python scripts/run_ablation.py --variant no_news      # single variant
    python scripts/run_ablation.py --dry-run              # placeholder actions, no LLM calls
    python scripts/run_ablation.py --weeks 2025-W02 2025-W10
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from environ.architectures import AblationNoNews, AblationNoCrypto, AblationNoMemory
from environ.data.coingecko import load_asset, get_raw_snapshots_all, SYMBOL_TO_ID
from environ.data.cointelegraph import CointelegraphFetcher

# Reuse Portfolio, generate_weeks, week_sunday, get_execution_prices from run_experiment
from scripts.run_experiment import (
    Portfolio,
    generate_weeks,
    week_sunday,
    get_execution_prices,
    BACKTEST_START,
    BACKTEST_END,
    INITIAL_CASH,
    LOOKBACK_DAYS,
    OUTPUT_DIR,
    NEWS_DIR,
    OPENAI_MODEL,
    _dry_run_actions,
)

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_log_file = _LOG_DIR / f"ablation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

ABLATION_VARIANTS = {
    "no_news":    AblationNoNews,
    "no_crypto":  AblationNoCrypto,
    "no_memory":  AblationNoMemory,
}

# All ablation runs use zero_shot capability (Hierarchical ZS is the reference)
CAPABILITY = "zero_shot"
UNIVERSE = list(SYMBOL_TO_ID.keys())


def run_ablation_variant(
    variant_name: str,
    weeks: list[str],
    dry_run: bool = False,
) -> None:
    arch_cls  = ABLATION_VARIANTS[variant_name]
    combo_name = f"ablation_{variant_name}_{CAPABILITY}"
    out_dir    = OUTPUT_DIR / combo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Ablation variant: %s", combo_name)
    logger.info("Weeks: %d  |  Output: %s", len(weeks), out_dir)
    logger.info("=" * 60)

    if not dry_run:
        mas = arch_cls(capability=CAPABILITY, model=OPENAI_MODEL, temperature=0.0)

    news_fetcher = CointelegraphFetcher(output_dir=str(NEWS_DIR))
    portfolio    = Portfolio()
    state_path   = out_dir / "_state.json"

    if state_path.exists():
        saved = json.loads(state_path.read_text())
        portfolio.cash         = saved["portfolio"]["cash"]
        portfolio.holdings     = saved["portfolio"]["holdings"]
        portfolio.cost_basis   = saved["portfolio"].get("cost_basis", {})
        portfolio.initial_value = saved["portfolio"].get("initial_value", INITIAL_CASH)
        if not dry_run and "memory" in saved:
            mas.set_memory_state(saved["memory"])
        logger.info("Resumed from checkpoint: cash=%.2f", portfolio.cash)

    for week in weeks:
        out_path = out_dir / f"{week}.json"
        if out_path.exists():
            logger.info("[%s] %s — already done, skipping", combo_name, week)
            record = json.loads(out_path.read_text())
            after  = record["portfolio_after"]
            portfolio.cash     = after["cash"]
            portfolio.holdings = after.get("holdings_qty") or after.get("holdings", {})
            portfolio.cost_basis = after.get(
                "cost_basis",
                {k: v["cost_basis_usd"] for k, v in after.get("holdings_detail", {}).items()},
            )
            continue

        logger.info("[%s] %s — processing", combo_name, week)

        sunday = week_sunday(week)
        try:
            assets_data = get_raw_snapshots_all(sunday, lookback_days=LOOKBACK_DAYS)
        except Exception as exc:
            logger.error("Failed to load market data for %s: %s", week, exc)
            continue

        exec_prices      = get_execution_prices(sunday)
        articles         = news_fetcher.load_week(week)
        portfolio_before = portfolio.to_record(exec_prices)

        if dry_run:
            crypto_signals, news_output, trading_actions = _dry_run_actions()
        else:
            try:
                trading_actions = mas.run(
                    week=week,
                    indicators=assets_data,
                    articles=articles,
                    portfolio=portfolio.to_prompt_dict(exec_prices),
                )
                crypto_signals = []
                news_output    = {}
            except Exception as exc:
                logger.critical(
                    "[%s] %s — MAS failed: %s", combo_name, week, exc, exc_info=True,
                )
                raise SystemExit(1) from exc

        portfolio.apply_actions(trading_actions, exec_prices)
        portfolio_after = portfolio.to_record(exec_prices)

        record = {
            "week":             week,
            "architecture":     f"ablation_{variant_name}",
            "capability":       CAPABILITY,
            "portfolio_before": portfolio_before,
            "execution_prices": {k: round(v, 2) for k, v in exec_prices.items()},
            "crypto_signals":   crypto_signals,
            "news_output":      news_output,
            "trading_actions":  trading_actions,
            "portfolio_after":  portfolio_after,
        }
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        logger.info(
            "[%s] %s — saved (value: $%.2f | P&L: $%.2f / %.2f%%)",
            combo_name, week,
            portfolio_after["total_value"],
            portfolio_after["pnl_usd"],
            portfolio_after["pnl_pct"],
        )

        checkpoint = {
            "portfolio": {
                "cash":          portfolio.cash,
                "holdings":      portfolio.holdings,
                "cost_basis":    portfolio.cost_basis,
                "initial_value": portfolio.initial_value,
            },
            "memory": mas.get_memory_state() if not dry_run else {},
        }
        state_path.write_text(json.dumps(checkpoint, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run ablation study (Experiment A).")
    parser.add_argument(
        "--variant",
        choices=list(ABLATION_VARIANTS),
        default=None,
        help="Run only this variant (default: all three).",
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
        logger.info("Restricted to weeks %s – %s (%d weeks)", start_w, end_w, len(all_weeks))

    variants = [args.variant] if args.variant else list(ABLATION_VARIANTS)
    logger.info("Running %d ablation variant(s) × %d weeks", len(variants), len(all_weeks))

    for variant in variants:
        run_ablation_variant(variant, all_weeks, dry_run=args.dry_run)

    logger.info("Ablation study complete.")


if __name__ == "__main__":
    main()
