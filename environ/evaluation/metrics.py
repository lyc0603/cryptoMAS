"""
Performance metrics for MAS backtest results.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

INITIAL_CASH  = 100_000.0
WEEKS_PER_YEAR = 52.0


def load_combination(combo_dir: Path) -> pd.DataFrame:
    """
    Load all weekly JSON records for one combination directory.

    Returns a DataFrame indexed by ISO week string with columns:
        total_value, weekly_return, cash, pnl_usd, pnl_pct
    """
    rows = []
    for f in sorted(combo_dir.glob("20??-W??.json")):
        try:
            data = json.loads(f.read_text())
            after = data.get("portfolio_after", {})
            tv    = after.get("total_value")
            if tv is None:
                continue
            rows.append({
                "week":        data.get("week", f.stem),
                "total_value": tv,
                "cash":        after.get("cash", 0.0),
                "pnl_usd":     after.get("pnl_usd", tv - INITIAL_CASH),
                "pnl_pct":     after.get("pnl_pct", (tv - INITIAL_CASH) / INITIAL_CASH * 100),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("week")
    prev = pd.Series([INITIAL_CASH] + df["total_value"].iloc[:-1].tolist(),
                     index=df.index)
    df["weekly_return"] = (df["total_value"] - prev) / prev
    return df


def load_all(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all combinations from output_dir. Returns {combo_name: DataFrame}."""
    result = {}
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        df = load_combination(d)
        if not df.empty:
            result[d.name] = df
    return result


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute standard performance metrics from a combination DataFrame.

    Returns a dict with:
        total_return_pct, annualized_return_pct, annualized_vol_pct,
        sharpe, sortino, max_drawdown_pct, calmar, win_rate_pct,
        best_week_pct, worst_week_pct, n_weeks, final_value
    """
    tv = df["total_value"]
    wr = df["weekly_return"]
    n  = len(tv)

    total_return     = (tv.iloc[-1] - INITIAL_CASH) / INITIAL_CASH
    ann_return       = (1 + total_return) ** (WEEKS_PER_YEAR / n) - 1
    ann_vol          = wr.std() * np.sqrt(WEEKS_PER_YEAR)

    sharpe           = ann_return / ann_vol if ann_vol > 0 else np.nan

    downside         = wr[wr < 0]
    downside_std     = downside.std() * np.sqrt(WEEKS_PER_YEAR) if len(downside) > 1 else np.nan
    sortino          = ann_return / downside_std if (downside_std and downside_std > 0) else np.nan

    running_max      = tv.cummax()
    drawdown         = (tv - running_max) / running_max
    max_drawdown     = drawdown.min()
    calmar           = ann_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    win_rate         = (wr > 0).mean()

    return {
        "final_value":          round(tv.iloc[-1], 2),
        "total_return_pct":     round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return  * 100, 2),
        "annualized_vol_pct":   round(ann_vol      * 100, 2),
        "sharpe":               round(sharpe,  3) if not np.isnan(sharpe)  else None,
        "sortino":              round(sortino, 3) if not np.isnan(sortino) else None,
        "max_drawdown_pct":     round(max_drawdown * 100, 2),
        "calmar":               round(calmar, 3)  if not np.isnan(calmar)  else None,
        "win_rate_pct":         round(win_rate * 100, 1),
        "best_week_pct":        round(wr.max() * 100, 2),
        "worst_week_pct":       round(wr.min() * 100, 2),
        "n_weeks":              n,
    }


def summary_table(output_dir: Path) -> pd.DataFrame:
    """
    Build a metrics summary table for all combinations in output_dir.

    Returns a DataFrame with one row per combination and columns for each metric.
    """
    combos = load_all(output_dir)
    rows = []
    for name, df in combos.items():
        m = compute_metrics(df)
        parts = name.split("_", 1)
        arch  = parts[0] if len(parts) == 2 else name
        # Handle multi-word arch names (e.g. "chain_of_thought")
        for known_arch in ("blackboard", "hierarchical", "collaborative", "debate", "benchmark"):
            if name.startswith(known_arch):
                arch = known_arch
                cap  = name[len(known_arch) + 1:]
                break
        else:
            cap = parts[1] if len(parts) == 2 else ""
        rows.append({"combination": name, "architecture": arch, "capability": cap, **m})

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .set_index("combination")
        .sort_values("total_return_pct", ascending=False)
    )
