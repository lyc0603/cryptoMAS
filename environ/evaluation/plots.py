"""
Visualisation functions for MAS backtest evaluation.

Each function accepts an output_dir (Path to processed_data/) and optional
save_path / show arguments. All return the Path where the figure was saved,
or None if show=True.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .metrics import load_all, compute_metrics, summary_table, INITIAL_CASH

# ── Style constants ───────────────────────────────────────────────────────────

ARCH_COLORS = {
    "blackboard":    "#2196F3",
    "hierarchical":  "#4CAF50",
    "collaborative": "#FF9800",
    "debate":        "#9C27B0",
    "benchmark":     "#607D8B",   # blue-grey for all benchmark strategies
}
CAP_STYLES = {
    "zero_shot":        "-",
    "chain_of_thought": "--",
    "skill_augmented":  ":",
    "rag":              "-.",
    # benchmark capabilities
    "btc_hold":         "-",
    "mcap_hold":        (0, (3, 1)),
    "lstm":             (0, (5, 1)),
    "informer":         (0, (4, 1, 1, 1)),
    "autoformer":       (0, (4, 1, 1, 1, 1, 1)),
    "timesnet":         (0, (3, 1, 1, 1, 1, 1)),
    "patchtst":         (0, (6, 1, 1, 1)),
    "sma7":             "-.",
    "slma":             ":",
    "macd":             (0, (5, 2, 1, 2)),   # dash-dot-dot
    "bb":               (0, (1, 1)),         # dense dots
}

def _arch_cap(combo_name: str) -> tuple[str, str]:
    for arch in ("blackboard", "hierarchical", "collaborative", "debate"):
        if combo_name.startswith(arch):
            return arch, combo_name[len(arch) + 1:]
    parts = combo_name.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (combo_name, "")


def _week_to_date(week_str: str) -> pd.Timestamp:
    year, w = week_str.split("-W")
    return pd.Timestamp.fromisocalendar(int(year), int(w), 1)


FIGURES_DIR = Path("figures")

def _save_or_show(fig, save_path, show: bool) -> Path | None:
    if show:
        plt.show()
        plt.close(fig)
        return None
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 1. Portfolio value timeseries ─────────────────────────────────────────────

def plot_portfolio(
    output_dir: Path,
    save_path: str | Path = "figures/portfolio_timeseries.pdf",
    show: bool = False,
) -> Path | None:
    """
    Line chart of portfolio total_value over time for every combination.
    Each architecture gets a distinct colour; each capability a distinct linestyle.
    Final P&L% is annotated at the end of each line.
    """
    combos = load_all(Path(output_dir))
    if not combos:
        print(f"No results found in {output_dir}")
        return None

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axhline(INITIAL_CASH, color="grey", linewidth=1, linestyle=":",
               label="Initial capital ($100k)", zorder=1)

    for combo_name, df in combos.items():
        arch, cap = _arch_cap(combo_name)
        color  = ARCH_COLORS.get(arch, "#333333")
        ls     = CAP_STYLES.get(cap, "-")
        label  = f"{arch} / {cap.replace('_', ' ')}"
        x      = [_week_to_date(w) for w in df.index]

        ax.plot(x, df["total_value"].values,
                color=color, linestyle=ls, linewidth=2,
                marker="o", markersize=3, label=label, zorder=2)

        last_val = df["total_value"].iloc[-1]
        pnl_pct  = (last_val - INITIAL_CASH) / INITIAL_CASH * 100
        sign     = "+" if pnl_pct >= 0 else ""
        ax.annotate(f"{sign}{pnl_pct:.1f}%",
                    xy=(x[-1], last_val),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, va="center")

    ax.set_title("Portfolio Value — All MAS Combinations", fontsize=14, pad=12)
    ax.set_xlabel("Week")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return _save_or_show(fig, save_path, show)


# ── 2. Drawdown timeseries ────────────────────────────────────────────────────

def plot_drawdown(
    output_dir: Path,
    save_path: str | Path = "figures/drawdown_timeseries.pdf",
    show: bool = False,
) -> Path | None:
    """
    Drawdown (%) from peak for every combination over time.
    Shaded area shows the magnitude of the underwater period.
    """
    combos = load_all(Path(output_dir))
    if not combos:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    for combo_name, df in combos.items():
        arch, cap = _arch_cap(combo_name)
        color  = ARCH_COLORS.get(arch, "#333333")
        ls     = CAP_STYLES.get(cap, "-")
        label  = f"{arch} / {cap.replace('_', ' ')}"
        x      = [_week_to_date(w) for w in df.index]
        tv     = df["total_value"]
        dd     = (tv - tv.cummax()) / tv.cummax() * 100

        ax.plot(x, dd.values, color=color, linestyle=ls, linewidth=1.5, label=label)
        ax.fill_between(x, dd.values, 0, color=color, alpha=0.06)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Drawdown — All MAS Combinations", fontsize=14, pad=12)
    ax.set_xlabel("Week")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return _save_or_show(fig, save_path, show)


# ── 3. Weekly returns distribution ───────────────────────────────────────────

def plot_weekly_returns(
    output_dir: Path,
    save_path: str | Path = "figures/weekly_returns.pdf",
    show: bool = False,
) -> Path | None:
    """
    Box plot of weekly returns (%) for each combination, sorted by median return.
    """
    combos = load_all(Path(output_dir))
    if not combos:
        return None

    labels, data, colors = [], [], []
    for combo_name, df in sorted(combos.items(),
                                  key=lambda kv: kv[1]["weekly_return"].median()):
        arch, cap = _arch_cap(combo_name)
        labels.append(f"{arch}\n{cap.replace('_', ' ')}")
        data.append(df["weekly_return"].dropna() * 100)
        colors.append(ARCH_COLORS.get(arch, "#333333"))

    fig, ax = plt.subplots(figsize=(max(8, len(combos) * 2), 6))
    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Weekly Return Distribution — All MAS Combinations", fontsize=14, pad=12)
    ax.set_ylabel("Weekly Return (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save_or_show(fig, save_path, show)


# ── 4. Metrics heatmap ────────────────────────────────────────────────────────

def plot_metrics_heatmap(
    output_dir: Path,
    save_path: str | Path = "figures/metrics_heatmap.pdf",
    show: bool = False,
) -> Path | None:
    """
    Heatmap comparing key metrics across all combinations (rows) normalised
    column-wise so each metric is colour-scaled independently.
    """
    tbl = summary_table(Path(output_dir))
    if tbl.empty:
        return None

    METRIC_COLS = [
        "total_return_pct", "annualized_return_pct", "annualized_vol_pct",
        "sharpe", "sortino", "max_drawdown_pct", "calmar", "win_rate_pct",
    ]
    METRIC_LABELS = [
        "Total Return %", "Ann. Return %", "Ann. Vol %",
        "Sharpe", "Sortino", "Max Drawdown %", "Calmar", "Win Rate %",
    ]
    # For drawdown a less-negative value is better → invert for colouring
    INVERT = {"max_drawdown_pct"}

    present = [c for c in METRIC_COLS if c in tbl.columns]
    sub = tbl[present].copy().astype(float)

    # Normalise each column to [0, 1] (higher = better)
    normed = pd.DataFrame(index=sub.index)
    for col in present:
        col_min, col_max = sub[col].min(), sub[col].max()
        rng = col_max - col_min
        if rng == 0:
            normed[col] = 0.5
        else:
            normed[col] = (sub[col] - col_min) / rng
            if col in INVERT:
                normed[col] = 1 - normed[col]

    fig, ax = plt.subplots(figsize=(len(present) * 1.4 + 2, len(sub) * 0.55 + 2))
    im = ax.imshow(normed.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(
        [METRIC_LABELS[METRIC_COLS.index(c)] for c in present],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub.index.tolist(), fontsize=9)

    # Annotate each cell with the raw value
    for r, row_name in enumerate(sub.index):
        for c, col in enumerate(present):
            val = sub.loc[row_name, col]
            txt = f"{val:.1f}" if val is not None and not np.isnan(val) else "—"
            ax.text(c, r, txt, ha="center", va="center", fontsize=8,
                    color="black" if 0.25 < normed.loc[row_name, col] < 0.85 else "white")

    ax.set_title("Performance Metrics — All MAS Combinations", fontsize=13, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Relative rank (green = better)")
    fig.tight_layout()
    return _save_or_show(fig, save_path, show)
