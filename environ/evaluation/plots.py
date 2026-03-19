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

# ── portfolio() style constants ───────────────────────────────────────────────
# Color encodes capability (SA + MAS); grey tones for benchmarks
_CAP_COLOR = {
    "zero_shot":        "#2196F3",   # blue
    "chain_of_thought": "#4CAF50",   # green
    "rag":              "#FF9800",   # orange
    "skill":            "#9C27B0",   # purple
}
_BM_HOLD_COLOR = "#444444"
_BM_DL_COLOR   = "#888888"

# Linestyle encodes MAS architecture; SA gets dash-dot
_ARCH_LS = {
    "hierarchical":  "-",
    "collaborative": "--",
    "debate":        ":",
}
_SA_LS = "-."

# Benchmark linestyles (so individual models are still distinguishable)
_BM_LS = {
    "btc_hold":  "-",
    "mcap_hold": "--",
    "lstm":      (0, (5, 1)),
    "informer":  (0, (4, 1, 1, 1)),
    "autoformer":(0, (4, 2)),
    "timesnet":  (0, (3, 1, 1, 1, 1, 1)),
    "patchtst":  (0, (6, 1, 2, 1)),
}

# Marker encodes group
_MARKER_GROUP = {
    "hold": "s",   # square
    "dl":   "^",   # triangle-up
    "sa":   "D",   # diamond
    "mas":  "o",   # circle
}

_BM_HOLD_CAPS = {"btc_hold", "mcap_hold"}
_BM_DL_CAPS   = {"lstm", "informer", "autoformer", "timesnet", "patchtst"}

_REGIME_COLOR = {
    "bull":     ("#c8e6c9", 0.55),   # light green
    "bear":     ("#ffcdd2", 0.65),   # light red/pink
    "sideways": ("#eeeeee", 0.50),   # light grey
}


def _group(arch: str, cap: str) -> str:
    if arch == "benchmark":
        return "hold" if cap in _BM_HOLD_CAPS else "dl"
    if arch == "single_agent":
        return "sa"
    return "mas"


def _classify_regimes(basket_vals: list[float]) -> list[str]:
    """Cagan (2024) ±20 % bull / bear / sideways classification."""
    if not basket_vals:
        return []
    peak = trough = basket_vals[0]
    out = []
    for v in basket_vals:
        peak   = max(peak, v)
        trough = min(trough, v)
        if v >= trough * 1.20:
            out.append("bull")
        elif v <= peak * 0.80:
            out.append("bear")
        else:
            out.append("sideways")
    return out


def _shade_regimes(ax, weeks: list[str], regimes: list[str]) -> None:
    """Draw contiguous bull / bear / sideways background bands."""
    dates = [_week_to_date(w) for w in weeks]
    i = 0
    while i < len(regimes):
        j = i + 1
        while j < len(regimes) and regimes[j] == regimes[i]:
            j += 1
        color, alpha = _REGIME_COLOR.get(regimes[i], ("#ffffff", 0))
        x0 = dates[i]
        x1 = dates[j - 1] + pd.Timedelta(days=7)
        ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=0, linewidth=0)
        i = j

def _arch_cap(combo_name: str) -> tuple[str, str]:
    for arch in ("blackboard", "hierarchical", "collaborative", "debate", "single_agent"):
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
    save_path: str | Path = "figures/portfolio.pdf",
    show: bool = False,
) -> Path | None:
    """
    Cumulative-return chart (indexed to 1.0) with bull/bear/sideways shading.

    Mimics the style of the reference figure:
      • Y-axis  : cumulative return ratio (1.0 = break-even)
      • Background: green = bull, pink = bear, grey = sideways
      • Dashed horizontal line at 1.0
      • Clean lines (no markers); end-of-line % annotations

    Visual encoding:
      • Color      → capability  (ZS=blue, CoT=green, RAG=orange, Skill=purple)
                     benchmarks  (Hold=dark-grey, DL=mid-grey)
      • Linestyle  → MAS architecture  (— Hier., -- Collab., ··· Debate)
                     Single Agent: -·-·   Benchmarks: model-specific
      • Linewidth  → MAS=2.0, SA=1.4, benchmarks=1.2
    Three legend boxes: Group/Architecture · Capability · Market Regime
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.dates as mdates

    combos = load_all(Path(output_dir))
    if not combos:
        print(f"No results found in {output_dir}")
        return None

    # ── Regime shading from mcap-hold basket ─────────────────────────────────
    basket_key = next((k for k in combos if "mcap_hold" in k), None)
    if basket_key:
        bdf          = combos[basket_key]
        regime_weeks = list(bdf.index)
        regimes      = _classify_regimes(bdf["total_value"].tolist())
    else:
        regime_weeks, regimes = [], []

    # Determine exact x bounds from the data
    all_dates = [_week_to_date(w) for df in combos.values() for w in df.index]
    x_min, x_max = min(all_dates), max(all_dates)

    fig, ax = plt.subplots(figsize=(14, 6))

    if regime_weeks:
        _shade_regimes(ax, regime_weeks, regimes)

    # Dashed break-even line
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--",
               alpha=0.6, zorder=2)

    global_min_tv = min(
        (df["total_value"].values / INITIAL_CASH).min()
        for df in combos.values()
    )

    for combo_name, df in combos.items():
        arch, cap = _arch_cap(combo_name)
        grp       = _group(arch, cap)
        x         = [_week_to_date(w) for w in df.index]
        tv        = df["total_value"].values / INITIAL_CASH   # normalise to 1.0

        if grp == "hold":
            color, ls, lw, alpha = _BM_HOLD_COLOR, _BM_LS.get(cap, "-"),   1.2, 0.90
        elif grp == "dl":
            color, ls, lw, alpha = _BM_DL_COLOR,   _BM_LS.get(cap, "-"),   1.2, 0.75
        elif grp == "sa":
            color, ls, lw, alpha = _CAP_COLOR.get(cap, "#333"), _SA_LS,     1.4, 0.80
        else:   # mas
            color = _CAP_COLOR.get(cap, "#333")
            ls    = _ARCH_LS.get(arch, "-")
            lw, alpha = 2.0, 1.0

        ax.plot(x, tv, color=color, linestyle=ls, linewidth=lw,
                alpha=alpha, zorder=3)

        # End-of-line annotation
        pct  = (tv[-1] - 1.0) * 100
        sign = "+" if pct >= 0 else ""
        ax.annotate(f"{sign}{pct:.0f}%",
                    xy=(x[-1], tv[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=6, va="center", color=color,
                    fontweight="bold", alpha=0.9)

    # ── Zoom inset: Feb–Jun 2025, x-axis proportionally aligned to main ───────
    zoom_start = pd.Timestamp("2025-02-01")
    zoom_end   = pd.Timestamp("2025-07-01")   # inclusive of June

    total_days = (x_max - x_min).days
    x0_frac    = (zoom_start - x_min).days / total_days
    width_frac = (zoom_end   - zoom_start).days / total_days

    # [x0, y0, width, height] in axes fraction — lower position
    axins = ax.inset_axes([x0_frac, 0.30, width_frac, 0.484])

    # Same regime shading clipped to zoom window
    if regime_weeks:
        zw = [w for w in regime_weeks
              if zoom_start <= _week_to_date(w) <= zoom_end]
        zr = [r for w, r in zip(regime_weeks, regimes)
              if zoom_start <= _week_to_date(w) <= zoom_end]
        _shade_regimes(axins, zw, zr)

    axins.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
                  alpha=0.5, zorder=2)

    zoom_tvs = []
    for combo_name, df in combos.items():
        arch, cap = _arch_cap(combo_name)
        grp       = _group(arch, cap)
        x_all     = [_week_to_date(w) for w in df.index]
        tv_all    = df["total_value"].values / INITIAL_CASH
        x_z  = [xi for xi in x_all if zoom_start <= xi <= zoom_end]
        tv_z = tv_all[[i for i, xi in enumerate(x_all) if zoom_start <= xi <= zoom_end]]
        if not x_z:
            continue
        zoom_tvs.extend(tv_z)

        if grp == "hold":
            color, ls, lw, alpha = _BM_HOLD_COLOR, _BM_LS.get(cap, "-"),   1.2, 0.90
        elif grp == "dl":
            color, ls, lw, alpha = _BM_DL_COLOR,   _BM_LS.get(cap, "-"),   1.2, 0.75
        elif grp == "sa":
            color, ls, lw, alpha = _CAP_COLOR.get(cap, "#333"), _SA_LS,     1.4, 0.80
        else:
            color = _CAP_COLOR.get(cap, "#333")
            ls    = _ARCH_LS.get(arch, "-")
            lw, alpha = 2.0, 1.0
        axins.plot(x_z, tv_z, color=color, linestyle=ls, linewidth=lw, alpha=alpha)

    # x-axis aligned with main: xlim matches the Feb–Jun portion exactly
    axins.set_xlim(zoom_start, zoom_end)
    if zoom_tvs:
        pad = 0.02
        axins.set_ylim(min(zoom_tvs) - pad, max(zoom_tvs) + pad)
    axins.xaxis.set_major_locator(mdates.MonthLocator())
    axins.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    axins.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    axins.tick_params(axis="both", labelsize=9)
    for lbl in axins.get_xticklabels() + axins.get_yticklabels():
        lbl.set_fontweight("bold")
    axins.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
    axins.grid(axis="x", linestyle=":",  alpha=0.20, zorder=1)
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    # ── Three legend boxes ────────────────────────────────────────────────────
    _g = "#555555"
    leg1_handles = [
        mlines.Line2D([], [], color=_g, ls="-",  lw=1.2, label="Hold"),
        mlines.Line2D([], [], color=_g, ls="-",  lw=1.2, label="Deep Learning"),
        mlines.Line2D([], [], color=_g, ls="-.", lw=1.4, label="Single Agent"),
        mlines.Line2D([], [], color=_g, ls="-",  lw=2.0, label="Hierarchical"),
        mlines.Line2D([], [], color=_g, ls="--", lw=2.0, label="Collaborative"),
        mlines.Line2D([], [], color=_g, ls=":",  lw=2.0, label="Debate"),
    ]
    leg2_handles = [
        mpatches.Patch(color=_CAP_COLOR["zero_shot"],        label="Zero-Shot"),
        mpatches.Patch(color=_CAP_COLOR["chain_of_thought"], label="Chain-of-Thought"),
        mpatches.Patch(color=_CAP_COLOR["rag"],              label="RAG"),
        mpatches.Patch(color=_CAP_COLOR["skill"],            label="Skill"),
    ]
    leg3_handles = [
        mpatches.Patch(color=c, alpha=a, label=r.capitalize())
        for r, (c, a) in _REGIME_COLOR.items()
    ]

    _leg_kw = dict(loc="upper left", frameon=False, fontsize=12,
                   title_fontsize=12, borderpad=0, handlelength=1.8,
                   columnspacing=1.0, handletextpad=0.5)

    def _bold_legend(leg):
        leg.get_title().set_fontweight("bold")
        for text in leg.get_texts():
            text.set_fontweight("bold")

    leg1 = ax.legend(handles=leg1_handles, ncol=len(leg1_handles),
                     bbox_to_anchor=(0.0, 1.00), title="Group / Architecture",
                     **_leg_kw)
    _bold_legend(leg1)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=leg2_handles, ncol=len(leg2_handles),
                     bbox_to_anchor=(0.0, 0.88), title="Capability",
                     **_leg_kw)
    _bold_legend(leg2)
    ax.add_artist(leg2)
    leg3 = ax.legend(handles=leg3_handles, ncol=len(leg3_handles),
                     bbox_to_anchor=(0.48, 0.88), title="Market Regime",
                     **_leg_kw)
    _bold_legend(leg3)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=global_min_tv)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))

    # x-axis: month ticks in "Mon'YY" style (e.g. Jan'25)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.tick_params(axis="x", labelsize=13, rotation=0)
    ax.tick_params(axis="y", labelsize=13)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
    ax.grid(axis="x", linestyle=":",  alpha=0.20, zorder=1)
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
