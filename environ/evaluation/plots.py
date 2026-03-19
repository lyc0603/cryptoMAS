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



# ── Helper: display labels ────────────────────────────────────────────────────

_ARCH_SHORT = {
    "hierarchical":  "Hier.",
    "collaborative": "Collab.",
    "debate":        "Debate",
    "single_agent":  "SA",
    "benchmark":     "",
}
_CAP_SHORT = {
    "zero_shot":        "ZS",
    "chain_of_thought": "CoT",
    "rag":              "RAG",
    "skill":            "Skill",
    "btc_hold":         "BTC Hold",
    "mcap_hold":        "MCap Hold",
    "lstm":             "LSTM",
    "informer":         "Informer",
    "autoformer":       "Autoformer",
    "timesnet":         "TimesNet",
    "patchtst":         "PatchTST",
}

def _display_name(combo_name: str) -> str:
    arch, cap = _arch_cap(combo_name)
    a = _ARCH_SHORT.get(arch, arch.title())
    c = _CAP_SHORT.get(cap, cap.replace("_", " ").title())
    return c if arch == "benchmark" else f"{a} ({c})"


def _strategy_style(combo_name: str) -> tuple[str, str, str]:
    """Return (color, marker, edgecolor) for scatter/bar plots."""
    arch, cap = _arch_cap(combo_name)
    grp = _group(arch, cap)
    if grp == "hold":
        color = _BM_HOLD_COLOR
    elif grp == "dl":
        color = _BM_DL_COLOR
    else:
        color = _CAP_COLOR.get(cap, "#333333")
    marker = _MARKER_GROUP.get(grp, "o")
    return color, marker, "white"


def _load_basket(output_dir: Path):
    """Load mcap_hold basket for regime classification; returns (weeks, regimes) or ([], [])."""
    combos = load_all(output_dir)
    basket_key = next((k for k in combos if "mcap_hold" in k), None)
    if basket_key is None:
        return [], []
    df = combos[basket_key]
    weeks   = list(df.index)
    regimes = _classify_regimes(df["total_value"].tolist())
    return weeks, regimes


# ── 2. Risk-return scatter ────────────────────────────────────────────────────

def plot_risk_return(
    output_dir: Path,
    save_path: str | Path = "figures/risk_return.pdf",
    show: bool = False,
) -> Path | None:
    """
    Scatter of annualised volatility (x) vs cumulative return (y).
    Color = capability; marker = group (Hold / DL / SA / MAS).
    """
    combos = load_all(Path(output_dir))
    if not combos:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot order: benchmarks first (so MAS dots sit on top)
    ordered = sorted(combos.items(),
                     key=lambda kv: 0 if _group(*_arch_cap(kv[0])) in ("hold", "dl") else 1)

    points = {}
    for combo_name, df in ordered:
        arch, cap = _arch_cap(combo_name)
        grp = _group(arch, cap)
        color, marker, ec = _strategy_style(combo_name)
        wr  = df["weekly_return"]
        vol = wr.std() * np.sqrt(52) * 100
        tv  = df["total_value"]
        cum = (tv.iloc[-1] - INITIAL_CASH) / INITIAL_CASH * 100
        points[combo_name] = (vol, cum)

        ms  = 120 if grp == "mas" else 80
        ax.scatter(vol, cum, color=color, marker=marker, s=ms,
                   edgecolors=ec, linewidths=0.6, zorder=3, alpha=0.88)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)

    # Annotate after scatter so we know the full x range; flip label left near right edge
    all_vols  = [v for v, _ in points.values()]
    vol_max   = max(all_vols)
    vol_range = vol_max - min(all_vols)
    for combo_name, (vol, cum) in points.items():
        color  = _strategy_style(combo_name)[0]
        near_right = vol > vol_max - 0.15 * vol_range
        xoff, ha   = (-7, "right") if near_right else (5, "left")
        ax.annotate(
            _display_name(combo_name),
            xy=(vol, cum), xytext=(xoff, 3), textcoords="offset points",
            fontsize=11, color=color, fontweight="bold", ha=ha,
        )

    # ── Stacked legends: Capability on top, Group below ──
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    cap_handles = [
        mpatches.Patch(color=_CAP_COLOR["zero_shot"],        label="Zero-shot (ZS)"),
        mpatches.Patch(color=_CAP_COLOR["chain_of_thought"], label="Chain-of-thought (CoT)"),
        mpatches.Patch(color=_CAP_COLOR["rag"],              label="RAG"),
        mpatches.Patch(color=_CAP_COLOR["skill"],            label="Skill"),
        mpatches.Patch(color=_BM_HOLD_COLOR,                 label="Hold"),
        mpatches.Patch(color=_BM_DL_COLOR,                   label="Deep Learning"),
    ]
    grp_handles = [
        mlines.Line2D([], [], color="#555", marker="o", ls="none", ms=9, label="MAS"),
        mlines.Line2D([], [], color="#555", marker="D", ls="none", ms=9, label="Single Agent"),
        mlines.Line2D([], [], color="#555", marker="s", ls="none", ms=9, label="Hold"),
        mlines.Line2D([], [], color="#555", marker="^", ls="none", ms=9, label="Deep Learning"),
    ]
    leg1 = ax.legend(handles=cap_handles, title="Capability", fontsize=11,
                     title_fontsize=12, frameon=False,
                     loc="upper left", bbox_to_anchor=(0.01, 0.99))
    ax.add_artist(leg1)
    ax.legend(handles=grp_handles, title="Group", fontsize=11,
              title_fontsize=12, frameon=False,
              loc="upper left", bbox_to_anchor=(0.01, 0.65))

    ax.set_xlabel("Annualised Volatility (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.tick_params(labelsize=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.grid(axis="both", linestyle="--", alpha=0.3, zorder=1)
    fig.tight_layout()
    return _save_or_show(fig, save_path, show)


