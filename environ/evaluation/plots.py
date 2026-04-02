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

from .metrics import load_all, compute_metrics, summary_table, INITIAL_CASH, WEEKS_PER_YEAR

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


# ── Model comparison bar chart ────────────────────────────────────────────────

_MAS_ARCHS = ("single_agent", "hierarchical", "collaborative", "debate")
_MAS_CAPS  = ("zero_shot", "chain_of_thought", "rag", "skill")

_ARCH_LABEL = {
    "single_agent":  "SA",
    "hierarchical":  "Hier",
    "collaborative": "Collab",
    "debate":        "Debate",
}
_CAP_LABEL = {
    "zero_shot":        "ZS",
    "chain_of_thought": "CoT",
    "rag":              "RAG",
    "skill":            "Skill",
}

# Pastel fill colors keyed by capability
_CAP_COLORS = {
    "zero_shot":        "#B3CDE3",   # pastel blue
    "chain_of_thought": "#CCEBC5",   # pastel green
    "rag":              "#FED9A6",   # pastel orange
    "skill":            "#DECBE4",   # pastel purple
}

# Hatch patterns keyed by model (solid = first model, hatched = second)
_MODEL_HATCHES = ["", "///", "xxx"]
_DIAGRAMS_DIR  = Path(__file__).parents[2] / "diagrams"


def _svg_to_array(svg_path: Path, size: int = 64) -> np.ndarray | None:
    """Rasterise an SVG to a float32 RGBA array, composited onto white."""
    try:
        import cairosvg
        from io import BytesIO
        import matplotlib.image as _mpimg
        png = cairosvg.svg2png(url=str(svg_path), output_width=size, output_height=size)
        arr = _mpimg.imread(BytesIO(png))          # float32 RGBA [0, 1]
        if arr.shape[2] == 4:
            alpha = arr[..., 3:4]
            rgb   = arr[..., :3] * alpha + (1.0 - alpha)
            arr   = np.concatenate([rgb, np.ones_like(alpha)], axis=2)
        return arr
    except Exception:
        return None


def _make_icon_handler(img: np.ndarray):
    """Return a legend HandlerBase that draws *img* inside the handle box."""
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.image import BboxImage
    from matplotlib.transforms import Bbox, TransformedBbox

    class _H(HandlerBase):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            tbox  = TransformedBbox(
                Bbox.from_bounds(xdescent, ydescent, width, height), trans
            )
            bimg  = BboxImage(tbox, zorder=3)
            bimg.set_data(img)
            return [bimg]

    return _H()


def _draw_comparison_panel(
    ax,
    all_combos: list[str],
    combos_data: dict[str, dict[str, tuple[float, float]]],
    models: list[str],
    val_idx: int,
    ylabel: str,
    show_legend: bool = True,
) -> None:
    """Draw one metric panel onto *ax*."""

    x     = np.arange(len(all_combos))
    bar_w = 0.25
    n_m   = len(models)
    offsets = np.linspace(-(n_m - 1) * bar_w / 2, (n_m - 1) * bar_w / 2, n_m)

    for m_idx, model in enumerate(models):
        hatch = _MODEL_HATCHES[m_idx] if m_idx < len(_MODEL_HATCHES) else ""
        for c_idx, combo in enumerate(all_combos):
            cap   = next((c for c in _MAS_CAPS if combo.endswith("_" + c)), "")
            color = _CAP_COLORS.get(cap, "#cccccc")
            val   = combos_data[model].get(combo, (np.nan, np.nan))[val_idx]
            bar   = ax.bar(
                x[c_idx] + offsets[m_idx], val,
                width=bar_w, color=color, hatch=hatch,
                edgecolor="#555555", linewidth=0.5,
            )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # architecture separators
    cap_count = len(_MAS_CAPS)
    for i in range(len(_MAS_ARCHS) - 1):
        ax.axvline((i + 1) * cap_count - 0.5,
                   color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)

    # architecture group labels
    for i, arch in enumerate(_MAS_ARCHS):
        centre = i * cap_count + (cap_count - 1) / 2
        ax.text(centre, 1.03, _ARCH_LABEL[arch],
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom",
                fontsize=15, fontweight="bold", color="#444444")

    # x-axis tick labels
    resolved = []
    for c in all_combos:
        arch = next((a for a in _MAS_ARCHS if c.startswith(a + "_")), "")
        cap  = c[len(arch) + 1:] if arch else c
        resolved.append(f"{_ARCH_LABEL.get(arch, arch)}\n{_CAP_LABEL.get(cap, cap)}")
    ax.set_xticks(x)
    ax.set_xticklabels([], fontsize=13)
    ax.set_xlim(-0.6, len(all_combos) - 0.4)

    if not show_legend:
        return

    # ── Legends: capability (colors) and model (hatches), both frameless on left ──
    from matplotlib.patches import Patch as _Patch
    cap_handles = [
        _Patch(facecolor=_CAP_COLORS[cap], edgecolor="#555555")
        for cap in _MAS_CAPS
    ]
    model_handles = [
        _Patch(facecolor="#dddddd",
               hatch=_MODEL_HATCHES[m_idx] if m_idx < len(_MODEL_HATCHES) else "",
               edgecolor="#555555")
        for m_idx in range(len(models))
    ]

    leg1 = ax.legend(
        cap_handles, [_CAP_LABEL[c] for c in _MAS_CAPS],
        title="Capability", title_fontsize=13,
        loc="upper left", fontsize=13,
        frameon=False, handlelength=1.5, handleheight=1.2,
    )
    ax.add_artist(leg1)

    ax.legend(
        model_handles, list(models),
        title="Model", title_fontsize=13,
        loc="upper left", bbox_to_anchor=(0.115, 1.0),
        fontsize=13, frameon=False,
        handlelength=1.5, handleheight=1.2,
    )


def plot_model_comparison(
    base_dir: Path = Path("processed_data"),
    models: tuple[str, ...] = ("gpt-4o", "gpt-5", "claude-sonnet-4-5"),
    save_path: Path | None = None,
    show: bool = False,
) -> list[Path]:
    """Two bar-chart figures comparing cumulative return and annualised
    volatility between *models* across every architecture × capability.

    Colors encode capability; hatch encodes model.  SVG icons are placed
    in the model legend when ``diagrams/<model>.svg`` exists.

    Returns a list of the two saved Paths (empty on *show* mode).
    """
    # ── 1. Load metrics ───────────────────────────────────────────────────────
    combos_data: dict[str, dict[str, tuple[float, float]]] = {}
    for model in models:
        model_dir = base_dir / model
        if not model_dir.is_dir():
            print(f"Warning: {model_dir} not found — skipping {model}")
            continue
        loaded = load_all(model_dir)
        metrics: dict[str, tuple[float, float]] = {}
        for arch in _MAS_ARCHS:
            for cap in _MAS_CAPS:
                name = f"{arch}_{cap}"
                df   = loaded.get(name)
                if df is None or df.empty:
                    continue
                wr = df["weekly_return"]
                metrics[name] = (
                    float((1 + wr).prod() - 1) * 100,
                    float(wr.std() * np.sqrt(WEEKS_PER_YEAR)) * 100,
                )
        combos_data[model] = metrics

    if not combos_data:
        print("No model data found — skipping model comparison chart.")
        return []

    all_combos = [
        f"{arch}_{cap}"
        for arch in _MAS_ARCHS
        for cap in _MAS_CAPS
        if any(f"{arch}_{cap}" in combos_data.get(m, {}) for m in models)
    ]
    model_list = list(combos_data.keys())

    metric_cfg = [
        (0, "Cumulative Return (%)",     "cum_ret",  True),
        (1, "Annualised Volatility (%)", "ann_vol",  False),
    ]

    figures_dir = (save_path.parent if save_path else
                   base_dir.parent / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for val_idx, ylabel, suffix, show_legend in metric_cfg:
        fig, ax = plt.subplots(figsize=(max(12, len(all_combos) * 0.9), 3.5))
        fig.subplots_adjust(top=0.88)
        _draw_comparison_panel(ax, all_combos, combos_data, model_list, val_idx, ylabel,
                               show_legend=show_legend)

        if show:
            plt.show()
            plt.close(fig)
            continue

        out = figures_dir / f"model_comparison_{suffix}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)

    return saved
