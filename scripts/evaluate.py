"""
Evaluation CLI — generates all performance plots and prints a metrics summary
broken down by market regime (All / Bull / Bear / Sideways).

Regime classification follows Cagan (2024) using the mcap-weighted basket
(benchmark_mcap_hold) as the market index:
    Bull    : basket ≥ +20 % above its running trough
    Bear    : basket ≥ −20 % below its running peak
    Sideways: neither (Bear takes priority over Bull)

Figures always cover the full backtest period.
Printed stats are shown for each regime separately.

Usage:
    python3 scripts/evaluate.py
    python3 scripts/evaluate.py --output-dir results/
    python3 scripts/evaluate.py --show              # display plots interactively
    python3 scripts/evaluate.py --plot portfolio drawdown   # subset of plots
    python3 scripts/evaluate.py --regime bear       # print only bear regime stats
    python3 scripts/evaluate.py --latex             # print LaTeX table to stdout
    python3 scripts/evaluate.py --latex-out table.tex  # save LaTeX table to file
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from environ.evaluation import (
    summary_table,
    plot_portfolio,
    plot_drawdown,
    plot_weekly_returns,
    plot_metrics_heatmap,
)
from environ.evaluation.metrics import load_combination, load_all, INITIAL_CASH

PLOTS = {
    "portfolio": plot_portfolio,
    "drawdown":  plot_drawdown,
    "returns":   plot_weekly_returns,
    "heatmap":   plot_metrics_heatmap,
}

FIGURES_DIR = Path("figures")
TABLES_DIR  = Path("tables")

FILENAMES = {
    "portfolio": "portfolio_timeseries.pdf",
    "drawdown":  "drawdown_timeseries.pdf",
    "returns":   "weekly_returns.pdf",
    "heatmap":   "metrics_heatmap.pdf",
}

REGIMES = ["all", "bull", "bear"]
WEEKS_PER_YEAR = 52.0


# ── Regime helpers (inlined from evaluate_regimes.py) ─────────────────────────

def _build_basket(data_dir: Path) -> pd.Series | None:
    """Load benchmark_mcap_hold total_value as the market basket. Returns None on failure."""
    basket_dir = data_dir / "benchmark_mcap_hold"
    df = load_combination(basket_dir)
    return df["total_value"] if not df.empty else None


def _classify_regimes(basket: pd.Series) -> pd.Series:
    """
    Classify each ISO week using Cagan (2024) thresholds:
        Bear    : basket ≤ 0.80 × running peak   (priority)
        Bull    : basket ≥ 1.20 × running trough
        Sideways: neither
    """
    running_max = basket.cummax()
    running_min = basket.cummin()
    drawdown    = (basket - running_max) / running_max
    rally       = (basket - running_min) / running_min

    regime = pd.Series("sideways", index=basket.index, dtype=str)
    regime[rally    >= 0.20]  = "bull"
    regime[drawdown <= -0.20] = "bear"
    return regime


def _metrics_subset(wr: pd.Series, tv: pd.Series) -> dict:
    """Performance metrics for a (possibly non-contiguous) week subset."""
    n = len(wr)
    if n == 0:
        return {}

    cum_ret      = float((1 + wr).prod() - 1)
    avg_weekly   = float(wr.mean())
    std_weekly   = float(wr.std()) if n > 1 else np.nan
    ann_vol      = float(wr.std() * np.sqrt(WEEKS_PER_YEAR)) if n > 1 else np.nan
    ann_ret      = float((1 + cum_ret) ** (WEEKS_PER_YEAR / n) - 1)
    sharpe       = ann_ret / ann_vol if (ann_vol and ann_vol > 0) else np.nan

    run_max  = tv.cummax()
    max_dd   = float(((tv - run_max) / run_max).min())
    win_rate = float((wr > 0).mean())

    def fmt(x, d=2):
        return round(x * 100, d) if not np.isnan(x) else None

    return {
        "n_weeks":            n,
        "cum_ret_pct":        fmt(cum_ret),
        "avg_weekly_ret_pct": fmt(avg_weekly),
        "std_weekly_ret_pct": fmt(std_weekly) if (n > 1 and not np.isnan(std_weekly)) else None,
        "ann_vol_pct":        fmt(ann_vol) if (ann_vol and not np.isnan(ann_vol)) else None,
        "sharpe":            round(sharpe, 3) if not np.isnan(sharpe) else None,
        "max_dd_pct":        fmt(max_dd),
        "win_rate_pct":      fmt(win_rate, 1),
    }


def _regime_rows(combos: dict[str, pd.DataFrame], regimes: pd.Series) -> dict[str, list[dict]]:
    """Build {regime: [row_dict, ...]} for every combination × regime."""
    rows: dict[str, list[dict]] = {r: [] for r in REGIMES}

    for name, df in combos.items():
        shared      = df.index.intersection(regimes.index)
        df_al       = df.loc[shared]
        reg_al      = regimes.loc[shared]

        # "all" = full intersection
        m_all = _metrics_subset(df_al["weekly_return"], df_al["total_value"])
        if m_all.get("n_weeks", 0) > 0:
            rows["all"].append({"combination": name, **m_all})

        for reg in ("bull", "bear"):
            sub = df_al[reg_al == reg]
            m   = _metrics_subset(sub["weekly_return"], sub["total_value"]) if not sub.empty else {}
            if m.get("n_weeks", 0) > 0:
                rows[reg].append({"combination": name, **m})

    return rows


# ── Printing helpers ──────────────────────────────────────────────────────────

_REGIME_COLS   = ["n_weeks", "cum_ret_pct", "avg_weekly_ret_pct", "ann_vol_pct",
                  "sharpe", "max_dd_pct", "win_rate_pct"]
_REGIME_LABELS = {
    "n_weeks":            "Wks",
    "cum_ret_pct":        "Cum Ret%",
    "avg_weekly_ret_pct": "Avg%",
    "ann_vol_pct":        "Ann Vol%",
    "sharpe":             "Sharpe",
    "max_dd_pct":         "MaxDD%",
    "win_rate_pct":       "Win%",
}

_KNOWN_ARCHS = ("blackboard", "hierarchical", "collaborative", "debate",
                "benchmark", "single_agent")


def _split_name(combo: str) -> tuple[str, str]:
    """Split 'benchmark_sma7' → ('benchmark', 'sma7'), handling multi-word archs."""
    for arch in _KNOWN_ARCHS:
        if combo.startswith(arch + "_"):
            return arch, combo[len(arch) + 1:]
    parts = combo.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (combo, "")


def _print_regime_table(rows: list[dict], regime: str) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    df.insert(0, "Arch", df["combination"].apply(lambda x: _split_name(x)[0]))
    df.insert(1, "Cap",  df["combination"].apply(lambda x: _split_name(x)[1]))
    df = df.drop(columns=["combination"])

    cols   = ["Arch", "Cap"] + [c for c in _REGIME_COLS if c in df.columns]
    df     = df[cols].rename(columns=_REGIME_LABELS)
    df     = df.sort_values("Cum Ret%", ascending=False, na_position="last")

    title = f"  {regime.upper()} MARKET"
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(df.to_string(index=False))
    print()


def _print_basket_header(basket: pd.Series, regimes: pd.Series) -> None:
    counts  = regimes.value_counts()
    n       = len(basket)
    ret_pct = (basket.iloc[-1] - basket.iloc[0]) / basket.iloc[0] * 100

    print("\n" + "=" * 80)
    print("  MARKET BASKET  (benchmark_mcap_hold · mcap-weighted · 15 assets)")
    print("=" * 80)
    print(f"  Total weeks : {n}   |   Basket return : {ret_pct:+.1f}%   "
          f"|   Range : {basket.min():,.0f} – {basket.max():,.0f}")
    print()
    print("  Regime breakdown  (Cagan 2024 · thresholds: ±20 %)")
    for reg in ("bull", "bear"):
        cnt = counts.get(reg, 0)
        print(f"    {reg.capitalize():>10s} : {cnt:3d} weeks ({cnt / n * 100:.0f} %)")
    print()


# ── LaTeX table ───────────────────────────────────────────────────────────────

# Fixed strategy ordering and display names for the LaTeX table.
# Groups are separated by \midrule in the output.
_LATEX_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Hold", [
        ("benchmark_btc_hold",  "BTC Hold"),
        ("benchmark_mcap_hold", "MCap Hold"),
    ]),
    ("Deep Learning", [
        ("benchmark_lstm",       "LSTM"),
        ("benchmark_informer",   "Informer"),
        ("benchmark_autoformer", "Autoformer"),
        ("benchmark_timesnet",   "TimesNet"),
        ("benchmark_patchtst",   "PatchTST"),
    ]),
    ("Agent", [
        ("single_agent_zero_shot",        "SA (ZS)"),
        ("single_agent_chain_of_thought", "SA (CoT)"),
        ("single_agent_rag",              "SA (RAG)"),
        ("single_agent_skill",            "SA (Skill)"),
    ]),
    ("Multi-Agent System", [
        ("hierarchical_zero_shot",         "Hier.\\ (ZS)"),
        ("hierarchical_chain_of_thought",  "Hier.\\ (CoT)"),
        ("hierarchical_rag",               "Hier.\\ (RAG)"),
        ("hierarchical_skill",             "Hier.\\ (Skill)"),
        ("collaborative_zero_shot",        "Collab.\\ (ZS)"),
        ("collaborative_chain_of_thought", "Collab.\\ (CoT)"),
        ("collaborative_rag",              "Collab.\\ (RAG)"),
        ("collaborative_skill",            "Collab.\\ (Skill)"),
        ("debate_zero_shot",               "Debate (ZS)"),
        ("debate_chain_of_thought",        "Debate (CoT)"),
        ("debate_rag",                     "Debate (RAG)"),
        ("debate_skill",                   "Debate (Skill)"),
    ]),
]

# Metrics emitted per regime (in column order)
# Tuple: (key, label, fmt, bold_flag, sub_key, direction)
# direction: "up" = higher is better (red ↑), "down" = lower is better (green ↓)
_LATEX_METRICS = [
    ("cum_ret_pct",        "Cum\\%", "{:+.2f}", False, None,                  "up"),
    ("avg_weekly_ret_pct", "Avg\\%", "{:+.2f}", False, None, "up"),
    ("ann_vol_pct",  "Vol\\%",   "{:.2f}",  False, None, "down"),
    ("sharpe",       "SR",       "{:+.3f}", False, None, "up"),
    ("max_dd_pct",   "MDD\\%",   "{:.2f}",  False, None, "up"),
    ("win_rate_pct", "Win\\%",   "{:.1f}",  False, None, "up"),
]
_N_METRICS = len(_LATEX_METRICS)


def _cell(value, fmt: str) -> str:
    """Format a metric value, returning \\textemdash{} for missing data."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "---"
    if fmt.startswith("{:d}"):
        return fmt.format(int(value))
    return fmt.format(float(value))


def _cell_with_sub(value, fmt: str, sub_value) -> str:
    """Format a metric with ±std as a math-mode subscript."""
    main = _cell(value, fmt)
    if main == "---" or sub_value is None:
        return main
    sub = "{:.2f}".format(float(sub_value))
    return f"${main}_{{\\pm {sub}}}$"


def latex_table(
    regime_data: dict[str, list[dict]],
    *,
    regime_counts: dict[str, int] | None = None,
    caption: str = "Strategy performance across market regimes. Each cell is colored to indicate the \\colorbox{orange!60}{best}, \\colorbox{orange!30}{second best}, and \\colorbox{yellow!30}{third best} performing strategy per metric and regime.",
    label: str = "tab:performance",
) -> str:
    """
    Build a LaTeX ``tabular`` table comparing all strategies across regimes.

    Layout
    ------
    Columns : Strategy | All (N cols) | Bull | Bear
    Rows    : grouped by strategy type, separated by \\midrule

    Returns the complete table as a string (suitable for print or file write).
    Requires booktabs and multirow in the LaTeX preamble.
    """
    # Index regime data by combination name for O(1) lookup
    by_regime: dict[str, dict[str, dict]] = {}
    for reg, rows in regime_data.items():
        by_regime[reg] = {r["combination"]: r for r in rows}

    # ── Top-3 shading ─────────────────────────────────────────────────────────
    _SHADE = {
        1: r"\cellcolor{orange!60}",
        2: r"\cellcolor{orange!30}",
        3: r"\cellcolor{yellow!30}",
    }
    rankings: dict[tuple[str, str, str], int] = {}
    for reg in REGIMES:
        reg_data = by_regime.get(reg, {})
        for key, _, _, _, _, direction in _LATEX_METRICS:
            values = [
                (float(row[key]), combo)
                for combo, row in reg_data.items()
                if row.get(key) is not None and not (isinstance(row.get(key), float) and np.isnan(row[key]))
            ]
            values.sort(key=lambda x: x[0], reverse=(direction == "up"))
            for rank, (_, combo) in enumerate(values[:3], start=1):
                rankings[(reg, key, combo)] = rank

    n_regime_cols = _N_METRICS
    n_total_cols  = 2 + n_regime_cols * 3   # group + strategy + 3 regimes × N metrics

    # ── Column spec ───────────────────────────────────────────────────────────
    metric_spec = "r" * n_regime_cols
    col_spec    = f"c l | {metric_spec} | {metric_spec} | {metric_spec}"

    # ── Header row 1: regime multicolumns ─────────────────────────────────────
    def _regime_label(name: str, defn: str) -> str:
        if regime_counts and name in regime_counts:
            n = regime_counts[name]
            return f"\\textbf{{{name}}} ({defn}, $N={n}$)"
        return f"\\textbf{{{name}}}"

    regime_headers = [
        r"\multicolumn{" + str(n_regime_cols) + r"}{c|}{" + _regime_label("All",  "full period") + "}",
        r"\multicolumn{" + str(n_regime_cols) + r"}{c|}{" + _regime_label("Bull", r"rally $\geq+20\%$") + "}",
        r"\multicolumn{" + str(n_regime_cols) + r"}{c}{"  + _regime_label("Bear", r"drawdown $\leq-20\%$") + "}",
    ]
    header1 = (r"\multicolumn{2}{c|}{\multirow{2}{*}{\textbf{Strategy}}} & "
               + " & ".join(regime_headers) + r" \\")

    # ── Header row 2: per-metric sub-headers with direction arrows ────────────
    _ARROW = {
        "up":   r"\textcolor{red}{$\uparrow$}",
        "down": r"\textcolor{teal}{$\downarrow$}",
    }
    sub = " & ".join(f"{m[1]}{_ARROW[m[5]]}" for m in _LATEX_METRICS)
    header2 = "& & " + " & ".join([sub] * 3) + r" \\"

    # ── cmidrule decorations under regime headers ─────────────────────────────
    cmidrules = []
    for i, reg in enumerate(REGIMES):
        start = 3 + i * n_regime_cols   # shift by 1 for group column
        end   = start + n_regime_cols - 1
        lr    = "r" if i < 2 else ""   # no right-padding on last group
        cmidrules.append(f"\\cmidrule(l{lr}){{{start}-{end}}}")
    cmidrule_line = " ".join(cmidrules)

    # ── Data rows ─────────────────────────────────────────────────────────────
    data_lines: list[str] = []
    for g_idx, (group_label, strategies) in enumerate(_LATEX_GROUPS):
        if g_idx > 0:
            prev_label = _LATEX_GROUPS[g_idx - 1][0]
            if prev_label == "Agent":
                data_lines.append(r"\cdashline{1-" + str(n_total_cols) + "}")
            else:
                data_lines.append(r"\cline{1-" + str(n_total_cols) + "}")
        n_strat = len(strategies)
        for s_idx, (combo, display) in enumerate(strategies):
            group_cell = (
                r"\multicolumn{1}{c}{\multirow{" + str(n_strat) +
                r"}{*}{\rotatebox[origin=c]{90}{{\tiny\textbf{" + group_label + r"}}}}}"
                if s_idx == 0 else ""
            )
            cells = [group_cell, display]
            for reg in REGIMES:
                row = by_regime.get(reg, {}).get(combo, {})
                for key, _, fmt, _, sub_key, _ in _LATEX_METRICS:
                    shade = _SHADE.get(rankings.get((reg, key, combo), 0), "")
                    if sub_key:
                        cells.append(shade + _cell_with_sub(row.get(key), fmt, row.get(sub_key)))
                    else:
                        cells.append(shade + _cell(row.get(key), fmt))
            data_lines.append(" & ".join(cells) + r" \\")

    # ── Assemble ──────────────────────────────────────────────────────────────
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header1,
        cmidrule_line,
        header2,
        r"\midrule",
    ] + data_lines + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate MAS backtest results.")
    parser.add_argument("--data-dir",   default="processed_data",
                        help="Directory containing combination sub-folders (default: processed_data)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save figures (default: figures/)")
    parser.add_argument("--plot", nargs="+", choices=list(PLOTS),
                        default=list(PLOTS),
                        help="Which plots to generate (default: all)")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively instead of saving")
    parser.add_argument("--regime", choices=REGIMES, default=None,
                        help="Print stats for this regime only (default: all regimes)")
    parser.add_argument("--latex", action="store_true",
                        help="Output a LaTeX tabular table of regime-conditioned metrics")
    parser.add_argument("--latex-out", default=None, metavar="FILE",
                        help="Save the LaTeX table to this file (default: tables/regime_performance.tex)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else FIGURES_DIR

    # ── Load data ─────────────────────────────────────────────────────────────
    combos = load_all(data_dir)
    if not combos:
        print(f"No results found in {data_dir}/")
        return

    # ── Regime setup ──────────────────────────────────────────────────────────
    basket  = _build_basket(data_dir)
    regimes = _classify_regimes(basket) if basket is not None else None

    if regimes is None:
        print("Warning: benchmark_mcap_hold not found — printing full-period stats only.")

    # ── Printed stats: per-regime tables ──────────────────────────────────────
    if regimes is not None:
        _print_basket_header(basket, regimes)
        regime_data = _regime_rows(combos, regimes)
        target_regimes = [args.regime] if args.regime else REGIMES
        for reg in target_regimes:
            _print_regime_table(regime_data[reg], reg)
    else:
        # Fallback: original full-period summary table
        tbl = summary_table(data_dir)
        if not tbl.empty:
            display_cols = [
                "architecture", "capability",
                "total_return_pct", "annualized_return_pct", "annualized_vol_pct",
                "sharpe", "max_drawdown_pct", "win_rate_pct", "n_weeks",
            ]
            display_cols = [c for c in display_cols if c in tbl.columns]
            col_names = {
                "architecture":          "Arch",
                "capability":            "Capability",
                "total_return_pct":      "Total Ret%",
                "annualized_return_pct": "Ann Ret%",
                "annualized_vol_pct":    "Ann Vol%",
                "sharpe":                "Sharpe",
                "max_drawdown_pct":      "MaxDD%",
                "win_rate_pct":          "Win%",
                "n_weeks":               "Weeks",
            }
            print("\n" + "=" * 80)
            print("  PERFORMANCE METRICS SUMMARY  (full period)")
            print("=" * 80)
            print(tbl[display_cols].rename(columns=col_names).to_string())
            print()

    # ── LaTeX table ───────────────────────────────────────────────────────────
    if regimes is None:
        print("Warning: benchmark_mcap_hold not found — cannot build LaTeX table.")
    else:
        counts = regimes.value_counts()
        regime_counts = {
            "All":  len(regimes),
            "Bull": int(counts.get("bull", 0)),
            "Bear": int(counts.get("bear", 0)),
        }
        tex = latex_table(regime_data, regime_counts=regime_counts)
        if args.latex_out:
            out_tex = Path(args.latex_out)
        else:
            out_tex = TABLES_DIR / "performance.tex"
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_tex.write_text(tex, encoding="utf-8")
        print(f"LaTeX table saved → {out_tex}")

    # ── Plots (always full period) ────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    for plot_name in args.plot:
        fn       = PLOTS[plot_name]
        out_path = output_dir / FILENAMES[plot_name]
        result   = fn(output_dir=data_dir, save_path=out_path, show=args.show)
        if result:
            print(f"Saved {plot_name:12s} → {result}")


if __name__ == "__main__":
    main()
