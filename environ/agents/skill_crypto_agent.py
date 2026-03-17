"""
Skill-augmented Crypto Agent — technical indicator signals.

Extends CryptoAgent by pre-computing four technical indicators from the
raw 30-day daily data window and injecting them into every prompt.
The indicator definitions mirror those used in scripts/benchmarks.py:

  SMA_7     — 7-day simple moving average of closing price
  SMA_30    — 30-day simple moving average of closing price
  MACD_hist — MACD histogram: EMA(12) − EMA(26) − EMA9(MACD line)
  BB_lower  — Bollinger Band lower band: SMA(20) − 2 × std(20)

Interpretation rules (same as the benchmark signal strategies):
  SMA_7:     close > SMA_7    → bullish  (price above short-term trend)
  SLMA:      SMA_7 > SMA_30   → bullish  (short-term trend above long-term)
  MACD:      MACD_hist > 0    → bullish  (upward momentum)
  BB:        close < BB_lower → bullish  (oversold; mean-reversion opportunity)
"""

from .crypto_agent import CryptoAgent, _SYSTEM

_SKILL_ADDENDUM = """\

You will also receive pre-computed technical indicator values for each asset, \
computed from the same daily close price data provided above.

Indicators and how to interpret them:

  SMA_7     — 7-day simple moving average of closing price.
              Rule: close > SMA_7  → bullish  (price is above its short-term trend)
                    close < SMA_7  → bearish

  SMA_30    — 30-day simple moving average of closing price.
              Rule: SMA_7 > SMA_30 → bullish  (short-term trend is above long-term trend)
                    SMA_7 < SMA_30 → bearish

  MACD_hist — MACD histogram = (EMA12 − EMA26) − EMA9(EMA12 − EMA26).
              Rule: MACD_hist > 0  → bullish  (MACD line above signal line; upward momentum)
                    MACD_hist < 0  → bearish

  BB_lower  — Bollinger Band lower band = SMA(20) − 2 × std(20).
              Rule: close < BB_lower → bullish  (price is oversold; mean-reversion signal)
                    close > BB_lower → no mean-reversion signal from this indicator

Use the number of bullish indicators (0–4) as a composite signal strength: \
4 bullish → strong buy, 0 bullish → strong sell, 2–3 → moderate buy, 1 → moderate sell.\
"""

_FINAL_INSTR = "Analyse each asset and return the JSON signal array."


def _ema(prices: list[float], span: int) -> list[float]:
    """Exponential moving average with the same convention as pandas ewm(adjust=False)."""
    if not prices:
        return []
    alpha = 2.0 / (span + 1)
    result = [prices[0]]
    for p in prices[1:]:
        result.append(alpha * p + (1 - alpha) * result[-1])
    return result


def _compute_indicators(closes: list) -> dict | None:
    """
    Compute SMA_7, SMA_30, MACD_hist, and BB_lower from a list of daily closes.
    Returns None if there are fewer than 2 valid prices.
    """
    prices = [float(c) for c in closes if c is not None]
    n = len(prices)
    if n < 2:
        return None

    close = prices[-1]

    # SMA_7 and SMA_30
    sma_7  = sum(prices[-7:])  / min(n, 7)
    sma_30 = sum(prices[-30:]) / min(n, 30)

    # MACD histogram
    ema12      = _ema(prices, 12)
    ema26      = _ema(prices, 26)
    macd_line  = [a - b for a, b in zip(ema12, ema26)]
    signal_line = _ema(macd_line, 9)
    macd_hist  = macd_line[-1] - signal_line[-1]

    # Bollinger Band lower = SMA(20) − 2 × std(20)
    window = prices[-20:] if n >= 20 else prices
    bb_sma = sum(window) / len(window)
    bb_std = (sum((p - bb_sma) ** 2 for p in window) / len(window)) ** 0.5
    bb_lower = bb_sma - 2 * bb_std

    return {
        "close":     close,
        "sma_7":     sma_7,
        "sma_30":    sma_30,
        "macd_hist": macd_hist,
        "bb_lower":  bb_lower,
    }


def _signal_summary(ind: dict) -> str:
    """
    Return a compact signal string, e.g.  "SMA✓ SLMA✓ MACD✗ BB✗  [2/4 bullish]"
    """
    close, sma_7, sma_30 = ind["close"], ind["sma_7"], ind["sma_30"]
    macd_hist, bb_lower  = ind["macd_hist"], ind["bb_lower"]

    sma_bull  = close    > sma_7
    slma_bull = sma_7    > sma_30
    macd_bull = macd_hist > 0
    bb_bull   = close    < bb_lower

    def _tick(b: bool) -> str:
        return "✓" if b else "✗"

    count = sum([sma_bull, slma_bull, macd_bull, bb_bull])
    return (
        f"SMA{_tick(sma_bull)} SLMA{_tick(slma_bull)} "
        f"MACD{_tick(macd_bull)} BB{_tick(bb_bull)}  [{count}/4 bullish]"
    )


def _build_indicator_table(assets: list[dict]) -> str:
    """
    Compute technical indicators for every asset and return a prompt-ready block.
    """
    lines = [
        "Technical indicator signals:",
        f"  {'Symbol':<8}  {'Close':>12}  {'SMA_7':>12}  {'SMA_30':>12}  "
        f"{'MACD_hist':>12}  {'BB_lower':>12}  Signals",
        "  " + "-" * 95,
    ]

    for asset in assets:
        sym    = asset["symbol"]
        closes = asset.get("close", [])
        ind    = _compute_indicators(closes)

        if ind is None:
            lines.append(f"  {sym:<8}  (insufficient data)")
            continue

        signals = _signal_summary(ind)
        lines.append(
            f"  {sym:<8}  {ind['close']:>12,.2f}  {ind['sma_7']:>12,.2f}  "
            f"{ind['sma_30']:>12,.2f}  {ind['macd_hist']:>+12.4f}  "
            f"{ind['bb_lower']:>12,.2f}  {signals}"
        )

    return "\n".join(lines)


class SkillCryptoAgent(CryptoAgent):
    """CryptoAgent augmented with technical indicator signals (SMA, MACD, BB)."""

    @property
    def system_prompt(self) -> str:
        return _SYSTEM + _SKILL_ADDENDUM

    def build_user_message(self, context: dict) -> str:
        base   = super().build_user_message(context)
        assets = context.get("assets", [])
        if not assets:
            return base

        indicator_block = "\n\n" + _build_indicator_table(assets)
        return base.replace(_FINAL_INSTR, indicator_block + "\n\n" + _FINAL_INSTR)
