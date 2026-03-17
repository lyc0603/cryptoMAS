#!/usr/bin/env python3
"""
Benchmark strategies for cryptocurrency portfolio management.

Strategies
----------
btc_hold     Buy-and-Hold Bitcoin (100%)
equal_hold   Buy-and-Hold Equal-Weight 1/N across all 15 assets
mcap_hold    Buy-and-Hold Market-Cap-Weighted across all 15 assets
sma7         Long when close > SMA(7), equal-weight, weekly full-rebalance
slma         Long when SMA(7) > SMA(30), equal-weight, weekly full-rebalance
macd         Long when MACD > signal line, equal-weight, weekly full-rebalance
lstm         LSTM next-day-close predictor; long when predicted > current close
informer     Informer transformer; long when predicted next-day close > current
autoformer   Autoformer; long when predicted next-day close > current
timesnet     TimesNet; long when predicted next-day close > current
patchtst     PatchTST; long when predicted next-day close > current

Output
------
Saves one JSON file per week per strategy under processed_data/benchmark_<name>/.
Format mirrors run_experiment.py so evaluate.py and the metrics/plots modules
load all strategies together without modification.

Usage
-----
    python scripts/benchmarks.py                        # run all strategies
    python scripts/benchmarks.py --strategy sma7        # single strategy
    python scripts/benchmarks.py --weeks 2025-W01 2025-W10
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from neuralforecast import NeuralForecast as _NeuralForecast
from neuralforecast.models import (
    Informer   as _Informer,
    Autoformer as _Autoformer,
    TimesNet   as _TimesNet,
    PatchTST   as _PatchTST,
)

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from environ.data.coingecko import SYMBOL_TO_ID, load_asset

# ── Constants (mirror run_experiment.py) ──────────────────────────────────────

UNIVERSE         = list(SYMBOL_TO_ID.keys())
BACKTEST_START   = "2025-01-01"
BACKTEST_END     = "2026-01-01"
INITIAL_CASH     = 100_000.0
TRANSACTION_COST = 0.001          # 0.1 % per trade side
OUTPUT_DIR       = Path("processed_data")

STRATEGIES = ["btc_hold", "mcap_hold", "sma7", "slma", "macd", "bb", "lstm",
              "informer", "autoformer", "timesnet", "patchtst"]

# ── LSTM hyper-parameters ──────────────────────────────────────────────────────

LSTM_LOOKBACK = 5     # days of history per input sequence
LSTM_HIDDEN   = 100
LSTM_LAYERS   = 2
LSTM_EPOCHS   = 100
LSTM_LR       = 0.001
LSTM_BATCH    = 64
LSTM_DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── LSTM model ────────────────────────────────────────────────────────────────

class _LSTMModel(nn.Module):
    """2-layer LSTM → 1 output (next-day close, scaled)."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, LSTM_HIDDEN, LSTM_LAYERS, batch_first=True)
        self.fc   = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(LSTM_LAYERS, x.size(0), LSTM_HIDDEN, device=x.device)
        c0 = torch.zeros(LSTM_LAYERS, x.size(0), LSTM_HIDDEN, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def _build_sequences(scaled: np.ndarray, look_back: int):
    """Return (X, Y) tensors from a 1-D scaled array."""
    X, Y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i : i + look_back])
        Y.append(scaled[i + look_back])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (N, L, 1)
    Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)  # (N, 1)
    return X, Y


def train_lstm_models(train_end: str = BACKTEST_START) -> dict[str, tuple]:
    """
    Train one LSTM per asset on all daily close prices before `train_end`.

    Returns {sym: (model, scaler)} ready for inference.
    Models are trained on pre-backtest history only; each weekly inference
    call uses only data up to (but not including) the current week.
    """
    end_ts  = pd.Timestamp(train_end, tz="UTC")
    results = {}
    logger.info("LSTM: training %d models on data before %s (device=%s)",
                len(UNIVERSE), train_end, LSTM_DEVICE)

    for sym in UNIVERSE:
        df    = load_asset(sym)
        close = df.loc[:end_ts, "close"].dropna().values.reshape(-1, 1)

        if len(close) < LSTM_LOOKBACK + 10:
            logger.warning("LSTM: insufficient training data for %s — skipped", sym)
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close).flatten()

        X, Y   = _build_sequences(scaled, LSTM_LOOKBACK)
        loader = DataLoader(TensorDataset(X, Y), batch_size=LSTM_BATCH, shuffle=True)

        model   = _LSTMModel().to(LSTM_DEVICE)
        opt     = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(LSTM_EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(LSTM_DEVICE), yb.to(LSTM_DEVICE)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            if epoch % 20 == 0:
                logger.debug("LSTM [%s] epoch %d — loss %.6f", sym, epoch, loss_fn(
                    model(X.to(LSTM_DEVICE)), Y.to(LSTM_DEVICE)).item())

        results[sym] = (model.eval(), scaler)
        logger.info("LSTM: %s trained on %d days", sym, len(close))

    logger.info("LSTM: training complete (%d/%d assets)", len(results), len(UNIVERSE))
    return results


def get_lstm_signals(
    models_scalers: dict[str, tuple],
    as_of: pd.Timestamp,
) -> list[str]:
    """
    Return symbols where the LSTM predicts the next-day close > current close.

    Uses all daily close data up to `as_of` for inference, so the model sees
    the full available history at query time (no lookahead — the model weights
    themselves were fixed at training time before the backtest).
    """
    signals = []
    for sym, (model, scaler) in models_scalers.items():
        df    = load_asset(sym)
        close = df.loc[:as_of, "close"].dropna()
        if len(close) < LSTM_LOOKBACK:
            continue

        seq        = close.values[-LSTM_LOOKBACK:].reshape(-1, 1)
        seq_scaled = scaler.transform(seq).flatten()
        x          = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(LSTM_DEVICE)

        with torch.no_grad():
            pred_scaled = model(x).cpu().numpy()

        pred    = float(scaler.inverse_transform(pred_scaled)[0, 0])
        current = float(close.iloc[-1])
        if pred > current:
            signals.append(sym)

    return signals


# ── Neural forecasting models (Informer / Autoformer / TimesNet / PatchTST) ───

# Hyperparameters
NF_HORIZON    = 1    # predict 1 day ahead
NF_INPUT_SIZE = 32   # days of lookback (power-of-2 friendly for transformers)
NF_MAX_STEPS  = 100  # gradient steps during training

_NF_REGISTRY: dict[str, tuple] = {
    "informer":   (_Informer,   {}),
    "autoformer": (_Autoformer, {}),
    "timesnet":   (_TimesNet,   {}),
    "patchtst":   (_PatchTST,   {"patch_len": 8, "stride": 4}),
}

# ── Validation-set hyperparameter tuning ──────────────────────────────────────

VAL_START = "2024-07-01"   # 6-month validation window immediately before backtest
VAL_END   = "2024-12-31"

_SMA7_GRID = [{"period": p} for p in [5, 7, 10, 14, 20, 30]]
_SLMA_GRID = [{"short": s, "long": l}
              for s, l in [(5,20),(5,30),(7,14),(7,30),(10,20),(10,30),(14,30)]]
_MACD_GRID = [{"fast": f, "slow": s, "signal": sg}
              for f, s, sg in [(12,26,9),(5,13,9),(8,17,9),(10,22,9)]]
_BB_GRID   = [{"period": p, "multiplier": m}
              for p, m in [(10,2.0),(20,1.5),(20,2.0),(20,2.5),(30,2.0)]]

_PARAM_GRIDS = {"sma7": _SMA7_GRID, "slma": _SLMA_GRID, "macd": _MACD_GRID, "bb": _BB_GRID}
_TUNE_STRATS = set(_PARAM_GRIDS)


def _build_nf_df(end_ts: pd.Timestamp) -> "pd.DataFrame":
    """Long-format DataFrame [unique_id, ds, y] for all assets up to end_ts."""
    frames = []
    for sym in UNIVERSE:
        df    = load_asset(sym)
        close = df.loc[:end_ts, "close"].dropna().reset_index()
        close.columns = ["ds", "y"]
        close["unique_id"] = sym
        close["ds"] = close["ds"].dt.tz_localize(None)   # neuralforecast expects tz-naive
        frames.append(close[["unique_id", "ds", "y"]])
    return pd.concat(frames, ignore_index=True)


def train_neural_model(model_name: str, train_end: str = BACKTEST_START) -> "_NeuralForecast":
    """
    Train a neuralforecast model on all assets using data before `train_end`.
    Returns the fitted NeuralForecast object for subsequent inference.
    """
    end_ts      = pd.Timestamp(train_end, tz="UTC")
    cls, kw     = _NF_REGISTRY[model_name]
    model       = cls(
        h=NF_HORIZON,
        input_size=NF_INPUT_SIZE,
        max_steps=NF_MAX_STEPS,
        scaler_type="standard",
        **kw,
    )
    train_df    = _build_nf_df(end_ts)
    nf          = _NeuralForecast(models=[model], freq="D")
    logger.info("Neural [%s]: training on %d rows before %s …", model_name, len(train_df), train_end)
    nf.fit(train_df)
    logger.info("Neural [%s]: training complete.", model_name)
    return nf


def get_neural_signals(nf: "_NeuralForecast", as_of: pd.Timestamp) -> list[str]:
    """Return symbols where the model predicts next-day close > current close."""
    inf_df    = _build_nf_df(as_of)
    forecasts = nf.predict(df=inf_df)
    model_col = next(c for c in forecasts.columns if c not in ("unique_id", "ds"))
    signals   = []
    for sym in UNIVERSE:
        row = forecasts[forecasts["unique_id"] == sym]
        if row.empty:
            continue
        pred          = float(row[model_col].iloc[0])
        current_close = load_asset(sym).loc[:as_of, "close"].dropna()
        if not current_close.empty and pred > float(current_close.iloc[-1]):
            signals.append(sym)
    return signals


# ── Parametric signals + validation tuning ────────────────────────────────────

def _get_signals_parametric(
    as_of: pd.Timestamp,
    strategy: str,
    params: dict,
) -> list[str]:
    """Compute buy signals with explicit hyperparameters."""
    signals = []
    for sym in UNIVERSE:
        df    = load_asset(sym)
        close = df.loc[:as_of, "close"].dropna()
        if close.empty:
            continue
        c = float(close.iloc[-1])

        if strategy == "sma7":
            sma = close.rolling(params["period"]).mean().iloc[-1]
            if pd.notna(sma) and c > float(sma):
                signals.append(sym)

        elif strategy == "slma":
            s = close.rolling(params["short"]).mean().iloc[-1]
            l = close.rolling(params["long"]).mean().iloc[-1]
            if pd.notna(s) and pd.notna(l) and float(s) > float(l):
                signals.append(sym)

        elif strategy == "macd":
            ema_f     = close.ewm(span=params["fast"],   adjust=False).mean()
            ema_s     = close.ewm(span=params["slow"],   adjust=False).mean()
            macd_line = ema_f - ema_s
            sig_line  = macd_line.ewm(span=params["signal"], adjust=False).mean()
            hist = float((macd_line - sig_line).iloc[-1])
            if pd.notna(hist) and hist > 0:
                signals.append(sym)

        elif strategy == "bb":
            per  = params["period"]
            mult = params["multiplier"]
            sma  = close.rolling(per).mean().iloc[-1]
            std  = close.rolling(per).std().iloc[-1]
            if pd.notna(sma) and pd.notna(std) and c < float(sma) - mult * float(std):
                signals.append(sym)

    return signals


def _val_return(strategy: str, params: dict) -> float:
    """Simulate weekly trading on the validation period; return total return."""
    val_weeks = generate_weeks(VAL_START, VAL_END)
    if not val_weeks:
        return 0.0
    portfolio = Portfolio()
    for week in val_weeks:
        sunday  = week_sunday(week)
        prices  = get_execution_prices(sunday)
        signals = _get_signals_parametric(sunday, strategy, params)
        actions = compute_actions(strategy, portfolio, signals)
        portfolio.apply_actions(actions, prices)
    final_prices = get_execution_prices(week_sunday(val_weeks[-1]))
    return portfolio.total_value(final_prices) / INITIAL_CASH - 1.0


def _tune_params(strategy: str) -> dict:
    """
    Grid-search the validation period for the best hyperparameters.
    Returns the param dict that achieved the highest total return on VAL.
    """
    grid = _PARAM_GRIDS[strategy]
    best_ret, best_params = -np.inf, grid[0]
    logger.info("Tuning [%s] on %s → %s (%d combos)", strategy, VAL_START, VAL_END, len(grid))
    for params in grid:
        ret = _val_return(strategy, params)
        logger.info("  params=%-45s  val_return=%+.4f", str(params), ret)
        if ret > best_ret:
            best_ret, best_params = ret, params
    logger.info("Tuning [%s] best=%s (val_return=%+.4f)", strategy, best_params, best_ret)
    return best_params


# ── Portfolio ─────────────────────────────────────────────────────────────────

@dataclass
class Portfolio:
    cash: float = INITIAL_CASH
    holdings: dict = field(default_factory=dict)   # symbol → quantity
    cost_basis: dict = field(default_factory=dict) # symbol → USD cost
    initial_value: float = INITIAL_CASH

    def total_value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(
            self.holdings.get(sym, 0.0) * prices.get(sym, 0.0)
            for sym in UNIVERSE
        )

    def _overall_pnl(self, prices: dict[str, float]) -> tuple[float, float]:
        tv = self.total_value(prices)
        abs_pnl = tv - self.initial_value
        pct_pnl = abs_pnl / self.initial_value * 100.0
        return round(abs_pnl, 2), round(pct_pnl, 4)

    def _per_asset_detail(self, prices: dict[str, float]) -> dict:
        detail = {}
        for sym in UNIVERSE:
            qty = self.holdings.get(sym, 0.0)
            if qty <= 0:
                continue
            price = prices.get(sym, 0.0)
            value = round(qty * price, 2)
            basis = round(self.cost_basis.get(sym, 0.0), 2)
            pnl   = round(value - basis, 2)
            pct   = round(pnl / basis * 100.0, 4) if basis > 0 else 0.0
            detail[sym] = {
                "value_usd": value,
                "cost_basis_usd": basis,
                "pnl_usd": pnl,
                "pnl_pct": pct,
            }
        return detail

    def to_record(self, prices: dict[str, float]) -> dict:
        abs_pnl, pct_pnl = self._overall_pnl(prices)
        return {
            "cash":            round(self.cash, 2),
            "holdings_qty":    {k: round(v, 8) for k, v in self.holdings.items()},
            "holdings_detail": self._per_asset_detail(prices),
            "total_value":     round(self.total_value(prices), 2),
            "pnl_usd":         abs_pnl,
            "pnl_pct":         pct_pnl,
        }

    def apply_actions(self, actions: list[dict], prices: dict[str, float]) -> None:
        """
        Identical semantics to run_experiment.py Portfolio.apply_actions:
          action < 0 → sell |action| fraction of current holdings
          action > 0 → spend action fraction of post-sell cash pool to buy
          action = 0 → hold
        """
        # Phase 1: sells
        for item in actions:
            sym    = item.get("symbol")
            action = float(item.get("action", 0.0))
            price  = prices.get(sym)
            if price is None or price <= 0 or action >= 0:
                continue
            sell_qty = abs(action) * self.holdings.get(sym, 0.0)
            if sell_qty <= 0:
                continue
            net_usd = sell_qty * price * (1 - TRANSACTION_COST)
            self.holdings[sym] = self.holdings.get(sym, 0.0) - sell_qty
            self.cash += net_usd
            if sym in self.cost_basis:
                self.cost_basis[sym] *= 1.0 - abs(action)

        # Phase 2: buys from post-sell cash pool
        post_sell_cash = self.cash
        buy_items = [
            (item, float(item.get("action", 0.0)))
            for item in actions
            if float(item.get("action", 0.0)) > 0
            and prices.get(item.get("symbol"), 0) > 0
        ]
        if not buy_items:
            return

        desired = {
            item["symbol"]: frac * post_sell_cash for item, frac in buy_items
        }
        total_desired = sum(desired.values())
        scale = min(1.0, post_sell_cash / total_desired) if total_desired > 0 else 0.0

        for item, _ in buy_items:
            sym      = item["symbol"]
            price    = prices[sym]
            spend    = desired[sym] * scale
            if spend <= 0:
                continue
            net_usd  = spend * (1 - TRANSACTION_COST)
            quantity = net_usd / price
            self.cash -= spend
            self.holdings[sym]   = self.holdings.get(sym, 0.0) + quantity
            self.cost_basis[sym] = self.cost_basis.get(sym, 0.0) + spend


# ── Week helpers ──────────────────────────────────────────────────────────────

def generate_weeks(start: str, end: str) -> list[str]:
    mondays = pd.date_range(start, end, freq="W-MON")
    return [f"{d.isocalendar().year}-W{d.isocalendar().week:02d}" for d in mondays]


def week_sunday(week_str: str) -> pd.Timestamp:
    year, w = week_str.split("-W")
    monday  = pd.Timestamp.fromisocalendar(int(year), int(w), 1).tz_localize("UTC")
    return monday + pd.Timedelta(days=6)


# ── Market data helpers ───────────────────────────────────────────────────────

def get_execution_prices(as_of: pd.Timestamp) -> dict[str, float]:
    """Close price on or before `as_of` for each asset."""
    prices = {}
    for sym in UNIVERSE:
        df  = load_asset(sym)
        row = df.loc[:as_of]["close"].dropna()
        prices[sym] = float(row.iloc[-1]) if not row.empty else 0.0
    return prices


def get_mcap_weights(as_of: pd.Timestamp) -> dict[str, float]:
    """Return market-cap weights (summing to 1) for all assets as of `as_of`."""
    mcaps = {}
    for sym in UNIVERSE:
        df   = load_asset(sym)
        col  = df.loc[:as_of, "market_cap"].dropna()
        mcaps[sym] = float(col.iloc[-1]) if not col.empty else 0.0
    total = sum(mcaps.values())
    if total <= 0:
        n = len(UNIVERSE)
        return {sym: 1.0 / n for sym in UNIVERSE}
    return {sym: v / total for sym, v in mcaps.items()}


def get_indicator_row(sym: str, as_of: pd.Timestamp) -> dict | None:
    """Return the indicator values needed by signal strategies, as of `as_of`."""
    df    = load_asset(sym)
    close = df.loc[:as_of, "close"].dropna()
    if close.empty:
        return None

    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd      = ema12 - ema26
    macd_sig  = macd.ewm(span=9, adjust=False).mean()
    bb_sma20  = close.rolling(20).mean()
    bb_lower  = bb_sma20 - 2 * close.rolling(20).std()

    def last(s: pd.Series) -> float | None:
        v = s.iloc[-1]
        return float(v) if pd.notna(v) else None

    return {
        "close":     last(close),
        "sma_7":     last(close.rolling(7).mean()),
        "sma_30":    last(close.rolling(30).mean()),
        "macd_hist": last(macd - macd_sig),
        "bb_lower":  last(bb_lower),
    }


# ── Signal generation ─────────────────────────────────────────────────────────

def get_signals(as_of: pd.Timestamp, strategy: str) -> list[str]:
    """
    Return the list of symbols with a buy signal for the given strategy.

    sma7 : close > SMA(7)
    slma : SMA(7) > SMA(30)    [short MA above long MA]
    macd : MACD histogram > 0  [MACD line above signal line]
    """
    if strategy in ("btc_hold", "mcap_hold", "lstm") or strategy in _NF_REGISTRY:
        return []   # handled separately



    signals = []
    for sym in UNIVERSE:
        row = get_indicator_row(sym, as_of)
        if row is None:
            continue

        if strategy == "sma7":
            if pd.notna(row.get("sma_7")) and row["close"] > row["sma_7"]:
                signals.append(sym)

        elif strategy == "slma":
            if (pd.notna(row.get("sma_7")) and pd.notna(row.get("sma_30"))
                    and row["sma_7"] > row["sma_30"]):
                signals.append(sym)

        elif strategy == "macd":
            if pd.notna(row.get("macd_hist")) and row["macd_hist"] > 0:
                signals.append(sym)

        elif strategy == "bb":
            # Mean-reversion: long when price is below the lower Bollinger Band
            if pd.notna(row.get("bb_lower")) and row["close"] < row["bb_lower"]:
                signals.append(sym)

    return signals


# ── Action computation ────────────────────────────────────────────────────────

def compute_actions(
    strategy: str,
    portfolio: Portfolio,
    signals: list[str],
    mcap_weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Return a list of action dicts {symbol, action, rationale}.

    Buy-and-hold strategies only trade on the very first week
    (detected by an all-cash, no-holdings portfolio).

    Signal strategies do a full equal-weight rebalance every week:
      • sell 100 % of every asset not in the signal set
      • buy 1/N of each asset in the signal set from post-sell cash
    """
    is_uninitialised = (not portfolio.holdings and
                        abs(portfolio.cash - portfolio.initial_value) < 0.01)

    if strategy == "btc_hold":
        if is_uninitialised:
            return [{"symbol": "BTC", "action": 1.0,
                     "rationale": "Initial purchase: 100 % BTC"}]
        return []   # hold forever — no actions needed

    if strategy == "mcap_hold":
        if is_uninitialised:
            weights = mcap_weights or {}
            return [{"symbol": sym, "action": weights.get(sym, 0.0),
                     "rationale": f"Initial purchase: market-cap weight {weights.get(sym, 0.0):.4f}"}
                    for sym in UNIVERSE if weights.get(sym, 0.0) > 0]
        return []   # hold forever

    # ── Signal-based strategies: full rebalance ───────────────────────────────
    signal_set  = set(signals)
    actions = []

    # Sell held assets that are no longer in the signal set
    for sym, qty in portfolio.holdings.items():
        if qty > 0:
            actions.append({
                "symbol":    sym,
                "action":    -0.5,
                "rationale": "Sell: position exiting signal set" if sym not in signal_set
                             else "Rebalance: partial sell before re-entry",
            })

    # Buy all signal-set assets from post-sell cash
    n = len(signal_set)
    if n > 0:
        frac = 0.5
        for sym in sorted(signal_set):
            actions.append({
                "symbol":    sym,
                "action":    frac,
                "rationale": f"Buy: {strategy} signal — fraction {frac:.2f}",
            })

    return actions


# ── Main strategy runner ──────────────────────────────────────────────────────

def run_strategy(strategy: str, weeks: list[str]) -> None:
    combo_name = f"benchmark_{strategy}"
    out_dir    = OUTPUT_DIR / combo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Benchmark: %s  |  weeks: %d  |  out: %s",
                combo_name, len(weeks), out_dir)
    logger.info("=" * 60)

    # Determine which weeks still need processing to avoid unnecessary training
    weeks_todo = [w for w in weeks if not (out_dir / f"{w}.json").exists()]
    if not weeks_todo:
        logger.info("[%s] All %d weeks already computed — skipping.", combo_name, len(weeks))
        return

    # Tune hyperparameters on validation set (signal strategies only)
    strategy_params: dict = {}
    if strategy in _TUNE_STRATS:
        strategy_params = _tune_params(strategy)
        logger.info("[%s] Using params: %s", combo_name, strategy_params)

    # Train models once before the weekly loop (pre-backtest data only)
    lstm_ms: dict[str, tuple] = {}
    if strategy == "lstm":
        lstm_ms = train_lstm_models(train_end=BACKTEST_START)

    nf_model = None
    if strategy in _NF_REGISTRY:
        nf_model = train_neural_model(strategy, train_end=BACKTEST_START)

    portfolio  = Portfolio()
    state_path = out_dir / "_state.json"

    # Resume from checkpoint if available
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        p = saved["portfolio"]
        portfolio.cash          = p["cash"]
        portfolio.holdings      = p["holdings"]
        portfolio.cost_basis    = p.get("cost_basis", {})
        portfolio.initial_value = p.get("initial_value", INITIAL_CASH)
        logger.info("Resumed — cash=%.2f", portfolio.cash)

    for week in weeks:
        out_path = out_dir / f"{week}.json"

        # Skip already-computed weeks (restore portfolio state from saved record)
        if out_path.exists():
            logger.info("[%s] %s — already done, skipping", combo_name, week)
            rec   = json.loads(out_path.read_text())
            after = rec["portfolio_after"]
            portfolio.cash       = after["cash"]
            portfolio.holdings   = after.get("holdings_qty") or after.get("holdings", {})
            portfolio.cost_basis = after.get("cost_basis",
                {k: v["cost_basis_usd"]
                 for k, v in after.get("holdings_detail", {}).items()})
            continue

        logger.info("[%s] %s — processing", combo_name, week)

        sunday      = week_sunday(week)
        exec_prices = get_execution_prices(sunday)

        portfolio_before = portfolio.to_record(exec_prices)

        # Signals
        if strategy == "lstm":
            signals = get_lstm_signals(lstm_ms, sunday)
        elif strategy in _NF_REGISTRY:
            signals = get_neural_signals(nf_model, sunday)
        elif strategy in _TUNE_STRATS:
            signals = _get_signals_parametric(sunday, strategy, strategy_params)
        else:
            signals = get_signals(sunday, strategy)
        logger.debug("[%s] %s — signals: %s", combo_name, week,
                     signals if signals else "(none)")

        mcap_weights = get_mcap_weights(sunday) if strategy == "mcap_hold" else None
        actions = compute_actions(strategy, portfolio, signals, mcap_weights=mcap_weights)
        portfolio.apply_actions(actions, exec_prices)

        portfolio_after = portfolio.to_record(exec_prices)

        record = {
            "week":             week,
            "architecture":     "benchmark",
            "capability":       strategy,
            "portfolio_before": portfolio_before,
            "execution_prices": {k: round(v, 6) for k, v in exec_prices.items()},
            "signals":          signals,
            "trading_actions":  actions,
            "portfolio_after":  portfolio_after,
        }
        out_path.write_text(json.dumps(record, indent=2))

        tv_after  = portfolio_after["total_value"]
        tv_before = portfolio_before["total_value"]
        weekly_ret = (tv_after - tv_before) / tv_before * 100 if tv_before else 0.0
        logger.info("[%s] %s — value: $%.0f  weekly: %+.2f %%",
                    combo_name, week, tv_after, weekly_ret)

        # Checkpoint
        state_path.write_text(json.dumps({
            "portfolio": {
                "cash":          portfolio.cash,
                "holdings":      portfolio.holdings,
                "cost_basis":    portfolio.cost_basis,
                "initial_value": portfolio.initial_value,
            }
        }, indent=2))

    logger.info("[%s] Done.", combo_name)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run benchmark strategies for the CryptoMAS evaluation."
    )
    p.add_argument(
        "--strategy", choices=STRATEGIES, default=None,
        help="Single strategy to run (default: all)",
    )
    p.add_argument(
        "--weeks", nargs=2, metavar=("START", "END"),
        default=[BACKTEST_START, BACKTEST_END],
        help="ISO-week range as calendar dates (default: full 2025)",
    )
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    weeks      = generate_weeks(args.weeks[0], args.weeks[1])
    strategies = [args.strategy] if args.strategy else STRATEGIES

    logger.info("Running %d benchmark(s) over %d weeks", len(strategies), len(weeks))
    for strategy in strategies:
        run_strategy(strategy, weeks)
    logger.info("All benchmarks complete.")


if __name__ == "__main__":
    main()
