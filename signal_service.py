"""
Phase 1 trading signal service.

A transparent, rule-based crypto signal engine:
  CCXT data pull -> indicators -> regime classifier -> weighted scoring
  -> risk-managed, structured signal object, served over HTTP.

No ML yet by design. Get this running and measure its expectancy first;
only add a model once you have a rule baseline to beat.

Run locally:   uvicorn signal_service:app --host 0.0.0.0 --port 8000
Then call:     GET /signal?symbol=BTC/USDT&timeframe=1h

Everything tunable lives in CONFIG at the top. Change weights and
thresholds there, not scattered through the code.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    import ccxt
except Exception:  # ccxt is only needed at request time, not import time
    ccxt = None

from fastapi import FastAPI, Query

# --------------------------------------------------------------------------- #
# CONFIG  --  the only place you should need to edit to tune behaviour
# --------------------------------------------------------------------------- #
CONFIG = {
    # indicator periods
    "ema_fast": 50,
    "ema_slow": 200,
    "rsi_period": 14,
    "atr_period": 14,
    "adx_period": 14,
    "bb_period": 20,
    "bb_mult": 2.0,
    "vwap_period": 24,

    # regime thresholds
    "adx_trend_min": 25.0,        # ADX above this = trending market
    "atr_pct_volatile": 0.04,     # ATR / price above this = "too volatile, stand aside"

    # signal gating
    "enter_threshold": 0.25,      # |aggregate score| must exceed this to fire
    "max_confidence": 95.0,       # never claim certainty

    # risk (ATR multiples)
    "stop_atr_mult": 1.5,
    "reward_risk": 2.0,           # target distance = stop distance * this

    # confidence dampeners
    "regime_factor": {"trend": 1.0, "range": 0.9, "volatile": 0.5},
    "htf_agree": 1.0,
    "htf_conflict": 0.6,
    "htf_neutral": 0.85,

    # regime-dependent component weights (each must sum to ~1.0)
    "weights": {
        "trend":    {"trend": 0.50, "macd": 0.30, "rsi": 0.10, "bb": 0.00, "funding": 0.05, "fng": 0.05},
        "range":    {"trend": 0.10, "macd": 0.10, "rsi": 0.30, "bb": 0.25, "funding": 0.15, "fng": 0.10},
        "volatile": {"trend": 0.25, "macd": 0.20, "rsi": 0.15, "bb": 0.10, "funding": 0.15, "fng": 0.15},
    },

    # higher-timeframe trend filter: maps a timeframe to its context timeframe
    "htf_map": {"5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d", "1d": "1w"},

    "ohlcv_limit": 300,           # bars to fetch (>= ema_slow for warmup)
}


# --------------------------------------------------------------------------- #
# Indicators  --  computed by hand so there is no TA-Lib / build dependency
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used by RSI, ATR, ADX)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100)  # no losses in window -> momentum maxed out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    return rma(true_range(df), period)


def adx(df: pd.DataFrame, period: int):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_ = rma(true_range(df), period).replace(0, np.nan)
    plus_di = 100 * rma(plus_dm, period) / atr_
    minus_di = 100 * rma(minus_dm, period) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return rma(dx, period), plus_di, minus_di


def bollinger(close: pd.Series, period: int, mult: float):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid, mid + mult * sd, mid - mult * sd


def rolling_vwap(df: pd.DataFrame, period: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    return pv.rolling(period).sum() / df["volume"].rolling(period).sum()


# --------------------------------------------------------------------------- #
# Feature frame
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = CONFIG
    df = df.copy()
    df["ema_fast"] = ema(df["close"], c["ema_fast"])
    df["ema_slow"] = ema(df["close"], c["ema_slow"])
    df["rsi"] = rsi(df["close"], c["rsi_period"])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    df["atr"] = atr(df, c["atr_period"])
    df["adx"], df["plus_di"], df["minus_di"] = adx(df, c["adx_period"])
    df["bb_mid"], df["bb_up"], df["bb_low"] = bollinger(df["close"], c["bb_period"], c["bb_mult"])
    df["vwap"] = rolling_vwap(df, c["vwap_period"])
    return df


# --------------------------------------------------------------------------- #
# Regime
# --------------------------------------------------------------------------- #
def classify_regime(row: pd.Series) -> str:
    atr_pct = row["atr"] / row["close"] if row["close"] else 0.0
    if atr_pct > CONFIG["atr_pct_volatile"]:
        return "volatile"
    if row["adx"] >= CONFIG["adx_trend_min"]:
        return "trend"
    return "range"


def trend_sign(row: pd.Series) -> int:
    """+1 bullish stack, -1 bearish stack, 0 mixed -- used for HTF filter."""
    if row["close"] > row["ema_fast"] > row["ema_slow"]:
        return 1
    if row["close"] < row["ema_fast"] < row["ema_slow"]:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# Component scores, each in [-1, +1]  (+ = bullish, - = bearish)
# --------------------------------------------------------------------------- #
def _trend_score(row: pd.Series) -> float:
    if row["close"] > row["ema_fast"] > row["ema_slow"]:
        return 1.0
    if row["close"] < row["ema_fast"] < row["ema_slow"]:
        return -1.0
    if row["close"] > row["ema_fast"]:
        return 0.5
    if row["close"] < row["ema_fast"]:
        return -0.5
    return 0.0


def _macd_score(row: pd.Series) -> float:
    scale = max(row["atr"] * 0.3, 1e-9)
    return float(np.tanh((row["macd"] - row["macd_signal"]) / scale))


def _rsi_score(row: pd.Series, regime: str) -> float:
    if regime == "trend":  # momentum confirmation, mild
        return float(np.clip((row["rsi"] - 50) / 30, -1, 1))
    # range / volatile -> contrarian: oversold bullish, overbought bearish
    return float(np.clip((50 - row["rsi"]) / 20, -1, 1))


def _bb_score(row: pd.Series) -> float:
    width = row["bb_up"] - row["bb_mid"]
    if width <= 0:
        return 0.0
    pos = (row["close"] - row["bb_mid"]) / width  # ~+1 at upper band, -1 at lower
    return float(np.clip(-pos, -1, 1))            # mean reversion


def _funding_score(funding: Optional[float]) -> float:
    if funding is None:
        return 0.0
    # crowded longs (positive funding) -> bearish contrarian, and vice versa
    return float(np.clip(-funding / 0.0005, -1, 1))


def _fng_score(fng: Optional[int]) -> float:
    if fng is None:
        return 0.0
    return float(np.clip((50 - fng) / 30, -1, 1))  # fear bullish, greed bearish


# --------------------------------------------------------------------------- #
# Signal assembly
# --------------------------------------------------------------------------- #
def make_signal(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    funding: Optional[float],
    fng: Optional[int],
    fng_label: Optional[str],
    htf_sign: int,
) -> dict:
    c = CONFIG
    feat = build_features(df)
    last = feat.iloc[-1]
    regime = classify_regime(last)

    scores = {
        "trend": _trend_score(last),
        "macd": _macd_score(last),
        "rsi": _rsi_score(last, regime),
        "bb": _bb_score(last),
        "funding": _funding_score(funding),
        "fng": _fng_score(fng),
    }
    weights = c["weights"][regime]
    aggregate = float(sum(scores[k] * weights[k] for k in scores))

    # direction
    if aggregate > c["enter_threshold"]:
        direction = "LONG"
    elif aggregate < -c["enter_threshold"]:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    # higher-timeframe agreement factor
    sig_sign = 1 if direction == "LONG" else -1 if direction == "SHORT" else 0
    if htf_sign == 0 or sig_sign == 0:
        htf_factor = c["htf_neutral"]
    elif htf_sign == sig_sign:
        htf_factor = c["htf_agree"]
    else:
        htf_factor = c["htf_conflict"]

    confidence = abs(aggregate) * 100 * c["regime_factor"][regime] * htf_factor
    confidence = round(min(confidence, c["max_confidence"]), 1)

    # risk levels off ATR
    price = float(last["close"])
    atr_val = float(last["atr"])
    stop_dist = c["stop_atr_mult"] * atr_val
    if direction == "LONG":
        stop = price - stop_dist
        target = price + stop_dist * c["reward_risk"]
    elif direction == "SHORT":
        stop = price + stop_dist
        target = price - stop_dist * c["reward_risk"]
    else:
        stop = target = None

    # human-readable rationale: biggest contributors first
    contrib = sorted(
        ((k, scores[k] * weights[k]) for k in scores),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    rationale = [_explain(k, scores[k], last, funding, fng, fng_label)
                 for k, w in contrib if abs(w) > 0.01][:4]

    def r(x):
        return None if x is None else round(x, 2 if price > 5 else 6)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candle_time": last.name.isoformat() if hasattr(last.name, "isoformat") else str(last.name),
        "direction": direction,
        "confidence": confidence,
        "regime": regime,
        "entry": r(price),
        "stop": r(stop),
        "target": r(target),
        "reward_risk": c["reward_risk"] if direction != "NEUTRAL" else None,
        "aggregate_score": round(aggregate, 3),
        "rationale": rationale,
        "components": {k: round(scores[k], 3) for k in scores},
        "context": {
            "adx": round(float(last["adx"]), 1),
            "rsi": round(float(last["rsi"]), 1),
            "atr_pct": round(atr_val / price * 100, 2),
            "funding_rate": funding,
            "fear_greed": fng,
            "fear_greed_label": fng_label,
            "htf_trend": {1: "up", -1: "down", 0: "mixed"}[htf_sign],
        },
        "disclaimer": "Decision-support only. Not financial advice. Paper-trade before risking capital.",
    }


def _explain(key, score, row, funding, fng, fng_label) -> str:
    d = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
    if key == "trend":
        return f"Trend {d}: price {'above' if row['close'] > row['ema_fast'] else 'below'} EMA{CONFIG['ema_fast']}"
    if key == "macd":
        return f"MACD {d}: histogram {'positive' if row['macd_hist'] > 0 else 'negative'}"
    if key == "rsi":
        return f"RSI {d} at {row['rsi']:.0f}"
    if key == "bb":
        return f"Bollinger {d}: price near {'upper' if row['close'] > row['bb_mid'] else 'lower'} band"
    if key == "funding":
        return f"Funding {d} (contrarian) at {funding:.4%}" if funding is not None else "Funding n/a"
    if key == "fng":
        return f"Fear & Greed {d} (contrarian): {fng} ({fng_label})" if fng is not None else "Fear & Greed n/a"
    return key


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt is not installed")
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    # CRITICAL: drop the still-forming candle so no future data leaks in
    return df.iloc[:-1]


def fetch_funding(symbol: str) -> Optional[float]:
    try:
        base, quote = symbol.split("/")
        perp = f"{base}/{quote}:{quote}"
        ex = ccxt.bybit({"enableRateLimit": True})
        return float(ex.fetch_funding_rate(perp)["fundingRate"])
    except Exception:
        return None


def fetch_fng():
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return None, None


def htf_trend_sign(exchange_id: str, symbol: str, timeframe: str) -> int:
    htf = CONFIG["htf_map"].get(timeframe)
    if not htf:
        return 0
    try:
        df = fetch_ohlcv(exchange_id, symbol, htf, CONFIG["ohlcv_limit"])
        return trend_sign(build_features(df).iloc[-1])
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
app = FastAPI(title="Phase 1 Signal Service", version="1.0")



@app.get("/")
def root():
    return {
        "service": "Phase 1 Signal Service",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "signal": "/signal?symbol=BTC/USDT&timeframe=1h&exchange=binance"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "time": dt.datetime.now(dt.timezone.utc).isoformat()}


@app.get("/signal")
def signal(
    symbol: str = Query("BTC/USDT"),
    timeframe: str = Query("1h"),
    exchange: str = Query("bybit"),
):
    try:
        df = fetch_ohlcv(exchange, symbol, timeframe, CONFIG["ohlcv_limit"])
        if len(df) < CONFIG["ema_slow"]:
            return {"error": f"not enough data ({len(df)} bars) for EMA{CONFIG['ema_slow']}"}
        funding = fetch_funding(symbol)
        fng, fng_label = fetch_fng()
        htf_sign = htf_trend_sign(exchange, symbol, timeframe)
        return make_signal(df, symbol, timeframe, funding, fng, fng_label, htf_sign)
    except Exception as e:
        return {"error": str(e), "symbol": symbol, "timeframe": timeframe}
