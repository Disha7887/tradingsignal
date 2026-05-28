"""
Phase 1 trading signal service — CoinAPI edition.

Data source: CoinAPI (no geo-blocking, pulls from Binance/Bybit/etc via their infra)
Indicators: computed by hand (no TA-Lib needed)
Signal: rule-based weighted scoring with regime gating

Run:  uvicorn signal_service:app --host 0.0.0.0 --port $PORT
Env:  COINAPI_KEY=your_key_here  (set in Railway → Variables)

Endpoints:
  GET /                   service info
  GET /health             liveness check
  GET /signal?symbol=BTC/USDT&timeframe=1h&exchange=BINANCE
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query

# --------------------------------------------------------------------------- #
# CONFIG  —  tune everything here, never scattered in the logic below
# --------------------------------------------------------------------------- #
CONFIG = {
    "ema_fast":     50,
    "ema_slow":     200,
    "rsi_period":   14,
    "atr_period":   14,
    "adx_period":   14,
    "bb_period":    20,
    "bb_mult":      2.0,
    "vwap_period":  24,

    "adx_trend_min":    25.0,
    "atr_pct_volatile": 0.04,

    "enter_threshold": 0.25,
    "max_confidence":  95.0,

    "stop_atr_mult": 1.5,
    "reward_risk":   2.0,

    "regime_factor": {"trend": 1.0, "range": 0.9, "volatile": 0.5},
    "htf_agree":     1.0,
    "htf_conflict":  0.6,
    "htf_neutral":   0.85,

    "weights": {
        "trend":    {"trend": 0.50, "macd": 0.30, "rsi": 0.10, "bb": 0.00, "funding": 0.05, "fng": 0.05},
        "range":    {"trend": 0.10, "macd": 0.10, "rsi": 0.30, "bb": 0.25, "funding": 0.15, "fng": 0.10},
        "volatile": {"trend": 0.25, "macd": 0.20, "rsi": 0.15, "bb": 0.10, "funding": 0.15, "fng": 0.15},
    },

    # CoinAPI timeframe for higher-timeframe trend filter
    "htf_map": {
        "1MIN": "5MIN", "5MIN": "1HRS", "15MIN": "1HRS",
        "30MIN": "4HRS", "1HRS": "4HRS", "4HRS": "1DAY", "1DAY": "7DAY"
    },

    "ohlcv_limit": 301,   # fetch one extra — last candle (still forming) gets dropped
}

# --------------------------------------------------------------------------- #
# CoinAPI helpers
# --------------------------------------------------------------------------- #
TF_MAP = {
    "1m":  "1MIN",  "3m":  "3MIN",  "5m":  "5MIN",
    "15m": "15MIN", "30m": "30MIN",
    "1h":  "1HRS",  "2h":  "2HRS",  "4h":  "4HRS",
    "6h":  "6HRS",  "12h": "12HRS",
    "1d":  "1DAY",  "1w":  "7DAY",
    # also accept CoinAPI native codes directly
    "1MIN": "1MIN", "5MIN": "5MIN", "15MIN": "15MIN", "30MIN": "30MIN",
    "1HRS": "1HRS", "4HRS": "4HRS", "1DAY": "1DAY",  "7DAY": "7DAY",
}

def to_period(timeframe: str) -> str:
    p = TF_MAP.get(timeframe)
    if not p:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Use: 1m 5m 15m 30m 1h 4h 1d")
    return p

def to_symbol_id(symbol: str, exchange: str) -> str:
    """BTC/USDT + BINANCE  ->  BINANCE_SPOT_BTC_USDT"""
    if "/" not in symbol:
        raise ValueError(f"Symbol must be BASE/QUOTE format, got '{symbol}'")
    base, quote = symbol.upper().split("/")
    return f"{exchange.upper()}_SPOT_{base}_{quote}"

def coinapi_headers() -> dict:
    key = os.getenv("COINAPI_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "COINAPI_KEY env var not set. "
            "Add it in Railway: service -> Variables -> COINAPI_KEY"
        )
    return {"X-CoinAPI-Key": key, "Accept": "application/json"}

# --------------------------------------------------------------------------- #
# Data fetching — CoinAPI only, no CCXT
# --------------------------------------------------------------------------- #
def fetch_ohlcv(symbol: str, timeframe: str, exchange: str, limit: int) -> pd.DataFrame:
    period   = to_period(timeframe)
    sym_id   = to_symbol_id(symbol, exchange)
    url      = f"https://rest.coinapi.io/v1/ohlcv/{sym_id}/latest"
    params   = {"period_id": period, "limit": limit}

    r = requests.get(url, params=params, headers=coinapi_headers(), timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"CoinAPI {r.status_code}: {r.text[:300]}")

    data = r.json()
    if not data:
        raise RuntimeError(f"CoinAPI returned no data for {sym_id} {period}")

    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["time_period_start"], utc=True)
    df = (df.rename(columns={
            "price_open":   "open",
            "price_high":   "high",
            "price_low":    "low",
            "price_close":  "close",
            "volume_traded":"volume",
          })
          [["ts", "open", "high", "low", "close", "volume"]]
          .sort_values("ts")
          .set_index("ts"))

    # Drop the still-forming candle — never use partial data
    return df.iloc[:-1]


def fetch_fng():
    """Fear & Greed index — free, no key, no geo-block."""
    try:
        d = requests.get(
            "https://api.alternative.me/fng/?limit=1", timeout=10
        ).json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return None, None


def htf_trend_sign(symbol: str, period: str, exchange: str) -> int:
    htf = CONFIG["htf_map"].get(period)
    if not htf:
        return 0
    try:
        df = fetch_ohlcv(symbol, htf, exchange, CONFIG["ohlcv_limit"])
        return trend_sign(build_features(df).iloc[-1])
    except Exception:
        return 0

# --------------------------------------------------------------------------- #
# Indicators  —  hand-rolled, no TA-Lib / build dependency
# --------------------------------------------------------------------------- #
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing used by RSI / ATR / ADX."""
    return s.ewm(alpha=1.0/n, adjust=False).mean()

def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    ag = rma(d.clip(lower=0), n)
    al = rma(-d.clip(upper=0), n)
    return (100 - 100 / (1 + ag / al.replace(0, np.nan))).fillna(100)

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    return m, s, m - s

def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"]-df["low"],
                      (df["high"]-pc).abs(),
                      (df["low"] -pc).abs()], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return rma(true_range(df), n)

def adx(df: pd.DataFrame, n: int):
    up   = df["high"].diff()
    down = -df["low"].diff()
    pdm  = pd.Series(np.where((up>down)&(up>0),   up,   0.0), index=df.index)
    mdm  = pd.Series(np.where((down>up)&(down>0), down, 0.0), index=df.index)
    atr_ = rma(true_range(df), n).replace(0, np.nan)
    pdi  = 100 * rma(pdm, n) / atr_
    mdi  = 100 * rma(mdm, n) / atr_
    dx   = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    return rma(dx, n), pdi, mdi

def bollinger(close: pd.Series, n: int, mult: float):
    mid = close.rolling(n).mean()
    sd  = close.rolling(n).std()
    return mid, mid+mult*sd, mid-mult*sd

def rolling_vwap(df: pd.DataFrame, n: int) -> pd.Series:
    tp = (df["high"]+df["low"]+df["close"]) / 3
    return (tp*df["volume"]).rolling(n).sum() / df["volume"].rolling(n).sum()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = CONFIG
    df = df.copy()
    df["ema_fast"]                    = ema(df["close"], c["ema_fast"])
    df["ema_slow"]                    = ema(df["close"], c["ema_slow"])
    df["rsi"]                         = rsi(df["close"], c["rsi_period"])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    df["atr"]                         = atr(df, c["atr_period"])
    df["adx"], df["plus_di"], df["minus_di"] = adx(df, c["adx_period"])
    df["bb_mid"], df["bb_up"], df["bb_low"]  = bollinger(df["close"], c["bb_period"], c["bb_mult"])
    df["vwap"]                        = rolling_vwap(df, c["vwap_period"])
    return df

# --------------------------------------------------------------------------- #
# Regime
# --------------------------------------------------------------------------- #
def classify_regime(row: pd.Series) -> str:
    atr_pct = row["atr"] / row["close"] if row["close"] else 0.0
    if atr_pct > CONFIG["atr_pct_volatile"]: return "volatile"
    if row["adx"] >= CONFIG["adx_trend_min"]:  return "trend"
    return "range"

def trend_sign(row: pd.Series) -> int:
    if row["close"] > row["ema_fast"] > row["ema_slow"]: return  1
    if row["close"] < row["ema_fast"] < row["ema_slow"]: return -1
    return 0

# --------------------------------------------------------------------------- #
# Component scores  [-1, +1]
# --------------------------------------------------------------------------- #
def _trend_score(row):
    if row["close"] > row["ema_fast"] > row["ema_slow"]: return  1.0
    if row["close"] < row["ema_fast"] < row["ema_slow"]: return -1.0
    if row["close"] > row["ema_fast"]: return  0.5
    if row["close"] < row["ema_fast"]: return -0.5
    return 0.0

def _macd_score(row):
    scale = max(row["atr"] * 0.3, 1e-9)
    return float(np.tanh((row["macd"] - row["macd_signal"]) / scale))

def _rsi_score(row, regime):
    if regime == "trend":
        return float(np.clip((row["rsi"]-50)/30, -1, 1))
    return float(np.clip((50-row["rsi"])/20, -1, 1))

def _bb_score(row):
    width = row["bb_up"] - row["bb_mid"]
    if width <= 0: return 0.0
    return float(np.clip(-((row["close"]-row["bb_mid"])/width), -1, 1))

def _fng_score(fng):
    if fng is None: return 0.0
    return float(np.clip((50-fng)/30, -1, 1))

# --------------------------------------------------------------------------- #
# Signal assembly
# --------------------------------------------------------------------------- #
def make_signal(df, symbol, timeframe, fng, fng_label, htf_sign) -> dict:
    c    = CONFIG
    feat = build_features(df)
    last = feat.iloc[-1]
    regime = classify_regime(last)

    scores = {
        "trend":   _trend_score(last),
        "macd":    _macd_score(last),
        "rsi":     _rsi_score(last, regime),
        "bb":      _bb_score(last),
        "funding": 0.0,          # funding rates not in CoinAPI; neutral
        "fng":     _fng_score(fng),
    }
    weights   = c["weights"][regime]
    aggregate = float(sum(scores[k]*weights[k] for k in scores))

    if   aggregate >  c["enter_threshold"]: direction = "LONG"
    elif aggregate < -c["enter_threshold"]: direction = "SHORT"
    else:                                   direction = "NEUTRAL"

    sig_sign = 1 if direction=="LONG" else -1 if direction=="SHORT" else 0
    if htf_sign == 0 or sig_sign == 0: htf_f = c["htf_neutral"]
    elif htf_sign == sig_sign:          htf_f = c["htf_agree"]
    else:                               htf_f = c["htf_conflict"]

    confidence = round(
        min(abs(aggregate)*100 * c["regime_factor"][regime] * htf_f, c["max_confidence"]), 1
    )

    price    = float(last["close"])
    atr_val  = float(last["atr"])
    stop_d   = c["stop_atr_mult"] * atr_val
    if direction == "LONG":
        stop, target = price - stop_d, price + stop_d*c["reward_risk"]
    elif direction == "SHORT":
        stop, target = price + stop_d, price - stop_d*c["reward_risk"]
    else:
        stop = target = None

    contrib  = sorted(((k, scores[k]*weights[k]) for k in scores),
                      key=lambda kv: abs(kv[1]), reverse=True)
    rationale = [_explain(k, scores[k], last, fng, fng_label)
                 for k, w in contrib if abs(w) > 0.01][:4]

    dp = 2 if price > 5 else 6
    r  = lambda x: None if x is None else round(x, dp)

    return {
        "symbol":         symbol,
        "timeframe":      timeframe,
        "timestamp":      dt.datetime.now(dt.timezone.utc).isoformat(),
        "candle_time":    str(last.name),
        "direction":      direction,
        "confidence":     confidence,
        "regime":         regime,
        "entry":          r(price),
        "stop":           r(stop),
        "target":         r(target),
        "reward_risk":    c["reward_risk"] if direction != "NEUTRAL" else None,
        "aggregate_score":round(aggregate, 3),
        "rationale":      rationale,
        "components":     {k: round(scores[k], 3) for k in scores},
        "context": {
            "adx":              round(float(last["adx"]), 1),
            "rsi":              round(float(last["rsi"]), 1),
            "atr_pct":          round(atr_val/price*100, 2),
            "fear_greed":       fng,
            "fear_greed_label": fng_label,
            "htf_trend":        {1:"up", -1:"down", 0:"mixed"}[htf_sign],
        },
        "data_source": "CoinAPI",
        "disclaimer":  "Decision-support only. Not financial advice. Paper-trade first.",
    }

def _explain(key, score, row, fng, fng_label) -> str:
    d = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
    if key == "trend":
        side = "above" if row["close"] > row["ema_fast"] else "below"
        return f"Trend {d}: price {side} EMA{CONFIG['ema_fast']}"
    if key == "macd":
        return f"MACD {d}: histogram {'positive' if row['macd_hist']>0 else 'negative'}"
    if key == "rsi":
        return f"RSI {d} at {row['rsi']:.0f}"
    if key == "bb":
        side = "upper" if row["close"] > row["bb_mid"] else "lower"
        return f"Bollinger {d}: price near {side} band"
    if key == "fng":
        return (f"Fear & Greed {d} (contrarian): {fng} ({fng_label})"
                if fng is not None else "Fear & Greed: n/a")
    return key

# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
app = FastAPI(title="Phase 1 Signal Service — CoinAPI", version="2.0")

@app.get("/")
def root():
    return {
        "service":   "Phase 1 Signal Service",
        "version":   "2.0 (CoinAPI)",
        "status":    "running",
        "endpoints": {
            "health": "/health",
            "signal": "/signal?symbol=BTC/USDT&timeframe=1h&exchange=BINANCE",
        },
        "timeframes": "1m 5m 15m 30m 1h 4h 1d",
        "exchanges":  "BINANCE BYBIT KRAKEN COINBASE (any CoinAPI exchange code)",
    }

@app.get("/health")
def health():
    key_set = bool(os.getenv("COINAPI_KEY", "").strip())
    return {
        "status":       "ok",
        "coinapi_key":  "set" if key_set else "MISSING — set COINAPI_KEY in Railway Variables",
        "time":         dt.datetime.now(dt.timezone.utc).isoformat(),
    }

@app.get("/signal")
def signal(
    symbol:    str = Query("BTC/USDT",  description="e.g. BTC/USDT  ETH/USDT"),
    timeframe: str = Query("1h",        description="1m 5m 15m 30m 1h 4h 1d"),
    exchange:  str = Query("BINANCE",   description="BINANCE BYBIT KRAKEN COINBASE etc."),
):
    try:
        period = to_period(timeframe)
        df     = fetch_ohlcv(symbol, period, exchange, CONFIG["ohlcv_limit"])

        if len(df) < CONFIG["ema_slow"]:
            return {"error": f"Only {len(df)} bars returned — need {CONFIG['ema_slow']} for EMA{CONFIG['ema_slow']}. Try a shorter timeframe or higher limit."}

        fng, fng_label = fetch_fng()
        htf_sign       = htf_trend_sign(symbol, period, exchange)

        return make_signal(df, symbol, timeframe, fng, fng_label, htf_sign)

    except Exception as e:
        return {"error": str(e), "symbol": symbol, "timeframe": timeframe, "exchange": exchange}
