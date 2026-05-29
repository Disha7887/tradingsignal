"""
Phase 2 trading signal service — Signal logging + Outcome tracking + Metrics

New vs Phase 1:
  - Every LONG/SHORT signal auto-logged to Supabase
  - /resolve  — checks unresolved signals, marks WIN/LOSS/EXPIRED using OHLCV
  - /metrics  — win rate, profit factor, expectancy, drawdown, by regime/symbol
  - /signals  — recent signal log with outcomes
  - /dashboard — HTML performance dashboard (open in browser)

Env vars needed in Railway → Variables:
  COINAPI_KEY    — from coinapi.io dashboard
  SUPABASE_URL   — https://xxxx.supabase.co
  SUPABASE_KEY   — service role key (Settings → API → service_role)
"""

from __future__ import annotations

import datetime as dt
import os
import json
from typing import Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "ema_fast": 50, "ema_slow": 200,
    "rsi_period": 14, "atr_period": 14, "adx_period": 14,
    "bb_period": 20, "bb_mult": 2.0, "vwap_period": 24,

    "adx_trend_min":    25.0,
    "atr_pct_volatile": 0.04,
    "enter_threshold":  0.25,
    "max_confidence":   95.0,
    "stop_atr_mult":    1.5,
    "reward_risk":      2.0,

    "regime_factor": {"trend": 1.0, "range": 0.9, "volatile": 0.5},
    "htf_agree": 1.0, "htf_conflict": 0.6, "htf_neutral": 0.85,

    "weights": {
        "trend":    {"trend":0.50,"macd":0.30,"rsi":0.10,"bb":0.00,"funding":0.05,"fng":0.05},
        "range":    {"trend":0.10,"macd":0.10,"rsi":0.30,"bb":0.25,"funding":0.15,"fng":0.10},
        "volatile": {"trend":0.25,"macd":0.20,"rsi":0.15,"bb":0.10,"funding":0.15,"fng":0.15},
    },

    "htf_map": {
        "1MIN":"5MIN","5MIN":"1HRS","15MIN":"1HRS",
        "30MIN":"4HRS","1HRS":"4HRS","4HRS":"1DAY","1DAY":"7DAY"
    },

    "ohlcv_limit":          301,
    "max_resolution_bars":   48,   # bars before a signal is marked EXPIRED (48×1h = 2 days)
    "auto_log":             True,  # log every LONG/SHORT signal automatically
}

TF_MAP = {
    "1m":"1MIN","3m":"3MIN","5m":"5MIN","15m":"15MIN","30m":"30MIN",
    "1h":"1HRS","2h":"2HRS","4h":"4HRS","6h":"6HRS","12h":"12HRS",
    "1d":"1DAY","1w":"7DAY",
    "1MIN":"1MIN","5MIN":"5MIN","15MIN":"15MIN","30MIN":"30MIN",
    "1HRS":"1HRS","4HRS":"4HRS","1DAY":"1DAY","7DAY":"7DAY",
}

# ═══════════════════════════════════════════════════════════════════════════════
# SUPABASE CLIENT  (pure HTTP, no extra library)
# ═══════════════════════════════════════════════════════════════════════════════
class Supabase:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL","").rstrip("/")
        self.key = os.getenv("SUPABASE_KEY","")

    def _h(self):
        if not self.url or not self.key:
            raise RuntimeError(
                "SUPABASE_URL or SUPABASE_KEY not set. "
                "Add both in Railway → Variables."
            )
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def insert(self, table: str, row: dict) -> dict:
        r = requests.post(
            f"{self.url}/rest/v1/{table}",
            headers=self._h(), json=row, timeout=15
        )
        r.raise_for_status()
        data = r.json()
        return data[0] if isinstance(data, list) and data else data

    def select(self, table: str, params: dict = None) -> list:
        r = requests.get(
            f"{self.url}/rest/v1/{table}",
            headers={**self._h(), "Prefer": ""},
            params=params or {}, timeout=15
        )
        r.raise_for_status()
        return r.json()

    def patch(self, table: str, match: dict, data: dict) -> list:
        params = {f"{k}": f"eq.{v}" for k, v in match.items()}
        r = requests.patch(
            f"{self.url}/rest/v1/{table}",
            headers=self._h(), params=params, json=data, timeout=15
        )
        r.raise_for_status()
        return r.json()

db = Supabase()

# ═══════════════════════════════════════════════════════════════════════════════
# COINAPI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def to_period(tf: str) -> str:
    p = TF_MAP.get(tf)
    if not p:
        raise ValueError(f"Unknown timeframe '{tf}'. Use: 1m 5m 15m 30m 1h 4h 1d")
    return p

def to_symbol_id(symbol: str, exchange: str) -> str:
    base, quote = symbol.upper().split("/")
    return f"{exchange.upper()}_SPOT_{base}_{quote}"

def coinapi_headers() -> dict:
    key = os.getenv("COINAPI_KEY","").strip()
    if not key:
        raise RuntimeError("COINAPI_KEY not set in Railway → Variables")
    return {"X-CoinAPI-Key": key, "Accept": "application/json"}

def fetch_ohlcv(symbol: str, period: str, exchange: str, limit: int) -> pd.DataFrame:
    sym_id = to_symbol_id(symbol, exchange)
    r = requests.get(
        f"https://rest.coinapi.io/v1/ohlcv/{sym_id}/latest",
        params={"period_id": period, "limit": limit},
        headers=coinapi_headers(), timeout=30
    )
    if r.status_code != 200:
        raise RuntimeError(f"CoinAPI {r.status_code}: {r.text[:300]}")
    data = r.json()
    if not data:
        raise RuntimeError(f"CoinAPI returned no data for {sym_id} {period}")
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["time_period_start"], utc=True)
    df = (df.rename(columns={
            "price_open":"open","price_high":"high",
            "price_low":"low","price_close":"close","volume_traded":"volume"
          })[["ts","open","high","low","close","volume"]]
          .sort_values("ts").set_index("ts"))
    return df.iloc[:-1]   # drop still-forming candle

def fetch_ohlcv_since(symbol: str, period: str, exchange: str, since: str) -> pd.DataFrame:
    """Fetch candles from a specific datetime (for outcome resolution)."""
    sym_id = to_symbol_id(symbol, exchange)
    r = requests.get(
        f"https://rest.coinapi.io/v1/ohlcv/{sym_id}/history",
        params={"period_id": period, "time_start": since, "limit": CONFIG["max_resolution_bars"] + 2},
        headers=coinapi_headers(), timeout=30
    )
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["time_period_start"], utc=True)
    df = (df.rename(columns={
            "price_open":"open","price_high":"high",
            "price_low":"low","price_close":"close","volume_traded":"volume"
          })[["ts","open","high","low","close","volume"]]
          .sort_values("ts").set_index("ts"))
    return df.iloc[:-1]  # drop still-forming

def fetch_fng():
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return None, None

# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rma(s, n): return s.ewm(alpha=1.0/n, adjust=False).mean()

def rsi(close, n):
    d = close.diff()
    return (100 - 100/(1 + rma(d.clip(lower=0),n) / rma(-d.clip(upper=0),n).replace(0,np.nan))).fillna(100)

def macd(close, fast=12, slow=26, sig=9):
    m = ema(close,fast) - ema(close,slow); s = ema(m,sig); return m, s, m-s

def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([df["high"]-df["low"],(df["high"]-pc).abs(),(df["low"]-pc).abs()],axis=1).max(axis=1)

def atr(df, n): return rma(true_range(df), n)

def adx(df, n):
    up = df["high"].diff(); dn = -df["low"].diff()
    pdm = pd.Series(np.where((up>dn)&(up>0),  up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn>up)&(dn>0), dn, 0.0), index=df.index)
    a   = rma(true_range(df),n).replace(0,np.nan)
    pdi = 100*rma(pdm,n)/a; mdi = 100*rma(mdm,n)/a
    dx  = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return rma(dx,n), pdi, mdi

def bollinger(close, n, mult):
    mid = close.rolling(n).mean(); sd = close.rolling(n).std()
    return mid, mid+mult*sd, mid-mult*sd

def rolling_vwap(df, n):
    tp = (df["high"]+df["low"]+df["close"])/3
    return (tp*df["volume"]).rolling(n).sum()/df["volume"].rolling(n).sum()

def build_features(df):
    c = CONFIG; df = df.copy()
    df["ema_fast"] = ema(df["close"],c["ema_fast"])
    df["ema_slow"] = ema(df["close"],c["ema_slow"])
    df["rsi"]      = rsi(df["close"],c["rsi_period"])
    df["macd"],df["macd_signal"],df["macd_hist"] = macd(df["close"])
    df["atr"]      = atr(df,c["atr_period"])
    df["adx"],df["plus_di"],df["minus_di"] = adx(df,c["adx_period"])
    df["bb_mid"],df["bb_up"],df["bb_low"]  = bollinger(df["close"],c["bb_period"],c["bb_mult"])
    df["vwap"]     = rolling_vwap(df,c["vwap_period"])
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# REGIME + SCORES
# ═══════════════════════════════════════════════════════════════════════════════
def classify_regime(row):
    if row["atr"]/row["close"] > CONFIG["atr_pct_volatile"]: return "volatile"
    if row["adx"] >= CONFIG["adx_trend_min"]:                return "trend"
    return "range"

def trend_sign(row):
    if row["close"]>row["ema_fast"]>row["ema_slow"]: return  1
    if row["close"]<row["ema_fast"]<row["ema_slow"]: return -1
    return 0

def htf_trend_sign(symbol, period, exchange):
    htf = CONFIG["htf_map"].get(period)
    if not htf: return 0
    try:
        df = fetch_ohlcv(symbol, htf, exchange, CONFIG["ohlcv_limit"])
        return trend_sign(build_features(df).iloc[-1])
    except Exception:
        return 0

def _trend_score(r):
    if r["close"]>r["ema_fast"]>r["ema_slow"]: return  1.0
    if r["close"]<r["ema_fast"]<r["ema_slow"]: return -1.0
    return  0.5 if r["close"]>r["ema_fast"] else -0.5

def _macd_score(r):
    return float(np.tanh((r["macd"]-r["macd_signal"])/max(r["atr"]*0.3,1e-9)))

def _rsi_score(r, regime):
    return float(np.clip((r["rsi"]-50)/30,-1,1)) if regime=="trend" \
      else float(np.clip((50-r["rsi"])/20,-1,1))

def _bb_score(r):
    w = r["bb_up"]-r["bb_mid"]
    return 0.0 if w<=0 else float(np.clip(-((r["close"]-r["bb_mid"])/w),-1,1))

def _fng_score(fng):
    return 0.0 if fng is None else float(np.clip((50-fng)/30,-1,1))

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════
def make_signal(df, symbol, timeframe, period, exchange, fng, fng_label, htf_sign):
    c = CONFIG
    feat = build_features(df); last = feat.iloc[-1]
    regime = classify_regime(last)

    scores = {
        "trend":   _trend_score(last),
        "macd":    _macd_score(last),
        "rsi":     _rsi_score(last, regime),
        "bb":      _bb_score(last),
        "funding": 0.0,
        "fng":     _fng_score(fng),
    }
    weights   = c["weights"][regime]
    aggregate = float(sum(scores[k]*weights[k] for k in scores))

    direction = ("LONG"    if aggregate >  c["enter_threshold"] else
                 "SHORT"   if aggregate < -c["enter_threshold"] else "NEUTRAL")

    sig_sign = 1 if direction=="LONG" else -1 if direction=="SHORT" else 0
    htf_f = (c["htf_neutral"] if htf_sign==0 or sig_sign==0 else
             c["htf_agree"]   if htf_sign==sig_sign else c["htf_conflict"])

    confidence = round(min(abs(aggregate)*100*c["regime_factor"][regime]*htf_f, c["max_confidence"]),1)

    price = float(last["close"]); atr_v = float(last["atr"]); sd = c["stop_atr_mult"]*atr_v
    stop   = (price-sd if direction=="LONG"  else price+sd  if direction=="SHORT"  else None)
    target = (price+sd*c["reward_risk"] if direction=="LONG"
              else price-sd*c["reward_risk"] if direction=="SHORT" else None)

    contrib = sorted(((k, scores[k]*weights[k]) for k in scores), key=lambda kv:abs(kv[1]), reverse=True)
    rationale = [_explain(k,scores[k],last,fng,fng_label) for k,w in contrib if abs(w)>0.01][:4]

    dp = lambda x: None if x is None else round(x, 2 if price>5 else 6)

    sig = {
        "symbol": symbol, "timeframe": timeframe, "exchange": exchange,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candle_time": str(last.name),
        "direction": direction, "confidence": confidence, "regime": regime,
        "entry": dp(price), "stop": dp(stop), "target": dp(target),
        "reward_risk": c["reward_risk"] if direction!="NEUTRAL" else None,
        "aggregate_score": round(aggregate,3),
        "rationale": rationale,
        "components": {k: round(scores[k],3) for k in scores},
        "context": {
            "adx": round(float(last["adx"]),1),
            "rsi": round(float(last["rsi"]),1),
            "atr_pct": round(atr_v/price*100,2),
            "fear_greed": fng, "fear_greed_label": fng_label,
            "htf_trend": {1:"up",-1:"down",0:"mixed"}[htf_sign],
        },
        "data_source": "CoinAPI",
        "disclaimer": "Decision-support only. Not financial advice. Paper-trade first.",
    }

    # Auto-log every LONG/SHORT signal to Supabase
    if c["auto_log"] and direction != "NEUTRAL":
        try:
            row = _signal_to_db_row(sig)
            saved = db.insert("signals", row)
            sig["log_id"] = saved.get("id") if isinstance(saved, dict) else None
        except Exception as e:
            sig["log_warning"] = f"Signal generated but DB log failed: {e}"

    return sig

def _signal_to_db_row(sig: dict) -> dict:
    return {
        "symbol":          sig["symbol"],
        "timeframe":       sig["timeframe"],
        "exchange":        sig["exchange"],
        "candle_time":     sig["candle_time"],
        "direction":       sig["direction"],
        "confidence":      sig["confidence"],
        "regime":          sig["regime"],
        "entry":           sig["entry"],
        "stop":            sig["stop"],
        "target":          sig["target"],
        "reward_risk":     sig["reward_risk"],
        "aggregate_score": sig["aggregate_score"],
        "components":      json.dumps(sig["components"]),
        "rationale":       json.dumps(sig["rationale"]),
        "context":         json.dumps(sig["context"]),
        "resolved":        False,
        "outcome":         None,
    }

def _explain(key, score, row, fng, fng_label):
    d = "bullish" if score>0 else "bearish" if score<0 else "neutral"
    if key=="trend": return f"Trend {d}: price {'above' if row['close']>row['ema_fast'] else 'below'} EMA{CONFIG['ema_fast']}"
    if key=="macd":  return f"MACD {d}: histogram {'positive' if row['macd_hist']>0 else 'negative'}"
    if key=="rsi":   return f"RSI {d} at {row['rsi']:.0f}"
    if key=="bb":    return f"Bollinger {d}: price near {'upper' if row['close']>row['bb_mid'] else 'lower'} band"
    if key=="fng":   return (f"Fear & Greed {d}: {fng} ({fng_label})" if fng else "F&G: n/a")
    return key

# ═══════════════════════════════════════════════════════════════════════════════
# OUTCOME RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════
def resolve_signal(row: dict) -> dict:
    """
    Walk candles since signal entry.
    LONG:  WIN if high >= target,  LOSS if low <= stop
    SHORT: WIN if low  <= target,  LOSS if high >= stop
    EXPIRED if neither hit within max_resolution_bars.
    """
    symbol    = row["symbol"]
    exchange  = row["exchange"]
    direction = row["direction"]
    entry     = float(row["entry"])
    stop      = float(row["stop"])
    target    = float(row["target"])
    period    = to_period(row["timeframe"])
    since     = row["candle_time"]

    try:
        candles = fetch_ohlcv_since(symbol, period, exchange, since)
    except Exception as e:
        return {"outcome": "ERROR", "note": str(e)}

    if candles.empty:
        return {"outcome": "EXPIRED", "bars": 0, "exit_price": None, "pnl_pct": None}

    # Skip the signal candle itself (index 0 = entry candle)
    for i, (ts, c) in enumerate(candles.iterrows()):
        if i == 0:
            continue
        if direction == "LONG":
            if float(c["high"]) >= target:
                pnl = (target-entry)/entry*100
                return {"outcome":"WIN",  "bars":i, "exit_price":round(target,4), "pnl_pct":round(pnl,3), "exit_time":str(ts)}
            if float(c["low"])  <= stop:
                pnl = (stop-entry)/entry*100
                return {"outcome":"LOSS", "bars":i, "exit_price":round(stop,4),   "pnl_pct":round(pnl,3), "exit_time":str(ts)}
        elif direction == "SHORT":
            if float(c["low"])  <= target:
                pnl = (entry-target)/entry*100
                return {"outcome":"WIN",  "bars":i, "exit_price":round(target,4), "pnl_pct":round(pnl,3), "exit_time":str(ts)}
            if float(c["high"]) >= stop:
                pnl = (entry-stop)/entry*100
                return {"outcome":"LOSS", "bars":i, "exit_price":round(stop,4),   "pnl_pct":round(pnl,3), "exit_time":str(ts)}

        if i >= CONFIG["max_resolution_bars"]:
            exit_p = float(candles.iloc[-1]["close"])
            pnl = ((exit_p-entry)/entry*100) if direction=="LONG" else ((entry-exit_p)/entry*100)
            return {"outcome":"EXPIRED","bars":i,"exit_price":round(exit_p,4),"pnl_pct":round(pnl,3),"exit_time":str(ts)}

    return {"outcome":"PENDING","bars":len(candles),"exit_price":None,"pnl_pct":None}

# ═══════════════════════════════════════════════════════════════════════════════
# METRICS CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
def calc_metrics(rows: list) -> dict:
    resolved = [r for r in rows if r.get("outcome") in ("WIN","LOSS","EXPIRED")]
    wins     = [r for r in resolved if r["outcome"]=="WIN"]
    losses   = [r for r in resolved if r["outcome"]=="LOSS"]
    expired  = [r for r in resolved if r["outcome"]=="EXPIRED"]

    if not resolved:
        return {"message": "No resolved signals yet. Run /resolve after signals accumulate."}

    win_pnls  = [float(r["pnl_pct"]) for r in wins    if r.get("pnl_pct") is not None]
    loss_pnls = [float(r["pnl_pct"]) for r in losses  if r.get("pnl_pct") is not None]
    exp_pnls  = [float(r["pnl_pct"]) for r in expired if r.get("pnl_pct") is not None]
    all_pnls  = win_pnls + loss_pnls + exp_pnls

    total      = len(resolved)
    win_rate   = round(len(wins)/total*100, 1) if total else 0
    avg_win    = round(np.mean(win_pnls),  3) if win_pnls  else 0
    avg_loss   = round(np.mean(loss_pnls), 3) if loss_pnls else 0
    gross_win  = sum(win_pnls)
    gross_loss = abs(sum(loss_pnls))
    pf         = round(gross_win/gross_loss, 2) if gross_loss else float("inf")
    expectancy = round(np.mean(all_pnls), 3) if all_pnls else 0

    # Equity curve + max drawdown
    equity = np.cumsum([0.0] + all_pnls)
    peak   = np.maximum.accumulate(equity)
    dd     = equity - peak
    max_dd = round(float(dd.min()), 3)

    # Breakdown by regime and symbol
    by_regime = {}
    for regime in ("trend","range","volatile"):
        sub = [r for r in resolved if r.get("regime")==regime]
        if sub:
            w = sum(1 for r in sub if r["outcome"]=="WIN")
            by_regime[regime] = {"total":len(sub),"wins":w,"win_rate":round(w/len(sub)*100,1)}

    by_symbol = {}
    for r in resolved:
        s = r.get("symbol","?")
        by_symbol.setdefault(s, {"total":0,"wins":0})
        by_symbol[s]["total"] += 1
        if r["outcome"]=="WIN": by_symbol[s]["wins"] += 1
    for s in by_symbol:
        t = by_symbol[s]["total"]
        by_symbol[s]["win_rate"] = round(by_symbol[s]["wins"]/t*100,1) if t else 0

    return {
        "summary": {
            "total_signals":    total,
            "wins":             len(wins),
            "losses":           len(losses),
            "expired":          len(expired),
            "win_rate_pct":     win_rate,
            "avg_win_pct":      avg_win,
            "avg_loss_pct":     avg_loss,
            "profit_factor":    pf,
            "expectancy_pct":   expectancy,
            "max_drawdown_pct": max_dd,
            "edge_detected":    pf > 1.2 and expectancy > 0,
        },
        "by_regime": by_regime,
        "by_symbol": by_symbol,
        "interpretation": _interpret(pf, expectancy, win_rate, total),
    }

def _interpret(pf, exp, wr, n) -> str:
    if n < 30:
        return f"Only {n} resolved signals — need at least 30 for statistical meaning. Keep logging."
    if pf < 1.0:
        return "Profit factor < 1.0: rules losing money on average. Review regime weights."
    if pf < 1.2:
        return "Marginal edge (PF 1.0–1.2): possible but weak. Increase sample size."
    if exp > 0 and pf >= 1.2:
        return f"Positive edge detected (PF={pf}, expectancy={exp}%). Consider paper trading at scale."
    return "Mixed results — continue accumulating signals."

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Trading Signal Service — Phase 2", version="2.0")

@app.get("/")
def root():
    return {
        "service": "Phase 2 Signal Service",
        "endpoints": {
            "/signal":    "Generate + auto-log a signal",
            "/resolve":   "Resolve outcomes for all pending signals (run hourly via n8n)",
            "/metrics":   "Performance stats — win rate, PF, expectancy, drawdown",
            "/signals":   "Recent signal log",
            "/health":    "Health + env check",
            "/dashboard": "HTML performance dashboard (open in browser)",
        }
    }

@app.get("/health")
def health():
    coinapi_ok  = bool(os.getenv("COINAPI_KEY","").strip())
    supabase_ok = bool(os.getenv("SUPABASE_URL","") and os.getenv("SUPABASE_KEY",""))
    return {
        "status":      "ok" if coinapi_ok and supabase_ok else "degraded",
        "coinapi_key": "set" if coinapi_ok  else "MISSING",
        "supabase":    "set" if supabase_ok else "MISSING — add SUPABASE_URL and SUPABASE_KEY",
        "time":        dt.datetime.now(dt.timezone.utc).isoformat(),
    }

@app.get("/signal")
def signal(
    symbol:    str = Query("BTC/USDT"),
    timeframe: str = Query("1h"),
    exchange:  str = Query("BINANCE"),
):
    try:
        period   = to_period(timeframe)
        df       = fetch_ohlcv(symbol, period, exchange, CONFIG["ohlcv_limit"])
        if len(df) < CONFIG["ema_slow"]:
            return {"error": f"Only {len(df)} bars — need {CONFIG['ema_slow']} for EMA{CONFIG['ema_slow']}"}
        fng, fng_label = fetch_fng()
        htf_sign = htf_trend_sign(symbol, period, exchange)
        return make_signal(df, symbol, timeframe, period, exchange, fng, fng_label, htf_sign)
    except Exception as e:
        return {"error": str(e), "symbol": symbol, "timeframe": timeframe}

@app.post("/resolve")
@app.get("/resolve")
def resolve(limit: int = Query(20, description="Max unresolved signals to check per run")):
    """
    Check unresolved signals and mark WIN/LOSS/EXPIRED.
    Call this every hour via n8n cron.
    """
    try:
        unresolved = db.select("signals", {
            "resolved": "eq.false",
            "direction": "neq.NEUTRAL",
            "order":    "created_at.asc",
            "limit":    str(limit),
            "select":   "id,symbol,exchange,timeframe,direction,entry,stop,target,candle_time,regime",
        })
    except Exception as e:
        return {"error": f"DB fetch failed: {e}"}

    if not unresolved:
        return {"message": "No unresolved signals", "resolved_count": 0}

    results = []
    for row in unresolved:
        res = resolve_signal(row)
        if res["outcome"] in ("WIN","LOSS","EXPIRED"):
            try:
                db.patch("signals", {"id": row["id"]}, {
                    "resolved":   True,
                    "outcome":    res["outcome"],
                    "exit_price": res.get("exit_price"),
                    "exit_time":  res.get("exit_time"),
                    "pnl_pct":    res.get("pnl_pct"),
                    "bars_to_resolution": res.get("bars"),
                })
                results.append({"id": row["id"], "symbol": row["symbol"],
                                 "direction": row["direction"], **res})
            except Exception as e:
                results.append({"id": row["id"], "error": str(e)})
        else:
            results.append({"id": row["id"], "symbol": row["symbol"],
                             "direction": row["direction"], **res})

    return {
        "resolved_count": sum(1 for r in results if r.get("outcome") in ("WIN","LOSS","EXPIRED")),
        "pending_count":  sum(1 for r in results if r.get("outcome") == "PENDING"),
        "results": results,
    }

@app.get("/metrics")
def metrics(symbol: str = Query(None), regime: str = Query(None)):
    try:
        params = {"resolved": "eq.true", "select": "*", "order": "created_at.desc", "limit": "500"}
        if symbol: params["symbol"] = f"eq.{symbol}"
        if regime: params["regime"] = f"eq.{regime}"
        rows = db.select("signals", params)
        return calc_metrics(rows)
    except Exception as e:
        return {"error": str(e)}

@app.get("/signals")
def signals_log(limit: int = Query(50), resolved_only: bool = Query(False)):
    try:
        params = {"select": "id,created_at,symbol,timeframe,direction,confidence,regime,entry,stop,target,outcome,pnl_pct,resolved",
                  "order": "created_at.desc", "limit": str(limit)}
        if resolved_only:
            params["resolved"] = "eq.true"
        return db.select("signals", params)
    except Exception as e:
        return {"error": str(e)}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    try:
        rows = db.select("signals", {"select": "*", "order": "created_at.desc", "limit": "200"})
        m    = calc_metrics([r for r in rows if r.get("resolved")])
        sm   = m.get("summary", {})
    except Exception as e:
        rows = []; sm = {}; m = {"error": str(e)}

    recent = rows[:20]
    def badge(outcome):
        colors = {"WIN":"#22c55e","LOSS":"#ef4444","EXPIRED":"#f59e0b","PENDING":"#6b7280"}
        return f'<span style="background:{colors.get(outcome,"#374151")};padding:2px 8px;border-radius:4px;font-size:12px">{outcome or "PENDING"}</span>'

    rows_html = "".join(f"""<tr>
      <td>{r.get("created_at","")[:16]}</td>
      <td>{r.get("symbol","")}</td>
      <td>{r.get("timeframe","")}</td>
      <td style="color:{'#22c55e' if r.get('direction')=='LONG' else '#ef4444'}">{r.get("direction","")}</td>
      <td>{r.get("confidence","")}%</td>
      <td>{r.get("regime","")}</td>
      <td>{r.get("entry","")}</td>
      <td>{r.get("stop","")}</td>
      <td>{r.get("target","")}</td>
      <td>{badge(r.get("outcome"))}</td>
      <td style="color:{'#22c55e' if (r.get('pnl_pct') or 0)>0 else '#ef4444'}">{r.get("pnl_pct","—")}</td>
    </tr>""" for r in recent)

    pf_color  = "#22c55e" if (sm.get("profit_factor",0) or 0) >= 1.2 else "#ef4444"
    exp_color = "#22c55e" if (sm.get("expectancy_pct",0) or 0) > 0   else "#ef4444"

    return f"""<!DOCTYPE html><html><head><title>Signal Dashboard</title>
<meta charset="utf-8"><style>
  body{{background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;margin:0;padding:24px}}
  h1{{color:#f8fafc;margin-bottom:4px}} p.sub{{color:#94a3b8;margin:0 0 24px}}
  .cards{{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:32px}}
  .card{{background:#1e293b;border-radius:12px;padding:20px 24px;min-width:160px}}
  .card .val{{font-size:28px;font-weight:700;margin:4px 0}}
  .card .lbl{{font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
  table{{width:100%;border-collapse:collapse;background:#1e293b;border-radius:12px;overflow:hidden}}
  th{{background:#0f172a;padding:10px 12px;text-align:left;font-size:12px;color:#94a3b8;text-transform:uppercase}}
  td{{padding:10px 12px;font-size:13px;border-top:1px solid #0f172a}}
  .interp{{background:#1e293b;border-radius:12px;padding:16px 20px;margin-bottom:24px;color:#cbd5e1;border-left:4px solid #3b82f6}}
</style></head><body>
<h1>Signal Performance Dashboard</h1>
<p class="sub">Phase 2 — rule-based signal engine | updated {dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>
<div class="cards">
  <div class="card"><div class="lbl">Total Signals</div><div class="val">{sm.get("total_signals","—")}</div></div>
  <div class="card"><div class="lbl">Win Rate</div><div class="val" style="color:#22c55e">{sm.get("win_rate_pct","—")}%</div></div>
  <div class="card"><div class="lbl">Profit Factor</div><div class="val" style="color:{pf_color}">{sm.get("profit_factor","—")}</div></div>
  <div class="card"><div class="lbl">Expectancy</div><div class="val" style="color:{exp_color}">{sm.get("expectancy_pct","—")}%</div></div>
  <div class="card"><div class="lbl">Max Drawdown</div><div class="val" style="color:#f59e0b">{sm.get("max_drawdown_pct","—")}%</div></div>
  <div class="card"><div class="lbl">Avg Win</div><div class="val" style="color:#22c55e">{sm.get("avg_win_pct","—")}%</div></div>
  <div class="card"><div class="lbl">Avg Loss</div><div class="val" style="color:#ef4444">{sm.get("avg_loss_pct","—")}%</div></div>
</div>
<div class="interp">{m.get("interpretation", m.get("message","Accumulating data..."))}</div>
<h2 style="margin-bottom:12px">Recent Signals</h2>
<table><thead><tr>
  <th>Time</th><th>Symbol</th><th>TF</th><th>Dir</th><th>Conf</th>
  <th>Regime</th><th>Entry</th><th>Stop</th><th>Target</th><th>Outcome</th><th>PnL%</th>
</tr></thead><tbody>{rows_html}</tbody></table>
</body></html>"""
