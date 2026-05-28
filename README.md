# Trading Signal Service — Phase 1

Rule-based crypto signal engine. Runs as a FastAPI service on Railway.

## Files
- `signal_service.py` — the full engine (data, indicators, regime, scoring, risk)
- `requirements.txt` — Python dependencies
- `Procfile` — tells Railway how to start the app

## Deploy to Railway
1. Upload this repo to GitHub
2. Railway → New Project → Deploy from GitHub repo → select this repo
3. Railway auto-detects Python and runs the Procfile
4. Settings → Networking → Generate Domain to get your public URL

## Usage
```
GET /health
GET /signal?symbol=BTC/USDT&timeframe=1h&exchange=binance
```

## Tuning
Everything is in the `CONFIG` dict at the top of `signal_service.py`.
Change weights, thresholds, and periods there — no logic changes needed.

## Supported timeframes
5m, 15m, 30m, 1h, 4h, 1d
Higher-timeframe trend filter is applied automatically based on CONFIG["htf_map"].

## Not financial advice
This is decision-support only. Paper-trade before risking any capital.
