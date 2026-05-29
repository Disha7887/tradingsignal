-- ================================================================
-- Phase 2 — signals table
-- Run this once in Supabase: SQL Editor → paste → Run
-- ================================================================

CREATE TABLE IF NOT EXISTS signals (
  id                  UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at          TIMESTAMPTZ DEFAULT NOW(),

  -- signal identity
  symbol              TEXT        NOT NULL,
  timeframe           TEXT        NOT NULL,
  exchange            TEXT        NOT NULL DEFAULT 'BINANCE',
  candle_time         TIMESTAMPTZ,

  -- signal output
  direction           TEXT        NOT NULL,   -- LONG / SHORT / NEUTRAL
  confidence          NUMERIC,
  regime              TEXT,                    -- trend / range / volatile
  entry               NUMERIC,
  stop                NUMERIC,
  target              NUMERIC,
  reward_risk         NUMERIC,
  aggregate_score     NUMERIC,

  -- detail (stored as JSON strings)
  components          TEXT,
  rationale           TEXT,
  context             TEXT,

  -- outcome (filled in by /resolve)
  resolved            BOOLEAN     DEFAULT FALSE,
  outcome             TEXT,                    -- WIN / LOSS / EXPIRED
  exit_price          NUMERIC,
  exit_time           TIMESTAMPTZ,
  pnl_pct             NUMERIC,
  bars_to_resolution  INTEGER
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_signals_resolved    ON signals(resolved);
CREATE INDEX IF NOT EXISTS idx_signals_symbol      ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_direction   ON signals(direction);
CREATE INDEX IF NOT EXISTS idx_signals_created_at  ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_regime      ON signals(regime);

-- Useful view for metrics
CREATE OR REPLACE VIEW signal_metrics AS
SELECT
  symbol,
  regime,
  COUNT(*)                                              AS total,
  COUNT(*) FILTER (WHERE outcome = 'WIN')               AS wins,
  COUNT(*) FILTER (WHERE outcome = 'LOSS')              AS losses,
  COUNT(*) FILTER (WHERE outcome = 'EXPIRED')           AS expired,
  ROUND(AVG(CASE WHEN outcome='WIN'  THEN pnl_pct END)::NUMERIC, 3) AS avg_win_pct,
  ROUND(AVG(CASE WHEN outcome='LOSS' THEN pnl_pct END)::NUMERIC, 3) AS avg_loss_pct,
  ROUND(AVG(pnl_pct)::NUMERIC, 3)                       AS expectancy_pct,
  ROUND(
    SUM(CASE WHEN pnl_pct > 0 THEN pnl_pct ELSE 0 END) /
    NULLIF(ABS(SUM(CASE WHEN pnl_pct < 0 THEN pnl_pct ELSE 0 END)), 0),
    2
  )                                                      AS profit_factor
FROM signals
WHERE resolved = TRUE
GROUP BY symbol, regime
ORDER BY symbol, regime;
