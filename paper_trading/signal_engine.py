"""
paper_trading/signal_engine.py — Signal generation engine for live paper trading.

Responsibilities
----------------
1. Train one CalibratedRF model per asset from its full 4-year historical CSV.
2. Maintain a rolling buffer of recent 1H OHLCV bars per asset.
3. Score a newly completed 1H bar: run features → regime → signal → apply filters.
4. Return (signal, buy_prob, sell_prob, atr_pct) for the new bar.

Signal pipeline (mirrors main.py)
----------------------------------
  generate_features(buffer)           — technical features
  add_regime_columns(buffer)          — trend + vol_regime + regime_gate
  add_session_column(buffer)          — session bucket
  apply_signals(model, feature_cols)  — predict_proba → signal column
  apply_trend_filter                  — block non-ALLOWED_TRENDS
  apply_vol_filter                    — block low_vol if BLOCK_LOW_VOL
  → return signal at last bar

Training
--------
  Uses fit_model() on the FULL historical dataset (no train/test split) so the
  most recent history informs the live model.  add_labels() is called only for
  training; the scoring path never needs forward-return labels.

Scoring
-------
  The buffer holds the last WARMUP_1H_BARS bars of 1H OHLCV.  New bars are
  appended as they arrive and old bars are trimmed.  All features are recomputed
  on the full buffer each time (300 rows × 35 features ≈ <1ms on modern hardware).
"""

import os
from typing import Optional

import pandas as pd

import config as pipeline_config
from data.loader import load_ohlcv
from features.generator import generate_features, add_labels
from features.market_regime import (
    add_regime_columns, add_session_column,
    apply_trend_filter, apply_vol_filter,
)
from models.classifier import fit_model, get_feature_columns, apply_signals

from paper_trading import config_live
from paper_trading.logger import get_logger

log = get_logger()


class SignalEngine:
    """
    One instance per paper-trading session.  Holds one model per asset.

    After construction call `train_all()` before the polling loop starts.
    """

    def __init__(self) -> None:
        self._models: dict[str, object]            = {}   # asset → fitted model
        self._feature_cols: dict[str, list[str]]   = {}   # asset → feature col list
        self._buffers: dict[str, pd.DataFrame]     = {}   # asset → rolling OHLCV buffer
        self._bars_since_retrain: dict[str, int]   = {}   # asset → counter
        self._atr_at_signal: dict[str, float]      = {}   # asset → atr_pct of last signal bar

    # ── Public API ─────────────────────────────────────────────────────────────

    def train_all(self) -> None:
        """Load historical CSVs and train one model per asset."""
        log.info(
            "Training basis: bars in [%s, %s)  (config.TRAINING_START_DATE / "
            "TRAINING_CUTOFF_DATE).  Live scoring + parity test use bars at or "
            "after the cutoff.",
            pipeline_config.TRAINING_START_DATE,
            pipeline_config.TRAINING_CUTOFF_DATE,
        )
        if config_live.RETRAIN_EVERY_N_BARS > 0:
            log.warning(
                "RETRAIN_EVERY_N_BARS=%d is enabled — periodic retrain uses the "
                "rolling buffer, which contains post-cutoff bars.  This BREAKS "
                "parity with backtest.  Keep retrain disabled (0) for parity runs.",
                config_live.RETRAIN_EVERY_N_BARS,
            )
        for asset in config_live.ASSETS:
            self._train_asset(asset)

    def push_bar(self, asset: str, bar: pd.Series) -> None:
        """
        Append a newly completed 1H bar to the asset's rolling buffer.

        `bar` must be a pd.Series with a DatetimeIndex timestamp as its name
        and columns: open, high, low, close, volume.
        """
        if asset not in self._buffers:
            log.warning("push_bar: no buffer for %s — skipping", asset)
            return

        new_row = pd.DataFrame([bar], index=[bar.name])[
            ["open", "high", "low", "close", "volume"]
        ]
        buf = self._buffers[asset]

        # Deduplicate: skip if this timestamp is already in the buffer
        if bar.name in buf.index:
            return

        buf = pd.concat([buf, new_row]).sort_index()
        # Keep last WARMUP_1H_BARS rows
        if len(buf) > config_live.WARMUP_1H_BARS:
            buf = buf.iloc[-config_live.WARMUP_1H_BARS:]
        self._buffers[asset] = buf

        self._bars_since_retrain[asset] = self._bars_since_retrain.get(asset, 0) + 1

    def score_bar(self, asset: str) -> Optional[dict]:
        """
        Score the most recent bar in the asset's buffer.

        Returns a dict:
            signal   : int   — +1 long, -1 short, 0 flat
            buy_prob : float — P(BUY) from the calibrated model
            sell_prob: float — P(SELL) from the calibrated model
            atr_pct  : float — ATR/close at the scored bar (for SL/TP sizing)
            close    : float — close price of the scored bar
            bar_ts   : pd.Timestamp — timestamp of the scored bar

        Returns None if the model is not trained or buffer is too short.
        """
        if asset not in self._models:
            log.warning("score_bar: model not trained for %s", asset)
            return None

        buf = self._buffers.get(asset)
        if buf is None or len(buf) < 50:
            log.warning("score_bar: buffer too short for %s (%d bars)",
                        asset, len(buf) if buf is not None else 0)
            return None

        model       = self._models[asset]
        feature_cols = self._feature_cols[asset]

        try:
            # Run full feature pipeline on buffer (no labels — scoring path)
            df = generate_features(buf.copy())
            df = add_regime_columns(df)
            df = add_session_column(df)
            df.dropna(inplace=True)

            if len(df) == 0:
                log.warning("score_bar: all rows NaN after feature gen for %s", asset)
                return None

            # Score all rows, take the last one
            scored = apply_signals(model, feature_cols, df)

            # Apply trend + vol filters (same as main.py pipeline)
            filtered = apply_trend_filter(scored)
            filtered["signal"] = filtered["signal_trend_filtered"]
            filtered = apply_vol_filter(filtered)
            # Use vol-filtered signal (blocks low_vol if BLOCK_LOW_VOL=True)
            final_signal_col = "signal_vol_filtered"

            last = filtered.iloc[-1]
            signal   = int(last[final_signal_col])
            buy_prob = float(last.get("buy_prob",  0.0))
            sell_prob = float(last.get("sell_prob", 0.0))
            atr_pct  = float(last.get("atr_pct",   0.0))
            close    = float(last["close"])
            bar_ts   = filtered.index[-1]

            return {
                "signal":    signal,
                "buy_prob":  buy_prob,
                "sell_prob": sell_prob,
                "atr_pct":   atr_pct,
                "close":     close,
                "bar_ts":    bar_ts,
            }

        except Exception as exc:
            log.error("score_bar failed for %s: %s", asset, exc, exc_info=True)
            return None

    def should_retrain(self, asset: str) -> bool:
        """Return True when the retrain counter has reached the configured interval."""
        if config_live.RETRAIN_EVERY_N_BARS <= 0:
            return False
        return self._bars_since_retrain.get(asset, 0) >= config_live.RETRAIN_EVERY_N_BARS

    def retrain(self, asset: str) -> None:
        """Re-train from the current rolling buffer (not the full CSV)."""
        if asset not in self._buffers:
            return
        log.info("Retraining model for %s ...", asset)
        try:
            self._train_from_df(asset, self._buffers[asset].copy())
            self._bars_since_retrain[asset] = 0
            log.info("Retrain complete for %s", asset)
        except Exception as exc:
            log.error("Retrain failed for %s: %s", asset, exc, exc_info=True)

    # ── Private ────────────────────────────────────────────────────────────────

    def _train_asset(self, asset: str) -> None:
        """
        Load 1H CSV for asset and train the model on history bounded by
        [TRAINING_START_DATE, TRAINING_CUTOFF_DATE).

        Both ends are anchored so CSV refreshes (which shift the data's actual
        start forward, because download_all.py uses YEARS_BACK from today) do
        not silently change the training set.  Bars at or after the cutoff are
        reserved for live scoring and out-of-sample backtest evaluation.
        """
        csv_path = os.path.join(config_live.DATA_DIR, f"{asset}_1h_4y.csv")
        if not os.path.exists(csv_path):
            log.error("Historical CSV not found: %s — skipping %s", csv_path, asset)
            return

        log.info("Training model for %s from %s ...", asset, csv_path)
        try:
            df_raw = load_ohlcv(csv_path)
            if len(df_raw) == 0:
                log.error("  %s: CSV is empty — skipping", asset)
                return

            start   = pd.Timestamp(pipeline_config.TRAINING_START_DATE)
            cutoff  = pd.Timestamp(pipeline_config.TRAINING_CUTOFF_DATE)
            n_total = len(df_raw)
            df_train = df_raw[(df_raw.index >= start) & (df_raw.index < cutoff)]
            n_post   = (df_raw.index >= cutoff).sum()

            csv_start = df_raw.index[0]
            if csv_start > start:
                missing_bars = int(((df_raw.index[0] - start) / pd.Timedelta(hours=1)))
                log.warning(
                    "  %s: CSV starts at %s but TRAINING_START_DATE is %s — "
                    "~%d bars are missing from the front of the training window. "
                    "Re-download data with an earlier start, or bump "
                    "TRAINING_START_DATE.  Training will proceed on the available "
                    "range (parity with backtest still holds, but the model differs "
                    "from one trained on the original window).",
                    asset, csv_start.date(), start.date(), missing_bars,
                )

            if len(df_train) < 200:
                log.error(
                    "  %s: only %d bars in [%s, %s) — refusing to train "
                    "(need ≥200).  Re-download earlier history or adjust bounds.",
                    asset, len(df_train), start.date(), cutoff.date(),
                )
                return

            self._train_from_df(asset, df_train)

            # Seed the rolling buffer with the last WARMUP_1H_BARS bars of the
            # FULL CSV (including post-cutoff bars when present).  The model
            # was fit only on pre-cutoff data above, but the buffer must be
            # contiguous up to "now" so that indicators (SMA200, ATR14, etc.)
            # compute on real recent context.  Seeding from df_train would
            # leave a gap between the cutoff and the first live bar; in that
            # window SMA200 averages months-old prices against today's price,
            # producing extreme features and false strong signals for ~320
            # bars (13 days) after every restart.
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            self._buffers[asset] = (
                df_raw[ohlcv_cols].iloc[-config_live.WARMUP_1H_BARS:].copy()
            )
            self._bars_since_retrain[asset] = 0

            buf = self._buffers[asset]
            log.info(
                "  %s: window=[%s, %s)  train_bars=%d  reserved_post_cutoff=%d  "
                "buffer=[%s, %s]  n=%d",
                asset, start.date(), cutoff.date(),
                len(df_train), int(n_post),
                buf.index[0], buf.index[-1], len(buf),
            )
        except Exception as exc:
            log.error("Training failed for %s: %s", asset, exc, exc_info=True)

    def _train_from_df(self, asset: str, df_raw: pd.DataFrame) -> None:
        """
        Run the full training pipeline on df_raw and store model + feature_cols.

        Trains on ALL rows (no train/test split) so every available bar informs
        the live model.
        """
        df = generate_features(df_raw.copy())
        df = add_regime_columns(df)
        df = add_session_column(df)
        df = add_labels(df)
        df.dropna(inplace=True)

        if len(df) < 200:
            raise ValueError(
                f"Only {len(df)} labeled rows — not enough to train {asset}"
            )

        feature_cols = get_feature_columns(df)
        model = fit_model(df, feature_cols)

        self._models[asset]       = model
        self._feature_cols[asset] = feature_cols
