"""
experiments/param_sweep.py — Grid search for optimal trading configuration.

Strategy
--------
1. Load data, generate features, regimes, sessions — once.
2. For each (tp_mult, sl_mult) pair:
     - Re-label with barrier-based labels (labels depend on TP/SL).
     - Train one CalibratedRandomForest model.
     - Compute buy_prob / sell_prob on the test set once.
3. For each (threshold, regime, session, vol_filter):
     - Apply filters to the pre-computed probability arrays (no re-training).
     - Run backtest, collect metrics.
4. Score all configs: score = PF × log1p(trades) × expectancy
   (requires exp > 0 to score; otherwise score = -inf).
5. For each unique (tp, sl) in top-20: run walk-forward once (silent),
   attach aggregate WF PF to all matching rows.
6. Print top-10 table; save full results CSV.

Run
---
    python experiments/param_sweep.py --data data/BTCUSDT_5m_4y.csv
"""

import argparse
import contextlib
import io
import math
import os
import sys
import time
import warnings
from itertools import product

warnings.filterwarnings("ignore")
# Force UTF-8 output so Unicode chars (arrows, dashes) work on all Windows locales
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from data.loader import load_ohlcv
from experiments.walk_forward import run_walk_forward
from features.generator import add_labels, generate_features
from features.market_regime import add_regime_columns, add_session_column
from models.classifier import fit_model, get_feature_columns


# ── Parameter grids ───────────────────────────────────────────────────────────

TP_MULTS   = [1.0, 1.25, 1.5, 2.0]
SL_MULTS   = [0.8, 1.0, 1.25, 1.5]
THRESHOLDS = [0.35, 0.40, 0.45, 0.50]

# Regime filter: name → list of allowed trend values
REGIMES = {
    "all":            ["range", "trend_up", "trend_down"],
    "directional":    ["trend_up", "trend_down"],
    "trend_up":       ["trend_up"],
    "range+up":       ["range", "trend_up"],
}

# Session filter: name → list of allowed session labels (None = no filter)
SESSIONS = {
    "all":          None,
    "active":       ["london_open", "eu_mid", "us_open", "us_afternoon"],
    "us":           ["us_open", "us_afternoon"],
    "asia_late":    ["asia", "late"],
}

# Volatility filter: name → action
VOL_FILTERS = {
    "all":       "all",
    "block_low": "block_low",   # exclude low_vol bars
    "high_only": "high_only",   # only high_vol bars
}

# Minimum trades in the test window for a config to be eligible for scoring.
# Test window ≈ 20% of 35k bars ≈ 290 days.  300 trades ≈ 1 trade/day minimum.
MIN_TRADES = 300


# ── Filter helpers (all operate on NumPy arrays — no config side-effects) ────

def _signals_from_probs(buy_prob: np.ndarray,
                        sell_prob: np.ndarray,
                        threshold: float) -> np.ndarray:
    """Apply symmetric threshold with conflict-resolution to get signal array."""
    buy  = buy_prob  >= threshold
    sell = sell_prob >= threshold
    sig  = np.where(buy & sell, 0,
           np.where(buy,         1,
           np.where(sell,       -1, 0)))
    return sig.astype(int)


def _apply_regime_filter(df: pd.DataFrame,
                         signal: np.ndarray,
                         regime_key: str) -> np.ndarray:
    allowed = REGIMES[regime_key]
    mask    = df["trend"].isin(allowed).values
    return signal * mask.astype(int)


def _apply_vol_filter(df: pd.DataFrame,
                      signal: np.ndarray,
                      vol_key: str) -> np.ndarray:
    if vol_key == "all":
        return signal
    vol = df["vol_regime"].values
    if vol_key == "block_low":
        return np.where(vol == "low_vol", 0, signal)
    if vol_key == "high_only":
        return np.where(vol == "high_vol", signal, 0)
    return signal


def _apply_session_filter(df: pd.DataFrame,
                          signal: np.ndarray,
                          sess_key: str) -> np.ndarray:
    allowed = SESSIONS[sess_key]
    if allowed is None:
        return signal
    mask = np.isin(df["session"].values, allowed)
    return np.where(mask, signal, 0)


# ── Silent walk-forward (returns aggregate avg PF) ────────────────────────────

def _run_wf_silent(df: pd.DataFrame) -> float:
    """
    Run walk-forward on a fully-prepared DataFrame (features + labels).
    Suppresses all print output.
    Returns the average profit factor across windows that had trades,
    or NaN if no windows produced trades.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = run_walk_forward(df)
    if results is None or results.empty:
        return float("nan")
    traded = results[results["n_trades"] > 0]
    if traded.empty:
        return float("nan")
    return float(traded["profit_factor"].mean())


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(data_path: str) -> pd.DataFrame:
    t0 = time.time()

    print(f"\n{'='*72}")
    print(f"  ARKAD MRK — Parameter Sweep")
    print(f"  TP: {TP_MULTS}  x  SL: {SL_MULTS}")
    n_model_runs = len(TP_MULTS) * len(SL_MULTS)
    n_filter_combos = (
        len(THRESHOLDS) * len(REGIMES) * len(SESSIONS) * len(VOL_FILTERS)
    )
    print(f"  {n_model_runs} model configs  ×  {n_filter_combos} filter combos"
          f"  =  {n_model_runs * n_filter_combos:,} total evaluations")
    print(f"{'='*72}\n")

    # ── Step 1: Load + feature-engineer once ─────────────────────────────────
    print("[1/3] Loading data and generating features...")
    df_base = load_ohlcv(data_path)
    df_base = generate_features(df_base)
    df_base = add_regime_columns(df_base)
    df_base = add_session_column(df_base)
    print(f"      {len(df_base):,} bars  |  {len(df_base.columns)} columns\n")

    # Feature columns are stable regardless of labels
    feature_cols = get_feature_columns(df_base)

    # ── Step 2: Sweep TP × SL ────────────────────────────────────────────────
    print("[2/3] Training models and sweeping filter combinations...\n")
    sep = "-" * 72
    print(f"  {sep}")
    print(f"  {'TP':>5}  {'SL':>5}  {'Labels (B/S/N)':>22}  "
          f"{'Filter combos':>14}  {'Time':>6}")
    print(f"  {sep}")

    all_rows   = []
    wf_cache   = {}    # (tp, sl) → float WF PF

    for tp_mult, sl_mult in product(TP_MULTS, SL_MULTS):
        t_model = time.time()

        # Update config so labels + backtest engine use the same TP/SL
        config.TAKE_PROFIT_ATR_MULT = tp_mult
        config.STOP_LOSS_ATR_MULT   = sl_mult

        # Re-label with new barriers
        df_lbl = add_labels(df_base.copy())
        df_lbl.dropna(inplace=True)

        n_buy  = int((df_lbl["label"] ==  1).sum())
        n_sell = int((df_lbl["label"] == -1).sum())
        n_neut = int((df_lbl["label"] ==  0).sum())

        # Chronological train/test split
        split    = int(len(df_lbl) * (1 - config.TEST_SIZE))
        train_df = df_lbl.iloc[:split]
        test_df  = df_lbl.iloc[split:]

        # Train calibrated RF
        model = fit_model(train_df, feature_cols)

        # Probabilities on test set — computed once
        proba    = model.predict_proba(test_df[feature_cols])
        classes  = list(model.classes_)
        buy_prob  = (proba[:, classes.index( 1)]
                     if  1 in classes else np.zeros(len(proba)))
        sell_prob = (proba[:, classes.index(-1)]
                     if -1 in classes else np.zeros(len(proba)))

        # Test-set date span for trades-per-day
        n_test_days = max((test_df.index[-1] - test_df.index[0]).days, 1)

        # ── Inner filter sweep (no re-training) ──────────────────────────────
        n_scored = 0
        for thr, reg_key, sess_key, vol_key in product(
            THRESHOLDS, REGIMES.keys(), SESSIONS.keys(), VOL_FILTERS.keys()
        ):
            sig = _signals_from_probs(buy_prob, sell_prob, thr)
            sig = _apply_regime_filter(test_df, sig, reg_key)
            sig = _apply_vol_filter(test_df, sig, vol_key)
            sig = _apply_session_filter(test_df, sig, sess_key)

            bt_df           = test_df.copy()
            bt_df["signal"] = sig

            trades, equity = run_backtest(bt_df)
            m = compute_metrics(trades, equity)

            n_trades = int(m.get("n_trades", 0) or 0)
            pf       = float(m.get("profit_factor", 0) or 0)
            exp_val  = float(m.get("expectancy", -9999) or -9999)
            wr       = float(m.get("win_rate", 0) or 0)
            max_dd   = float(m.get("max_drawdown", 0) or 0)

            trades_per_day   = n_trades / n_test_days
            est_daily_return = trades_per_day * exp_val

            # Score: positive only if exp > 0 and meets minimum trade count
            if n_trades >= MIN_TRADES and pf > 0 and exp_val > 0:
                score = pf * math.log1p(n_trades) * exp_val
                n_scored += 1
            else:
                score = float("-inf")

            all_rows.append({
                "tp_mult":        tp_mult,
                "sl_mult":        sl_mult,
                "threshold":      thr,
                "regime":         reg_key,
                "session":        sess_key,
                "vol_filter":     vol_key,
                "n_trades":       n_trades,
                "trades_per_day": round(trades_per_day,   4),
                "win_rate_pct":   round(wr * 100,         2),
                "profit_factor":  round(pf,               4),
                "expectancy":     round(exp_val,          4),
                "est_daily_ret":  round(est_daily_return, 6),
                "max_drawdown":   round(max_dd,           4),
                "score":          round(score,            4) if score != float("-inf") else float("-inf"),
                "wf_pf":          None,   # filled after ranking
            })

        elapsed_model = time.time() - t_model
        print(f"  TP={tp_mult:.2f}  SL={sl_mult:.2f}  "
              f"[{n_buy:>7,} / {n_sell:>7,} / {n_neut:>7,}]  "
              f"{n_scored:>4} scored  "
              f"{elapsed_model:.0f}s")

    # ── Step 3: Rank, walk-forward for top configs ────────────────────────────
    print(f"\n[3/3] Ranking results and running walk-forward for top configs...\n")

    results_df = pd.DataFrame(all_rows)

    # Sort: scored rows first (score > -inf), then by score desc
    finite   = results_df[results_df["score"] != float("-inf")].copy()
    infinite = results_df[results_df["score"] == float("-inf")].copy()

    finite.sort_values("score", ascending=False, inplace=True)
    results_df = pd.concat([finite, infinite], ignore_index=True)

    # Identify unique (tp, sl) pairs in top-20 for WF
    top20    = results_df.head(20)
    wf_pairs = top20[["tp_mult", "sl_mult"]].drop_duplicates().values.tolist()

    print(f"  Running walk-forward for {len(wf_pairs)} unique (TP, SL) pair(s):\n")
    for tp_mult, sl_mult in wf_pairs:
        key = (tp_mult, sl_mult)
        if key in wf_cache:
            continue
        print(f"    TP={tp_mult:.2f}  SL={sl_mult:.2f} ...", end=" ", flush=True)

        config.TAKE_PROFIT_ATR_MULT = tp_mult
        config.STOP_LOSS_ATR_MULT   = sl_mult
        config.BUY_PROB_THRESHOLD   = 0.40
        config.SELL_PROB_THRESHOLD  = 0.40

        df_wf = add_labels(df_base.copy())
        df_wf.dropna(inplace=True)

        wf_pf = _run_wf_silent(df_wf)
        val   = round(wf_pf, 3) if not math.isnan(wf_pf) else None
        wf_cache[key] = val
        print(f"WF avg PF = {val}")

    # Attach WF PF to all rows
    results_df["wf_pf"] = results_df.apply(
        lambda r: wf_cache.get((r["tp_mult"], r["sl_mult"])), axis=1
    )

    # ── Print top-10 ──────────────────────────────────────────────────────────
    top10   = results_df.head(10)
    elapsed = time.time() - t0

    print(f"\n  Completed in {elapsed / 60:.1f} minutes.\n")
    print(f"{'='*72}")
    print(f"  Top-10 Configurations  (score = PF × log1p(trades) × expectancy)")
    print(f"{'='*72}\n")

    # Header
    cols = (
        f"  {'#':>2}  {'TP':>4}  {'SL':>4}  {'Thr':>4}  "
        f"{'Regime':<10}  {'Session':<12}  {'Vol':<10}  "
        f"{'Trades':>6}  {'T/Day':>5}  {'WR%':>5}  "
        f"{'PF':>5}  {'Exp$':>6}  {'DlyRet%':>8}  {'WF_PF':>5}  {'Score':>8}"
    )
    print(cols)
    print(f"  {'-' * (len(cols) - 2)}")

    for rank, (_, row) in enumerate(top10.iterrows(), start=1):
        wf_str  = f"{row['wf_pf']:.3f}" if row["wf_pf"] is not None else "  N/A"
        exp_str = f"{row['expectancy']:+.2f}"
        dly_str = f"{row['est_daily_ret'] * 100:+.4f}"
        sc_str  = f"{row['score']:.4f}" if row["score"] != float("-inf") else "   N/A"
        print(
            f"  {rank:>2}  {row['tp_mult']:>4.2f}  {row['sl_mult']:>4.2f}  "
            f"{row['threshold']:>4.2f}  "
            f"{row['regime']:<10}  {row['session']:<12}  {row['vol_filter']:<10}  "
            f"{int(row['n_trades']):>6}  {row['trades_per_day']:>5.3f}  "
            f"{row['win_rate_pct']:>5.1f}  "
            f"{row['profit_factor']:>5.3f}  {exp_str:>6}  {dly_str:>8}  "
            f"{wf_str:>5}  {sc_str:>8}"
        )

    print(f"\n  {'='*72}")

    # Save
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    out_path = os.path.join(config.EXPERIMENTS_DIR, "param_sweep_results.csv")
    # Replace -inf with NaN for CSV
    results_df["score"] = results_df["score"].replace(float("-inf"), float("nan"))
    results_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n  Total configs evaluated : {len(results_df):,}")
    print(f"  Configs with score > 0  : {len(finite):,}")
    print(f"  Full results saved      : {out_path}\n")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARKAD MRK Parameter Sweep")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV")
    args = parser.parse_args()
    run_sweep(args.data)
