"""
main.py — Entry point for the ARKAD MRK trading research pipeline.

Usage:
    python main.py --data data/sample.csv
    python main.py --data data/sample.csv --symbol AAPL
"""

import argparse
import math
import os
import sys

import config
from data.loader import load_ohlcv
from features.generator import generate_features, add_labels
from features.market_regime import (
    add_regime_columns, add_session_column, print_regime_stats,
    apply_trend_filter, apply_vol_filter, apply_regime_threshold_filter,
    apply_session_filter,
)
from models.classifier import train_classifier, generate_signals, print_prob_diagnostics
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from reports.reporter import print_report, save_report
from experiments.threshold_sweep import run_threshold_sweep
from experiments.walk_forward import run_walk_forward
from experiments.regime_analysis import run_regime_analysis, run_session_analysis
from features.intraday import add_5m_features
from features.context_4h import add_4h_context, apply_4h_context_filter, print_4h_bias_stats
from features.execution_15m import (
    load_15m_data, load_5m_as_15m, annotate_signals_AB, coverage_stats, print_coverage_stats,
)


def _fmt_val(v, fmt: str) -> str:
    try:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "N/A"
        return format(v, fmt)
    except (TypeError, ValueError):
        return "N/A"


def _strategy_score(m: dict) -> float:
    """
    Frequency-weighted score: PF × log(1 + trades) × expectancy.

    Tier 1 (fully qualified): trades >= MIN_TRADES, PF > 0, exp > 0
      score = PF × log(1 + trades) × expectancy  (positive, comparable)

    Tier 2 (below frequency threshold but edge is positive): PF > 0 and exp > 0
      score = -(MIN_TRADES / trades) + PF × exp × 0.01
      Negative, but orderable — selects the best low-frequency profitable option.

    Tier 3 (losing): PF <= 0 or exp <= 0
      score = -inf  (never preferred over any positive-edge option)
    """
    n   = int(m.get("n_trades", 0) or 0)
    pf  = float(m.get("profit_factor", 0) or 0)
    exp = float(m.get("expectancy", -9_999) or -9_999)
    if pf <= 0 or exp <= 0:
        return float("-inf")
    if n < config.MIN_TRADES:
        # Tier 2: profitable but infrequent — rank by edge quality only (PF × exp).
        # A constant offset of -1000 keeps all Tier 2 scores below any Tier 1 score.
        return -1000.0 + pf * exp
    return pf * math.log1p(n) * exp


def parse_args():
    parser = argparse.ArgumentParser(description="ARKAD MRK — Trading Research Pipeline")
    parser.add_argument("--data",   required=True, help="Path to 1H OHLCV CSV file")
    parser.add_argument("--data5m",  default=None, help="Path to 5m OHLCV CSV (optional, enables intraday features)")
    parser.add_argument("--data15m", default=None, help="Path to 15m or 5m OHLCV CSV (optional, enables 15m-AB execution layer)")
    parser.add_argument("--symbol",  default="ASSET", help="Symbol label for reports")
    return parser.parse_args()


def _print_strategy_comparison(strategies: list[tuple[str, dict]]) -> None:
    """
    Print a side-by-side performance table for N strategies.

    Parameters
    ----------
    strategies : list of (name, metrics_dict) pairs in display order.
    """
    import math

    def _fmt(val, fmt_str, fallback="N/A"):
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return fallback
            return format(val, fmt_str)
        except (TypeError, ValueError):
            return fallback

    col_w  = 16
    lbl_w  = 22
    sep    = "-" * (lbl_w + col_w * len(strategies) + 2)

    print(f"\n  Strategy Comparison")
    print(f"  {sep}")
    header = f"  {'Metric':<{lbl_w}}" + "".join(f"{name:>{col_w}}" for name, _ in strategies)
    print(header)
    print(f"  {sep}")

    metric_rows = [
        ("Trades",        "n_trades",      "d",    ""),
        ("Win rate",      "win_rate",      ".1%",  ""),
        ("Profit factor", "profit_factor", ".2f",  ""),
        ("Expectancy $",  "expectancy",    ".2f",  ""),
        ("Max drawdown",  "max_drawdown",  ".1f",  "%"),
        ("Score",         "_score",        ".3f",  ""),
    ]

    for label, key, fmt_str, suffix in metric_rows:
        row = f"  {label:<{lbl_w}}"
        for _, m in strategies:
            val = m.get(key)
            row += f"{_fmt(val, fmt_str) + suffix:>{col_w}}"
        print(row)

    print(f"  {sep}\n")


def _print_filter_funnel(signals_df, raw_signal) -> None:
    """
    Show how many signals survive each filter stage and trades/day at each level.

    Helps identify which filter is the biggest frequency bottleneck.
    All filters (vol, regime_threshold, session) are measured relative to the
    trend-filtered baseline, since they are applied in parallel from that base.

    Bar-granularity detection: if the DatetimeIndex has no intraday time
    variation (all bars at midnight), the data is treated as daily bars and
    trades/day = signals / n_bars.  Otherwise treated as 1H bars and
    trades/day = signals / (n_bars / 24).
    """
    import pandas as _pd
    n_bars = len(signals_df)

    # Detect bar granularity from index time component
    idx = signals_df.index
    if isinstance(idx, _pd.DatetimeIndex) and idx.hour.max() == 0:
        # Daily bars: 1 bar ≈ 1 trading day
        bar_label = "daily"
        test_days = n_bars
    else:
        # Intraday bars: assume 1H (24 bars per calendar day)
        bar_label = "1H"
        test_days = n_bars / 24

    def _nonzero(col: str) -> int:
        return int((signals_df[col] != 0).sum()) if col in signals_df.columns else 0

    raw_n    = int((raw_signal != 0).sum())
    trend_n  = int((signals_df["signal"] != 0).sum())   # signal was overwritten with trend-filtered
    vol_n    = _nonzero("signal_vol_filtered")
    regime_n = _nonzero("signal_regime_filtered")
    sess_n   = _nonzero("signal_session_filtered")

    def _pct(n: int, base: int) -> str:
        return f"{n / base * 100:.1f}%" if base > 0 else "N/A"

    def _tpd(n: int) -> str:
        return f"{n / test_days:.1f}/d" if test_days > 0 else "N/A"

    sep = "-" * 62
    print(f"\n  Filter Frequency Funnel  ({n_bars:,} test bars, ~{test_days:.0f} {bar_label} days)")
    print(f"  {sep}")
    print(f"  {'Stage':<30} {'Signals':>8}  {'vs Raw':>7}  {'vs Trend':>9}  {'Rate':>7}")
    print(f"  {sep}")
    print(f"  {'Raw model output':<30} {raw_n:>8}  {'100.0%':>7}  {'—':>9}  {_tpd(raw_n):>7}")
    print(f"  {'After trend filter':<30} {trend_n:>8}  {_pct(trend_n, raw_n):>7}  {'100.0%':>9}  {_tpd(trend_n):>7}")
    print(f"  {'After vol filter':<30} {vol_n:>8}  {_pct(vol_n, raw_n):>7}  {_pct(vol_n, trend_n):>9}  {_tpd(vol_n):>7}")
    print(f"  {'After regime threshold':<30} {regime_n:>8}  {_pct(regime_n, raw_n):>7}  {_pct(regime_n, trend_n):>9}  {_tpd(regime_n):>7}")
    print(f"  {'After session filter':<30} {sess_n:>8}  {_pct(sess_n, raw_n):>7}  {_pct(sess_n, trend_n):>9}  {_tpd(sess_n):>7}")
    print(f"  {sep}\n")


def _save_strategy_comparison(
    strategies: list[tuple[str, dict]],
    out_path: str,
) -> None:
    """Save strategy comparison metrics to a CSV file."""
    import os
    import math

    metric_keys = ["n_trades", "win_rate", "profit_factor", "expectancy", "max_drawdown"]
    rows = []
    for name, m in strategies:
        row = {"strategy": name}
        for key in metric_keys:
            val = m.get(key)
            if isinstance(val, float) and math.isnan(val):
                val = None
            row[key] = val
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved: {out_path}\n")


def run_pipeline(data_path: str, symbol: str, data5m_path: str = None, data15m_path: str = None):
    print(f"\n{'='*60}")
    print(f"  ARKAD MRK Research Pipeline  |  {symbol}")
    print(f"{'='*60}\n")

    # 1. Load data
    print("[1/9] Loading market data...")
    df = load_ohlcv(data_path)
    print(f"      Loaded {len(df):,} bars from {df.index[0].date()} to {df.index[-1].date()}\n")

    # 2. Generate features
    print("[2/9] Generating candidate features...")
    df = generate_features(df)
    if data5m_path:
        print(f"      Loading 5m intraday features from {data5m_path}...")
        df = add_5m_features(df, data5m_path)
    print(f"      Generated {len(df.columns)} columns total\n")

    # 3. Add market regime and session columns
    print("[3/9] Classifying market regimes and sessions...")
    df = add_regime_columns(df)
    df = add_session_column(df)
    trend_counts   = df["trend"].value_counts()
    session_counts = df["session"].value_counts()
    print(f"      trend_up={trend_counts.get('trend_up', 0)}  "
          f"range={trend_counts.get('range', 0)}  "
          f"trend_down={trend_counts.get('trend_down', 0)}")
    print(f"      tradeable bars (regime_gate=1): {df['regime_gate'].sum():,}")
    print(f"      asia={session_counts.get('asia', 0)}  "
          f"eu={session_counts.get('eu', 0)}  "
          f"us={session_counts.get('us', 0)}  "
          f"late={session_counts.get('late', 0)}\n")

    # 3b. 4H context layer (optional)
    if config.USE_4H_CONTEXT:
        print("[3b/9] Adding 4H directional context"
              f" (mode={config.CONTEXT_4H_MODE})...")
        df = add_4h_context(df)
        print_4h_bias_stats(df)

    # 4. Add labels
    print("[4/9] Creating forward-return labels...")
    df = add_labels(df)
    df.dropna(inplace=True)
    n_buy   = (df["label"] ==  1).sum()
    n_sell  = (df["label"] == -1).sum()
    n_neut  = (df["label"] ==  0).sum()
    atr_med = df["fwd_return_atr"].abs().median()
    print(f"      {n_buy} buy  |  {n_sell} sell  |  {n_neut} neutral (excluded from training — 2-class model)")
    print(f"      median |fwd_return_atr| = {atr_med:.2f} ATR units\n")

    # 5. Train model
    print("[5/9] Training baseline classifier...")
    model, feature_cols, X_test, y_test = train_classifier(df)
    print()

    # 6. Generate signals, apply filters, run regime analysis
    print("[6/9] Generating trade signals and running regime analysis...")
    signals_df = generate_signals(model, feature_cols, df)
    # Save raw signal counts before trend filter overwrites the 'signal' column.
    _raw_signal = signals_df["signal"].copy()
    # Optional 4H context gate: applied before all other filters.
    # Blocks long signals that contradict the 4H bear bias (and vice versa).
    if config.USE_4H_CONTEXT:
        signals_df = apply_4h_context_filter(signals_df)
        signals_df["signal"] = signals_df["signal_4h_filtered"]
    # Trend gate applied first as a hard filter — only ALLOWED_TRENDS pass through.
    # All downstream filters (vol, regime, session) operate on trend-gated signals.
    signals_df = apply_trend_filter(signals_df)
    signals_df["signal"] = signals_df["signal_trend_filtered"]
    signals_df = apply_vol_filter(signals_df)
    signals_df = apply_regime_threshold_filter(signals_df)
    signals_df = apply_session_filter(signals_df)
    print_prob_diagnostics(signals_df)
    print_regime_stats(df)
    _print_filter_funnel(signals_df, _raw_signal)
    run_regime_analysis(signals_df)
    run_session_analysis(signals_df)

    # 7. Walk-forward validation — re-trains on each expanding window
    print("[7/9] Running walk-forward validation...")
    run_walk_forward(df)

    # 8. Threshold sweep
    print("[8/9] Running threshold sweep...")
    run_threshold_sweep(signals_df)

    # 9. Final backtest: 4-strategy comparison
    print("[9/9] Running final backtest (4-strategy comparison)...")

    raw_df    = signals_df.copy()
    volf_df   = signals_df.copy()
    regf_df   = signals_df.copy()
    sesf_df   = signals_df.copy()
    volf_df["signal"] = volf_df["signal_vol_filtered"]
    regf_df["signal"] = regf_df["signal_regime_filtered"]
    sesf_df["signal"] = sesf_df["signal_session_filtered"]

    trades_raw,  eq_raw  = run_backtest(raw_df)
    trades_volf, eq_volf = run_backtest(volf_df)
    trades_regf, eq_regf = run_backtest(regf_df)
    trades_sesf, eq_sesf = run_backtest(sesf_df)

    m_raw  = compute_metrics(trades_raw,  eq_raw)
    m_volf = compute_metrics(trades_volf, eq_volf)
    m_regf = compute_metrics(trades_regf, eq_regf)
    m_sesf = compute_metrics(trades_sesf, eq_sesf)

    # Inject frequency-weighted score into each metrics dict for display
    backtest_map = {
        "Raw":              (m_raw,  trades_raw,  eq_raw),
        "Vol-Filtered":     (m_volf, trades_volf, eq_volf),
        "Regime-Aware":     (m_regf, trades_regf, eq_regf),
        "Session-Filtered": (m_sesf, trades_sesf, eq_sesf),
    }
    # Keep a reference to each strategy's signals DataFrame so we can apply
    # the 15m-AB execution layer later without re-running the model.
    _signals_lookup = {
        "Raw":              raw_df,
        "Vol-Filtered":     volf_df,
        "Regime-Aware":     regf_df,
        "Session-Filtered": sesf_df,
    }
    for name, (m, _, _) in backtest_map.items():
        m["_score"] = _strategy_score(m)

    strategies = [(name, m) for name, (m, _, _) in backtest_map.items()]
    _print_strategy_comparison(strategies)
    _save_strategy_comparison(
        strategies,
        out_path=f"{config.EXPERIMENTS_DIR}/session_filter_comparison.csv",
    )

    # Select the strategy with the highest frequency-weighted score.
    # Score = PF × log(1 + trades) × expectancy — rewards both profitability
    # and trade frequency; -inf when trades < MIN_TRADES or expectancy <= 0.
    best_name = max(
        backtest_map,
        key=lambda n: (
            backtest_map[n][0]["_score"]
            if backtest_map[n][0]["_score"] != float("-inf")
            else -9_999
        ),
    )
    best_score = backtest_map[best_name][0]["_score"]
    _, final_trades, final_eq = backtest_map[best_name]

    if best_score == float("-inf"):
        print(f"      No strategy meets any criteria (all have PF<=0 or exp<=0).")
        print(f"      Reporting '{best_name}' as the least-bad option.")
    elif best_score < 0:
        print(f"      Best strategy (profitable but below {config.MIN_TRADES}-trade threshold): "
              f"{best_name}  (score={best_score:.3f})")
    else:
        print(f"      Best strategy by score: {best_name}"
              f"  (score={best_score:.3f})")

    print()
    print_report(symbol, final_trades, final_eq)
    save_report(symbol, final_trades, final_eq)

    # ── Optional 15m-AB execution layer ──────────────────────────────────────
    # Enabled by passing --data15m (15m CSV) or --data15m (5m CSV, auto-resampled).
    # Applied to the best 4-strategy result.  Reported separately — does not
    # override the main report; use the printed comparison to decide.
    if data15m_path:
        print(f"\n{'-'*60}")
        print(f"  15m-AB Execution Layer  (applied to: {best_name})")
        print(f"{'-'*60}")
        is_5m = "5m" in os.path.basename(data15m_path)
        df_15m = load_5m_as_15m(data15m_path) if is_5m else load_15m_data(data15m_path)
        print(f"  15m data: {len(df_15m):,} bars  "
              f"({df_15m.index[0].date()} to {df_15m.index[-1].date()})")

        best_signals = _signals_lookup[best_name]
        ann = annotate_signals_AB(best_signals, df_15m)
        ann["signal"] = ann["signal_15m_A"]

        trades_ab, eq_ab = run_backtest(ann)
        m_ab = compute_metrics(trades_ab, eq_ab)

        stats = coverage_stats(best_signals, ann, df_15m, window_label="15m-AB test")
        print_coverage_stats(stats)

        # Side-by-side comparison: baseline best vs 15m-AB
        n_base = int(m.get("n_trades", 0) if (m := backtest_map[best_name][0]) else 0)
        n_ab   = int(m_ab.get("n_trades", 0))
        print(f"\n  {'Metric':<22} {'Baseline (' + best_name + ')':>20} {'15m-AB':>12}")
        print(f"  {'-'*56}")
        for lbl, key, fmt in [
            ("Trades",        "n_trades",      "d"),
            ("Win rate",      "win_rate",      ".1%"),
            ("Profit factor", "profit_factor", ".3f"),
            ("Expectancy $",  "expectancy",    "+.2f"),
            ("Max drawdown",  "max_drawdown",  ".2f"),
        ]:
            v_base = backtest_map[best_name][0].get(key)
            v_ab   = m_ab.get(key)
            print(f"  {lbl:<22} {str(_fmt_val(v_base, fmt)):>20} {str(_fmt_val(v_ab, fmt)):>12}")
        print(f"  {'-'*56}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.data, args.symbol, args.data5m, args.data15m)
