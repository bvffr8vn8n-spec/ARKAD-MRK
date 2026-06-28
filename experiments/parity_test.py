"""
experiments/parity_test.py — Compare live paper-trader signals vs offline replay.

Premise
-------
With `config.TRAINING_CUTOFF_DATE` shared by paper trader and this script, both
sides train on identical pre-cutoff data.  If they then score the same 1H bar
they MUST produce the same (signal, buy_prob, atr_pct).  Any divergence is a
bug — either in data ingestion, feature generation, or filter ordering.

How it works
------------
1. Instantiate a fresh `SignalEngine` and call `train_all()` — uses the exact
   same code path as paper trader, so models are bit-identical (modulo RF
   `random_state`, which is fixed in config).
2. For each Tier 1 asset, iterate the historical CSV bars where
   `index >= TRAINING_CUTOFF_DATE` in chronological order, feed each one
   through `push_bar` + `score_bar`, and capture the result.
3. Save the offline replay to `experiments/parity_replay_signals.csv`.
4. Load `paper_trading/signal_log.csv` (written by the live paper trader, one
   row per scored 1H bar) and inner-join on (asset, bar_ts).
5. Report:
     - bars present on both sides         (the comparison universe)
     - signal-direction agreement count   (exact match on the int signal)
     - probability max-abs-diff           (should be ~1e-9 if pipelines match)
     - bars present only in live          (paper trader saw bars not in CSV;
                                            normal if CSV is older than live)
     - bars present only in replay        (live paper trader missed bars;
                                            this is a real parity concern)

Usage
-----
    python -m experiments.parity_test
or
    python experiments/parity_test.py

Exit code is 0 if parity holds, 1 if any mismatch is detected.
"""

import os
import sys
from typing import Optional

# Force UTF-8 console output on Windows (avoids cp1251 codec errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as pipeline_config
from data.loader import load_ohlcv
from paper_trading import config_live
from paper_trading.signal_engine import SignalEngine

REPLAY_OUTPUT = os.path.join(_ROOT, "experiments", "parity_replay_signals.csv")
LIVE_LOG_PATH = config_live.SIGNAL_LOG_FILE

PROB_TOLERANCE  = 1e-6   # float roundtrip headroom across pandas / numpy / pickle
ATR_TOLERANCE   = 1e-8
OHLC_TOLERANCE  = 1e-6   # raw OHLC: tighter than prob (no model layer in between)


def run_offline_replay() -> pd.DataFrame:
    """Train and replay all post-cutoff bars for every Tier 1 asset."""
    print(f"  Training cutoff:  {pipeline_config.TRAINING_CUTOFF_DATE}")
    print(f"  Assets:           {', '.join(config_live.ASSETS)}")
    print(f"  Training models (full pre-cutoff history) ...")

    engine = SignalEngine()
    engine.train_all()

    cutoff = pd.Timestamp(pipeline_config.TRAINING_CUTOFF_DATE)
    rows: list[dict] = []

    for asset in config_live.ASSETS:
        if asset not in engine._models:
            print(f"  {asset}: model missing (training failed); skipping replay.")
            continue

        csv_path = os.path.join(config_live.DATA_DIR, f"{asset}_1h_4y.csv")
        df_raw = load_ohlcv(csv_path)
        post = df_raw[df_raw.index >= cutoff].sort_index()
        ohlcv_cols = ["open", "high", "low", "close", "volume"]

        # Reset the buffer to PRE-cutoff bars only.  signal_engine._train_asset
        # seeds the buffer from the full CSV (including post-cutoff bars) for
        # correct live operation — but for replay that causes push_bar to dedup-
        # skip every post-cutoff bar, freezing the buffer at csv_last and
        # producing the same score for every iteration.  Re-seeding here lets
        # replay roll the buffer forward bar-by-bar as if from a cold start.
        df_pre = df_raw[df_raw.index < cutoff]
        engine._buffers[asset] = (
            df_pre[ohlcv_cols].iloc[-config_live.WARMUP_1H_BARS:].copy()
        )

        print(f"  {asset}: replaying {len(post)} post-cutoff bars ...", end="", flush=True)
        for ts, row in post.iterrows():
            bar = row[ohlcv_cols].copy()
            bar.name = ts
            engine.push_bar(asset, bar)
            result = engine.score_bar(asset)
            if result is None:
                continue
            rows.append({
                "bar_ts":   ts.isoformat(),
                "asset":    asset,
                "signal":   int(result["signal"]),
                "buy_prob": float(result["buy_prob"]),
                "sell_prob": float(result["sell_prob"]),
                "atr_pct":  float(result["atr_pct"]),
                "open":     float(row["open"]),
                "high":     float(row["high"]),
                "low":      float(row["low"]),
                "close":    float(result["close"]),
                "volume":   float(row.get("volume", 0.0)),
            })
        print(f" {sum(1 for r in rows if r['asset'] == asset)} scored")

    return pd.DataFrame(rows)


def load_live_signal_log() -> Optional[pd.DataFrame]:
    if not os.path.exists(LIVE_LOG_PATH):
        return None
    df = pd.read_csv(LIVE_LOG_PATH)
    if len(df) == 0:
        return df
    df["bar_ts"]    = df["bar_ts"].astype(str)
    df["signal"]    = df["signal"].astype(int)
    df["buy_prob"]  = df["buy_prob"].astype(float)
    df["sell_prob"] = df["sell_prob"].astype(float)
    df["atr_pct"]   = df["atr_pct"].astype(float)
    df["close"]     = df["close"].astype(float)
    # OHLCV columns are present only after the 2026-06-01 schema bump.
    # If absent (older log file), leave as NaN — comparison code tolerates that.
    for col in ("open", "high", "low", "volume"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def compare(replay: pd.DataFrame, live: pd.DataFrame) -> int:
    """
    Print parity diff. Returns 0 on full parity, 1 on any mismatch.

    Diagnostics distinguish three classes of "only in replay" bars:
      - bars before live ever started   → expected, paper trader wasn't running
      - bars after CSV ends             → impossible by construction (filtered out)
      - bars inside the live window     → ACTUAL parity concern (live missed them)

    Only the third class fails parity.
    """
    sep = "-" * 72
    print(f"\n  Parity diff")
    print(f"  {sep}")

    if live is None:
        print(f"  Live signal log not found: {LIVE_LOG_PATH}")
        print(f"  Paper trader hasn't run yet (or hasn't scored a bar).")
        print(f"  Offline replay saved to {REPLAY_OUTPUT} — re-run after live"
              f" has logged at least one bar to actually compare.")
        return 0

    if len(live) == 0:
        print(f"  Live signal log is empty (header only). No comparison possible yet.")
        return 0

    key = ["asset", "bar_ts"]
    merged = replay.merge(
        live, on=key, how="outer", suffixes=("_replay", "_live"),
        indicator=True,
    )

    both        = merged[merged["_merge"] == "both"]
    only_live   = merged[merged["_merge"] == "right_only"]
    only_replay = merged[merged["_merge"] == "left_only"]

    live_start = live["bar_ts"].min()
    csv_end    = replay["bar_ts"].max() if len(replay) > 0 else None

    only_replay_before_live = only_replay[only_replay["bar_ts"] < live_start]
    only_replay_after_live  = only_replay[only_replay["bar_ts"] >= live_start]

    signal_mismatch = both[both["signal_replay"] != both["signal_live"]]
    prob_diff       = (both["buy_prob_replay"] - both["buy_prob_live"]).abs()
    prob_mismatch   = both[prob_diff > PROB_TOLERANCE]
    atr_diff        = (both["atr_pct_replay"] - both["atr_pct_live"]).abs()
    atr_mismatch    = both[atr_diff > ATR_TOLERANCE]

    # Raw OHLC diffs — present only if both sides logged the new schema.
    ohlc_stats: dict[str, tuple[int, float]] = {}   # col → (count_over_tol, max_diff)
    has_ohlc_live = "open_live" in both.columns
    if has_ohlc_live:
        for col in ("open", "high", "low", "close", "volume"):
            r_col = f"{col}_replay"
            l_col = f"{col}_live"
            if r_col in both.columns and l_col in both.columns:
                d = (both[r_col] - both[l_col]).abs()
                tol = OHLC_TOLERANCE * (both[r_col].abs().clip(lower=1.0))
                n_over = int((d > tol).sum())
                max_d  = float(d.max()) if len(d) else 0.0
                ohlc_stats[col] = (n_over, max_d)

    print(f"  Live log starts:                       {live_start}")
    print(f"  CSV replay ends:                       {csv_end}")
    print(f"  Compared bars (both sides):         {len(both):>6}")
    if len(both) > 0:
        print(f"    signal mismatches:                {len(signal_mismatch):>6}")
        print(f"    buy_prob diff > {PROB_TOLERANCE:g}:        {len(prob_mismatch):>6}  "
              f"(max diff = {prob_diff.max():.3e})")
        print(f"    atr_pct diff > {ATR_TOLERANCE:g}:         {len(atr_mismatch):>6}  "
              f"(max diff = {atr_diff.max():.3e})")
        if ohlc_stats:
            print(f"    raw OHLC (rel-tol {OHLC_TOLERANCE:g}):")
            for col, (n_over, max_d) in ohlc_stats.items():
                print(f"      {col:<7}  bars over tol: {n_over:>4}  "
                      f"max abs diff = {max_d:.3e}")
        elif has_ohlc_live is False:
            print(f"    raw OHLC: live log predates the OHLC schema; "
                  f"restart paper trader to start logging open/high/low/volume.")
    print(f"  Only in live  (after CSV's last bar): {len(only_live):>6}  "
          f"— expected if CSV is stale; refresh data to extend overlap")
    print(f"  Only in replay (before live started): "
          f"{len(only_replay_before_live):>6}  — expected, paper trader started "
          f"on {live_start[:10]}")
    print(f"  Only in replay (LIVE actually MISSED): "
          f"{len(only_replay_after_live):>6}  — real parity concern")
    print(f"  {sep}")

    parity_ok = (
        len(signal_mismatch) == 0
        and len(prob_mismatch) == 0
        and len(atr_mismatch) == 0
        and len(only_replay_after_live) == 0
    )

    if len(both) == 0:
        print(f"  ⚠ No overlap between live log and replay.")
        print(f"    Live started after CSV last bar, OR CSV ends before live "
              f"start.  Refresh CSV (download_all.py --force) so the comparison "
              f"window actually exists, then re-run.")
    elif parity_ok:
        print(f"  ✓ Model parity holds on all {len(both)} overlapping bars.")
    else:
        print(f"  ✗ PARITY BROKEN — see mismatches above.")
        if len(signal_mismatch) > 0:
            print(f"\n  First 5 signal mismatches:")
            cols = ["asset", "bar_ts", "signal_replay", "signal_live",
                    "buy_prob_replay", "buy_prob_live"]
            print(signal_mismatch[cols].head(5).to_string(index=False))

    if len(only_replay_after_live) > 0:
        print(f"\n  Live missed {len(only_replay_after_live)} bars inside its "
              f"running window.  Check polling reliability / Bybit fetch failures:")
        print(only_replay_after_live[["asset", "bar_ts"]].head(10).to_string(index=False))

    return 0 if parity_ok else 1


def main() -> int:
    print("=" * 72)
    print("  ARKAD MRK — Live↔Backtest Parity Test")
    print("=" * 72)

    replay = run_offline_replay()
    replay.to_csv(REPLAY_OUTPUT, index=False, encoding="utf-8")
    print(f"\n  Replay saved: {REPLAY_OUTPUT}  ({len(replay)} rows)")

    live = load_live_signal_log()
    rc = compare(replay, live)

    print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
