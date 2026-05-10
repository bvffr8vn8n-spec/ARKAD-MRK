"""
paper_trader.py — Entry point for the ARKAD MRK automated paper trading system.

Usage
-----
    # Start (or resume after restart) the paper trader:
    python paper_trader.py

    # Run in background (Windows):
    pythonw paper_trader.py

    # Check current state:
    python paper_trader.py --status

    # Reset state (clears all open trades / monitors — does NOT delete CSV log):
    python paper_trader.py --reset

What it does
------------
1. Loads or initialises persistent state (paper_trading/state.json)
2. Trains one CalibratedRF signal model per asset from 4-year historical CSV
3. Enters the 30-second polling loop:
   - Detects new 1H bar closes → generates signals → starts 15m A/B monitors
   - Detects new 15m bar closes → advances monitors → opens/closes paper trades
   - Writes closed trades to paper_trades_tier1.csv
   - Survives restart: reloads state.json and replays any missed bars

Output files
------------
  paper_trades_tier1.csv          — trade log (appended, never overwritten)
  paper_trading/state.json        — persistent state (monitors + open trades)
  paper_trading/paper_trader.log  — rotating log (10 MB × 5 files)
"""

import argparse
import json
import os
import sys

# Force UTF-8 console output on Windows (avoids cp1251 codec errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure the project root is on sys.path so all imports resolve correctly
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from paper_trading.logger import get_logger
from paper_trading.config_live import ASSETS, INITIAL_CAPITAL, STATE_FILE, CSV_FILE
from paper_trading.data_feed import LiveFeed
from paper_trading.signal_engine import SignalEngine
from paper_trading.state_store import StateStore
from paper_trading.csv_writer import CsvWriter
from paper_trading.poller import run_poll_loop

log = get_logger()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ARKAD MRK — Automated Paper Trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python paper_trader.py            # start / resume\n"
            "  python paper_trader.py --status   # print current state summary\n"
            "  python paper_trader.py --reset    # wipe state (NOT the CSV log)\n"
        ),
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print a summary of the current paper trading state and exit.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete state.json and start fresh (does NOT delete the CSV trade log).",
    )
    return parser.parse_args()


def _print_status(store: StateStore) -> None:
    """Print a human-readable summary of the current state."""
    state = store.load()

    print(f"\n{'='*60}")
    print(f"  ARKAD MRK — Paper Trader Status")
    print(f"{'='*60}")
    print(f"  Equity          : ${state['equity']:,.2f}  "
          f"(start=${INITIAL_CAPITAL:,.2f}  "
          f"PnL=${state['equity'] - INITIAL_CAPITAL:+,.2f})")
    print(f"  Trades logged   : {state['trade_id_counter'] - 1}")
    print(f"  CSV log         : {CSV_FILE}")
    print()

    print(f"  {'Asset':<12}  {'Last 1H bar':>22}  {'Last 15m bar':>22}  {'Monitor':>12}  {'Trade':>8}")
    print(f"  {'-'*80}")
    for asset in ASSETS:
        last_1h  = state["processed_1h_bars"].get(asset, "-")[:19]
        last_15m = state["processed_15m_bars"].get(asset, "-")[:19]
        mon      = state["monitors"].get(asset, {})
        trade    = state["open_trades"].get(asset, {})
        mon_str  = mon.get("phase", "-") if mon else "-"
        trd_str  = f"#{trade['trade_id']} {trade['direction']}" if trade else "-"
        print(f"  {asset:<12}  {last_1h:>22}  {last_15m:>22}  {mon_str:>12}  {trd_str:>8}")

    if state["open_trades"]:
        print()
        print("  Open trades:")
        for asset, t in state["open_trades"].items():
            print(f"    {asset}  dir={t['direction']}  "
                  f"entry={t['entry_price']:.6f}  "
                  f"stop={t['stop_price']:.6f}  "
                  f"tp={t['tp_price']:.6f}")
    print(f"{'='*60}\n")


def _reset_state(store: StateStore) -> None:
    """Delete state.json after user confirmation."""
    if os.path.exists(STATE_FILE):
        ans = input(
            f"Delete state file '{STATE_FILE}'?\n"
            f"This will clear all monitors and open trades.\n"
            f"The CSV trade log will NOT be deleted.\n"
            f"Type YES to confirm: "
        ).strip()
        if ans == "YES":
            os.remove(STATE_FILE)
            print("State file deleted.  Starting fresh on next run.")
        else:
            print("Cancelled.")
    else:
        print("No state file found — nothing to reset.")


def main() -> None:
    args = _parse_args()
    store = StateStore()

    if args.status:
        _print_status(store)
        return

    if args.reset:
        _reset_state(store)
        return

    # ── Normal startup ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  ARKAD MRK — Paper Trader  starting up")
    log.info("  Assets    : %s", ", ".join(ASSETS))
    log.info("  State file: %s", STATE_FILE)
    log.info("  CSV log   : %s", CSV_FILE)
    log.info("=" * 60)

    # Verify historical data files exist
    from paper_trading.config_live import DATA_DIR
    missing = []
    for asset in ASSETS:
        csv = os.path.join(DATA_DIR, f"{asset}_1h_4y.csv")
        if not os.path.exists(csv):
            missing.append(csv)
    if missing:
        log.error("Missing historical data files:")
        for f in missing:
            log.error("  %s", f)
        log.error("Run  python data/download_all.py  to download missing data.")
        sys.exit(1)

    # Initialise components
    engine     = SignalEngine()
    feed       = LiveFeed()
    csv_writer = CsvWriter()

    # Train models (this takes ~15–30 seconds per asset)
    log.info("Training signal models ...")
    engine.train_all()
    log.info("All models trained.  Entering poll loop.")

    # Run main loop (blocks until CTRL+C)
    run_poll_loop(engine, store, feed, csv_writer)

    log.info("Paper trader stopped.")


if __name__ == "__main__":
    main()
