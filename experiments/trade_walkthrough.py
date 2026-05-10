"""
experiments/trade_walkthrough.py
Пошаговая разборка ОДНОЙ сделки от закрытия 1H бара до выхода.
Показывает реальные значения фичей, предсказания модели, 15m бары, SL/TP.

Сделка: ADAUSDT LONG, сигнал на закрытии 2026-01-01 02:00, entry=0.3327
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from features.generator import generate_features, add_labels
from features.market_regime import add_regime_columns, add_session_column
from models.classifier import get_feature_columns, fit_model
from features.execution_15m import load_5m_as_15m

# ── Целевая сделка ──────────────────────────────────────────────────────────
SYMBOL       = "ADAUSDT"
SIGNAL_TIME  = pd.Timestamp("2026-01-01 02:00")
HIST_1H      = "data/ADAUSDT_1h_4y.csv"
NEW_1H       = "data/ADAUSDT_2026_1h.csv"
NEW_5M       = "data/ADAUSDT_2026_5m.csv"

# ── Параметры ───────────────────────────────────────────────────────────────
SL_ATR_MULT  = 1.5
TP_ATR_MULT  = 2.5
SLIP         = 0.0002
EQUITY       = 10_000.0
POS_PCT      = 0.10

W = 78
SEP = "═" * W
sub = "─" * W


def _load_1h():
    df_h = pd.read_csv(HIST_1H, parse_dates=["date"]).set_index("date")
    df_n = pd.read_csv(NEW_1H,  parse_dates=["date"]).set_index("date")
    df = pd.concat([df_h, df_n])
    return df[~df.index.duplicated(keep="last")].sort_index()


def main():
    # ──────────────────────────────────────────────────────────────────────────
    # ШАГ 0 — Загрузка данных
    # ──────────────────────────────────────────────────────────────────────────
    print(SEP)
    print(f"  ARKAD MRK — Trade Walkthrough")
    print(f"  {SYMBOL}  |  Signal time: {SIGNAL_TIME}  |  Direction: LONG")
    print(SEP)

    df_1h = _load_1h()
    bar = df_1h.loc[SIGNAL_TIME]

    print(f"\n[ШАГ 0] СЫРАЯ СВЕЧА (1H бар, закрылся {SIGNAL_TIME})")
    print(sub)
    print(f"  open   = {bar['open']:.4f}")
    print(f"  high   = {bar['high']:.4f}")
    print(f"  low    = {bar['low']:.4f}")
    print(f"  close  = {bar['close']:.4f}   ← это reference_price")
    print(f"  volume = {bar['volume']:,.0f}")

    # ──────────────────────────────────────────────────────────────────────────
    # ШАГ 1 — Генерация фичей
    # ──────────────────────────────────────────────────────────────────────────
    df_feat = generate_features(df_1h.copy())
    df_feat = add_labels(df_feat)
    df_feat = add_regime_columns(df_feat)
    df_feat = add_session_column(df_feat)
    df_feat = df_feat.dropna()

    feat_row = df_feat.loc[SIGNAL_TIME]

    print(f"\n[ШАГ 1] ФИЧИ НА БАРЕ {SIGNAL_TIME} (что видит модель)")
    print(sub)
    show = [
        ("ret_4",           "лог-доход за 4ч"),
        ("ret_24",          "лог-доход за 24ч"),
        ("ret_48",          "лог-доход за 48ч"),
        ("sma_20_ratio",    "цена − SMA20 (%)"),
        ("sma_50_ratio",    "цена − SMA50 (%)"),
        ("sma_cross_20_50", "SMA20 vs SMA50"),
        ("rsi",             "RSI(14), сырой"),
        ("rsi_norm",        "RSI центр-0"),
        ("rsi_slope",       "наклон RSI(3)"),
        ("atr_pct",         "ATR(14) в % от цены"),
        ("atr_expansion_14","расширение волы"),
        ("bb_pos",          "позиция в Bollinger"),
        ("bb_width",        "ширина Bollinger"),
        ("vol_ratio",       "объём vs MA(20)"),
        ("vol_spike_5",     "объём vs MA(5)"),
        ("body_ratio",      "тело свечи / range"),
        ("upper_wick",      "верхний фитиль"),
        ("lower_wick",      "нижний фитиль"),
        ("macd_hist",       "MACD гистограмма"),
        ("macd_hist_slope", "ускорение MACD"),
    ]
    for col, desc in show:
        if col in feat_row.index:
            v = feat_row[col]
            print(f"  {col:<22} = {v:>+10.5f}    ({desc})")

    print(f"\n  vol_regime = {feat_row['vol_regime']}    (фильтр BLOCK_LOW_VOL)")
    print(f"  trend      = {feat_row['trend']}")
    print(f"  session    = {feat_row['session']}")

    # ──────────────────────────────────────────────────────────────────────────
    # ШАГ 2 — Модель → P(BUY)
    # ──────────────────────────────────────────────────────────────────────────
    train = df_feat[df_feat.index < SIGNAL_TIME]
    feat_cols = get_feature_columns(train)
    model = fit_model(train, feat_cols)

    X_now = df_feat.loc[[SIGNAL_TIME], feat_cols]
    proba = model.predict_proba(X_now)[0]
    classes = list(model.classes_)
    p_buy  = proba[classes.index(1)]
    p_sell = proba[classes.index(-1)]

    print(f"\n[ШАГ 2] МОДЕЛЬ → ВЕРОЯТНОСТИ")
    print(sub)
    print(f"  Trained on: {len(train):,} баров до {SIGNAL_TIME}")
    print(f"  Features used: {len(feat_cols)}")
    print(f"")
    print(f"  P(BUY)  = {p_buy:.4f}")
    print(f"  P(SELL) = {p_sell:.4f}")
    print(f"")
    print(f"  Порог BUY  ≥ 0.55  →  {'✓ LONG сигнал' if p_buy >= 0.55 else '✗ нет сигнала'}")
    print(f"  Порог SELL ≥ 0.55  →  {'✓ SHORT сигнал' if p_sell >= 0.55 else '✗ нет сигнала'}")

    # ──────────────────────────────────────────────────────────────────────────
    # ШАГ 3 — 1H фильтры
    # ──────────────────────────────────────────────────────────────────────────
    atr_pct_now    = feat_row["atr_pct"]
    atr_median_72  = df_feat["atr_pct"].rolling(72).median().loc[SIGNAL_TIME]
    vol_ratio      = atr_pct_now / atr_median_72

    print(f"\n[ШАГ 3] 1H ФИЛЬТРЫ")
    print(sub)
    print(f"  Low-vol проверка:")
    print(f"    atr_pct сейчас      = {atr_pct_now:.5f}")
    print(f"    median(atr_pct,72)  = {atr_median_72:.5f}")
    print(f"    ratio               = {vol_ratio:.3f}")
    print(f"    нужно ≥ 0.75        →  {'✓ PASS' if vol_ratio >= 0.75 else '✗ BLOCK (low_vol)'}")
    print(f"  Открытая позиция?      → нет (предположим)")

    # ──────────────────────────────────────────────────────────────────────────
    # ШАГ 4 — 15m бары после сигнала
    # ──────────────────────────────────────────────────────────────────────────
    df_15m = load_5m_as_15m(NEW_5M)
    entry_start = SIGNAL_TIME + pd.Timedelta(hours=1)  # т.е. 03:00 — первый бар после
    # Но Approach A смотрит первые 4 бара, начиная от 02:00 (закрытия сигнала)
    # entry_start в коде = ts + 1H, но это сделано для look-ahead защиты.
    # Реально первые 4 бара 15m идут с 02:15 до 03:00 (closed='left' label='left').
    # Покажу окно с 02:00 (первый бар, который полностью ПОСЛЕ закрытия 1H 02:00)

    pos = df_15m.index.searchsorted(entry_start)
    win_a = df_15m.iloc[pos:pos+4]
    win_b = df_15m.iloc[pos+4:pos+8]

    direction = 1   # LONG
    ref = float(bar["close"])

    print(f"\n[ШАГ 4] APPROACH A — фильтр моментума (4 бара 15m, {entry_start} → +1ч)")
    print(sub)
    print(f"  {'Время':<19}  {'open':>9}  {'close':>9}  {'диф':>8}  {'Aligned?':>10}")
    aligned = 0
    for ts15, b15 in win_a.iterrows():
        diff = b15["close"] - b15["open"]
        ok = (diff > 0) if direction == 1 else (diff < 0)
        aligned += int(ok)
        flag = "✓" if ok else "✗"
        print(f"  {str(ts15)[:16]:<19}  {b15['open']:>9.4f}  {b15['close']:>9.4f}  "
              f"{diff:>+8.5f}  {flag:>10}")
    print(f"")
    print(f"  Aligned bars: {aligned}/4    →  {'✓ A PASS' if aligned >= 2 else '✗ A FAIL → CANCEL'}")

    if aligned < 2:
        print("\n  Сделка отменена. Дальше не идём.")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # ШАГ 5 — Approach B (пуллбэк)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[ШАГ 5] APPROACH B — пуллбэк (бары 5-8 после сигнала)")
    print(sub)
    print(f"  ref_price (1H close) = {ref:.4f}")
    print(f"  Ищем: бар закрылся ниже ref → следующий закрылся вверх → ВХОД")
    print(f"")
    print(f"  {'Время':<19}  {'open':>9}  {'close':>9}  {'<ref?':>7}  {'close>open?':>13}  {'Phase':>10}")

    phase = 0
    entry_px = None
    for ts15, b15 in win_b.iterrows():
        cl = float(b15["close"])
        op = float(b15["open"])
        below = "✓" if cl < ref else "✗"
        bullish = "✓" if cl > op else "✗"

        action = ""
        if phase == 0 and direction == 1 and cl < ref:
            phase = 1
            action = "→ Phase 1"
        elif phase == 1 and direction == 1 and cl > op:
            entry_px = cl
            action = "← ENTRY"
        elif phase == 0:
            action = "Phase 0"
        else:
            action = "Phase 1"

        print(f"  {str(ts15)[:16]:<19}  {op:>9.4f}  {cl:>9.4f}  {below:>7}  "
              f"{bullish:>13}  {action:>10}")
        if entry_px is not None:
            break

    if entry_px is not None:
        print(f"\n  Пуллбэк найден → entry_price = {entry_px:.4f}")
    else:
        entry_px = ref
        print(f"\n  Пуллбэк не найден за 4 бара → fallback: entry_price = {ref:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # ШАГ 6 — SL/TP/размер позиции
    # ──────────────────────────────────────────────────────────────────────────
    atr_dollar = atr_pct_now * ref
    entry_fill = entry_px * (1 + SLIP)
    sl_price   = entry_fill - SL_ATR_MULT * atr_dollar
    tp_price   = entry_fill + TP_ATR_MULT * atr_dollar
    pos_value  = EQUITY * POS_PCT
    units      = pos_value / entry_fill
    risk_1r    = units * (entry_fill - sl_price)
    tp_gain    = units * (tp_price - entry_fill)

    print(f"\n[ШАГ 6] РАСЧЁТ ВХОДА, SL, TP, РАЗМЕРА")
    print(sub)
    print(f"  ATR(14) на 1H        = {atr_pct_now:.5f}  (= {atr_pct_now*100:.3f}% от цены)")
    print(f"  ATR в долларах       = ATR_pct × close = {atr_dollar:.5f}")
    print(f"  entry_pre_slip       = {entry_px:.4f}")
    print(f"  entry_fill           = entry × (1 + 0.0002) = {entry_fill:.4f}")
    print(f"")
    print(f"  SL = entry − 1.5×ATR = {sl_price:.4f}")
    print(f"  TP = entry + 2.5×ATR = {tp_price:.4f}")
    print(f"  R:R                  = 2.5 / 1.5 = 1.667")
    print(f"")
    print(f"  position_value       = ${pos_value:,.2f}  (10% от ${EQUITY:,.0f})")
    print(f"  units (ADA)          = {units:,.2f}")
    print(f"  риск 1R              = ${risk_1r:.2f}")
    print(f"  потенциал TP         = ${tp_gain:.2f}  (= 1.667R)")

    # ──────────────────────────────────────────────────────────────────────────
    # ШАГ 7 — Эволюция трейда от входа до выхода
    # ──────────────────────────────────────────────────────────────────────────
    max_hold = 24
    after = df_1h.loc[SIGNAL_TIME:].iloc[1:max_hold+1]   # следующие до 24 баров

    print(f"\n[ШАГ 7] ЭВОЛЮЦИЯ ТРЕЙДА ПО 1H БАРАМ (max hold = 24h)")
    print(sub)
    print(f"  {'Бар':>3}  {'Время':<19}  {'low':>9}  {'high':>9}  "
          f"{'SL hit?':>8}  {'TP hit?':>8}")

    exit_reason = None
    exit_bar    = None
    exit_price  = None
    for i, (ts, b) in enumerate(after.iterrows(), 1):
        sl_hit = b["low"]  <= sl_price
        tp_hit = b["high"] >= tp_price
        flag_sl = "★" if sl_hit else "·"
        flag_tp = "★" if tp_hit else "·"
        print(f"  {i:>3}  {str(ts)[:16]:<19}  {b['low']:>9.4f}  {b['high']:>9.4f}  "
              f"{flag_sl:>8}  {flag_tp:>8}")

        # SL имеет приоритет при одном и том же баре
        if sl_hit:
            exit_reason = "stop"
            exit_bar    = i
            exit_price  = sl_price * (1 - SLIP)
            break
        if tp_hit:
            exit_reason = "tp"
            exit_bar    = i
            exit_price  = tp_price                 # лимит, без проскальзывания
            break

    if exit_reason is None:
        # Time exit на 24-м баре
        last = after.iloc[-1]
        exit_reason = "time"
        exit_bar    = len(after)
        exit_price  = float(last["close"]) * (1 - SLIP)

    pnl_gross  = units * (exit_price - entry_fill)
    commission = (entry_fill + exit_price) * units * 0.001
    pnl_net    = pnl_gross - commission
    r_mult     = pnl_net / risk_1r

    print(f"\n[ШАГ 8] ИТОГ ТРЕЙДА")
    print(sub)
    print(f"  Выход на баре #{exit_bar}, причина: {exit_reason.upper()}")
    print(f"  exit_price        = {exit_price:.4f}")
    print(f"  gross PnL         = ({exit_price:.4f} − {entry_fill:.4f}) × {units:,.2f}"
          f"  = ${pnl_gross:+,.2f}")
    print(f"  commission        = ${commission:.2f}")
    print(f"  NET PnL           = ${pnl_net:+,.2f}")
    print(f"  R-multiple        = {r_mult:+.3f}R")

    print(f"\n{SEP}")
    print(f"  Сравни с фактом из лога: PnL_orig = +27.98,  R = +1.524")
    print(SEP)


if __name__ == "__main__":
    main()
