"""
models/classifier.py — Train a calibrated RandomForest classifier and produce trade signals.

Model choice: RandomForestClassifier + CalibratedClassifierCV
  - Handles non-linear relationships without feature scaling
  - Built-in feature importance for interpretability
  - Isotonic calibration maps raw RF scores to empirical win rates

The model is trained on a chronological train/test split (no shuffling) to
respect the time-series nature of the data and avoid look-ahead bias.

2-class design
--------------
NEUTRAL bars (label == 0) are excluded from the training set.  The model
learns only from clearly-directional examples (BUY vs SELL), which produces
cleaner decision boundaries and better probability spread than a 3-class setup.

At inference time the model scores every bar and returns P(BUY) + P(SELL) = 1.
A bar generates a signal only when one probability exceeds the threshold while
the other stays below it.  With threshold T >= 0.50 the effective entry zones are:
  Long  entry : P(BUY)  >= T  (P(SELL) = 1-P(BUY) < T automatically — no conflict)
  Short entry : P(SELL) >= T  (i.e. P(BUY) <= 1-T)
  Flat        : 1-T < P(BUY) < T  (model is near 50/50 — no trade)

Labels fed to add_labels():
  +1 = BUY   (forward return >= +LABEL_ATR_MULT)
   0 = NEUTRAL (excluded from training; kept in data for signal generation)
  -1 = SELL  (forward return <= -LABEL_ATR_MULT)

signal encoding (output of apply_signals):
   1 = long entry
  -1 = short entry
   0 = flat (neither threshold met, or conflict)
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import config


# Columns that are NOT features (raw OHLCV + labels)
_NON_FEATURE_COLS = {
    # Raw OHLCV
    "open", "high", "low", "close", "volume",
    # Labels and intermediate label columns
    "fwd_return", "fwd_return_atr", "label",
    # Regime and session columns — analysis/filter layers, not learned features
    "trend", "vol_regime", "regime_gate", "session",
    # 4H context layers — categorical strings, used for signal gating only
    "bias_4h",       # SMA crossover context (proven, WF PF = 0.988)
    "bias_4h_adx",   # ADX context (Design A, experimental)
}


# ── Primitives (used by both the main pipeline and walk-forward) ──────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names by excluding OHLCV and label columns."""
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def fit_model(train_df: pd.DataFrame, feature_cols: list[str]) -> CalibratedClassifierCV:
    """
    Train a RandomForestClassifier wrapped in CalibratedClassifierCV.

    No printing — suitable for use inside loops (e.g. walk-forward).

    2-class filter
        NEUTRAL rows (label == 0) are dropped before fitting.  This focuses
        the model on clearly-directional outcomes (BUY vs SELL only), improving
        probability spread and decision boundary quality.

    CalibratedClassifierCV (cv=3, isotonic)
        Maps RF's raw probability outputs to empirical win rates on held-out
        CV folds.  After calibration, P(BUY)=0.55 corresponds to ~55%
        observed win rate on directional bars — thresholds become meaningful.
    """
    # 2-class: train only on directional bars — drop NEUTRAL (label == 0).
    train_df = train_df[train_df["label"] != 0]

    leaf = config.RF_MIN_SAMPLES_LEAF
    base_rf = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=6,
        min_samples_leaf=leaf,
        min_samples_split=leaf * 2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    n_samples = len(train_df)
    method    = "isotonic" if n_samples >= 1000 else "sigmoid"
    model     = CalibratedClassifierCV(base_rf, method=method, cv=3)
    model.fit(train_df[feature_cols], train_df["label"])
    return model


def apply_signals(
    model: CalibratedClassifierCV,
    feature_cols: list[str],
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run a trained model on test_df and return a copy with probability and signal columns.

    Columns added
    -------------
    buy_prob  : P(label == +1) for each bar
    sell_prob : P(label == -1) for each bar
    signal    :  1 = long  (buy_prob  >= BUY_PROB_THRESHOLD,  sell_prob below threshold)
                -1 = short (sell_prob >= SELL_PROB_THRESHOLD, buy_prob  below threshold)
                 0 = flat  (neither threshold met, or both met simultaneously — conflict)

    No printing — suitable for use inside loops (e.g. walk-forward).
    """
    result = test_df.copy()
    proba   = model.predict_proba(result[feature_cols])
    classes = list(model.classes_)

    buy_prob  = proba[:, classes.index(1)]  if  1 in classes else np.zeros(len(proba))
    sell_prob = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(proba))

    result["buy_prob"]  = buy_prob
    result["sell_prob"] = sell_prob

    buy_signal  = buy_prob  >= config.BUY_PROB_THRESHOLD
    sell_signal = sell_prob >= config.SELL_PROB_THRESHOLD

    # Conflicting signal on same bar → stay flat (conservative)
    signal = np.where(buy_signal & sell_signal, 0,
             np.where(buy_signal,                1,
             np.where(sell_signal,              -1, 0)))
    result["signal"] = signal.astype(int)
    return result


# ── Main-pipeline wrappers (keep original interface, now delegate to primitives) ─

def train_classifier(df: pd.DataFrame):
    """
    Train the classifier on the training portion of df, print evaluation metrics.

    Returns
    -------
    model       : fitted CalibratedClassifierCV
    feature_cols: list of feature column names
    X_test      : test features (pd.DataFrame)
    y_test      : test labels   (pd.Series)
    """
    feature_cols = get_feature_columns(df)
    split        = int(len(df) * (1 - config.TEST_SIZE))
    train_df     = df.iloc[:split]
    test_df      = df.iloc[split:]

    model = fit_model(train_df, feature_cols)

    # Evaluate on non-neutral bars only: shows true 2-class accuracy.
    eval_df = test_df[test_df["label"] != 0]
    y_pred  = model.predict(eval_df[feature_cols])

    print(classification_report(eval_df["label"], y_pred, labels=[-1, 1],
                                 target_names=["SELL", "BUY"],
                                 zero_division=0))

    # CalibratedClassifierCV stores one fitted RF per CV fold inside
    # .calibrated_classifiers_[i].estimator — average importances across folds.
    imp_arrays = [
        cal.estimator.feature_importances_
        for cal in model.calibrated_classifiers_
    ]
    importances = pd.Series(np.mean(imp_arrays, axis=0), index=feature_cols)
    top10 = importances.nlargest(10)
    print("      Top-10 feature importances (avg across calibration folds):")
    for feat, imp in top10.items():
        print(f"        {feat:<25} {imp:.4f}")

    return model, feature_cols, test_df[feature_cols], test_df["label"]


def generate_signals(model, feature_cols: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the trained model on the test portion of df and return a DataFrame
    with 'buy_prob' and 'signal' columns added.
    """
    split   = int(len(df) * (1 - config.TEST_SIZE))
    test_df = df.iloc[split:]
    return apply_signals(model, feature_cols, test_df)


def print_prob_diagnostics(signals_df: pd.DataFrame) -> None:
    """
    Print a breakdown of BUY and SELL probabilities on the test set.

    Shows where the probability mass sits for both directions relative to
    their thresholds.  Also prints the raw signal count breakdown.
    """
    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60]
    n_bars = len(signals_df)

    for label, col, active_thr in [
        ("BUY",  "buy_prob",  config.BUY_PROB_THRESHOLD),
        ("SELL", "sell_prob", config.SELL_PROB_THRESHOLD),
    ]:
        if col not in signals_df.columns:
            continue
        probs = signals_df[col]
        print(f"      {label} probability distribution ({n_bars:,} test bars):")
        print(f"        min={probs.min():.4f}  mean={probs.mean():.4f}  max={probs.max():.4f}")
        print(f"        current threshold = {active_thr:.2f}")
        for t in thresholds:
            n   = (probs >= t).sum()
            pct = n / n_bars * 100
            marker = "  <-- active" if t == active_thr else ""
            print(f"        >= {t:.2f} : {n:>5} bars  ({pct:.1f}%){marker}")

    if "signal" in signals_df.columns:
        sig = signals_df["signal"]
        n_long  = (sig ==  1).sum()
        n_short = (sig == -1).sum()
        n_flat  = (sig ==  0).sum()
        print(f"      Signal breakdown: long={n_long}  short={n_short}  flat={n_flat}")
