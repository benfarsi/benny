"""
Train Benny's ML model on historical BTC/USDT data.
Usage:
  python3 train.py          # full training
  python3 train.py --fast   # skip GARCH, fewer trees (quicker iteration)
"""
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from config import SYMBOL, TIMEFRAME, TAKE_PROFIT, STOP_LOSS, BUY_CONF, SELL_CONF, CRYPTO_SYMBOLS
from features import build
from sentiment import fetch_fg_history

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")

HORIZON           = 60      # predict direction 60 candles (~1 hour) ahead
MIN_MOVE          = 0.003   # label=1 only if price rises ≥0.3%
N_FOLDS           = 5
FEE               = 0.001
TRAIN_WINDOW_DAYS = 90


# ── labels ────────────────────────────────────────────────────────────────────

def make_labels(closes: pd.Series) -> pd.Series:
    """1 if price rises at least MIN_MOVE in HORIZON candles, 0 otherwise."""
    future = closes.shift(-HORIZON)
    return ((future - closes) / closes >= MIN_MOVE).astype(int)


# ── metrics ───────────────────────────────────────────────────────────────────

def simulate_strategy(proba: pd.Series, prices: pd.Series) -> dict:
    position    = False
    entry_price = None
    pnl_list    = []
    usdt        = 1000.0
    btc         = 0.0

    for i in range(len(proba) - HORIZON):
        p     = proba.iloc[i]
        price = prices.iloc[i]

        if not position and p > BUY_CONF:
            spent       = usdt * 0.95
            btc         = spent / price * (1 - FEE)
            usdt       -= spent
            entry_price = price
            cost_basis  = spent
            position    = True

        elif position and (p < SELL_CONF or
                           price >= entry_price * (1 + TAKE_PROFIT) or
                           price <= entry_price * (1 - STOP_LOSS)):
            proceeds = btc * price * (1 - FEE)
            pnl_list.append(proceeds / cost_basis - 1)
            usdt    += proceeds
            btc      = 0.0
            position = False

    final  = usdt + btc * prices.iloc[-1]
    rets   = pd.Series(pnl_list)
    # Annualize per-trade Sharpe by √(trades/year), not √252 (which assumes daily samples).
    # TIMEFRAME=1m → 525,600 candles per year, so years_in_fold = len(prices) / 525_600.
    if len(rets) > 1 and rets.std() > 0:
        trades_per_year = len(rets) * 525_600 / max(len(prices), 1)
        sharpe = rets.mean() / rets.std() * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    return {
        "sharpe":    round(sharpe, 3),
        "total_ret": round((final / 1000 - 1) * 100, 2),
        "n_trades":  len(pnl_list),
        "win_rate":  round((rets > 0).mean() * 100, 1) if len(rets) > 0 else 0.0,
    }


# ── walk-forward cross-validation ─────────────────────────────────────────────

def make_model(fast: bool) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators     = 200 if fast else 600,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        gamma            = 0.1,
        eval_metric      = "logloss",
        verbosity        = 0,
        random_state     = 42,
    )


def _aggregate_sim(sym_sims: list[dict]) -> dict:
    """Average per-symbol Sharpe/win-rate, sum trades, and return mean total_ret."""
    if not sym_sims:
        return {"sharpe": 0.0, "total_ret": 0.0, "n_trades": 0, "win_rate": 0.0}
    return {
        "sharpe":    round(float(np.mean([s["sharpe"]    for s in sym_sims])), 3),
        "total_ret": round(float(np.mean([s["total_ret"] for s in sym_sims])), 2),
        "n_trades":  int(sum   ([s["n_trades"]  for s in sym_sims])),
        "win_rate":  round(float(np.mean([s["win_rate"]  for s in sym_sims])), 1),
    }


def cross_validate(X: pd.DataFrame, y: pd.Series, prices: pd.Series,
                   sym_ids: pd.Series, fast: bool):
    """
    Temporal walk-forward CV.

    Input rows must already be sorted by timestamp (ascending), with sym_ids
    marking which symbol each row came from. Folds are pure time slices, so
    train and test never overlap in calendar time. Inside each fold the
    simulator is run *per symbol* — a single continuous price series across
    multiple assets would be nonsense.
    """
    fold_size = len(X) // (N_FOLDS + 1)
    results   = []

    print(f"\nWalk-forward validation  ({N_FOLDS} folds, horizon={HORIZON} candles)")
    print(f"  {'Fold':>4}  {'Train':>8}  {'Test':>8}  {'AUC':>6}  {'Sharpe':>8}  {'Return':>8}  {'Trades':>7}  {'Win%':>6}")
    print("  " + "─" * 70)

    for fold in range(N_FOLDS):
        tr_end = fold_size * (fold + 1)
        te_end = fold_size * (fold + 2)

        X_tr, y_tr = X.iloc[:tr_end],       y.iloc[:tr_end]
        X_te, y_te = X.iloc[tr_end:te_end], y.iloc[tr_end:te_end]
        p_te       = prices.iloc[tr_end:te_end]
        s_te       = sym_ids.iloc[tr_end:te_end]

        m = make_model(fast)
        m.fit(X_tr, y_tr)

        proba  = pd.Series(m.predict_proba(X_te)[:, 1], index=X_te.index)
        auc    = roc_auc_score(y_te, proba)

        # Per-symbol simulation. Each symbol's rows in the test fold are
        # already time-ordered because the global sort was by timestamp.
        sym_sims = []
        for s in s_te.unique():
            mask = (s_te == s)
            if mask.sum() < 2:
                continue
            sym_sims.append(simulate_strategy(proba[mask], p_te[mask]))
        sim = _aggregate_sim(sym_sims)

        results.append({"auc": auc, **sim})
        print(
            f"  {fold+1:>4}  {len(X_tr):>8,}  {len(X_te):>8,}  "
            f"{auc:>6.3f}  {sim['sharpe']:>8.3f}  "
            f"{sim['total_ret']:>+7.2f}%  {sim['n_trades']:>7}  {sim['win_rate']:>5.1f}%"
        )

    print("  " + "─" * 70)
    avg_auc    = np.mean([r["auc"]       for r in results])
    avg_sharpe = np.mean([r["sharpe"]    for r in results])
    avg_ret    = np.mean([r["total_ret"] for r in results])
    print(f"  {'avg':>4}  {'':>8}  {'':>8}  {avg_auc:>6.3f}  {avg_sharpe:>8.3f}  {avg_ret:>+7.2f}%")

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def _load_symbol(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, symbol.replace("/", "_") + f"_{TIMEFRAME}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if TRAIN_WINDOW_DAYS > 0:
        cutoff_ms = int(df["timestamp"].max()) - TRAIN_WINDOW_DAYS * 24 * 60 * 60 * 1000
        df = df[df["timestamp"] >= cutoff_ms].reset_index(drop=True)
    df["_symbol"] = symbol
    return df


def train():
    fast = "--fast" in sys.argv

    # load all available symbol CSVs, build features per symbol, then combine.
    # features MUST be built per symbol — return calculations cross asset
    # boundaries if we concat first and build after.
    all_X, all_y, all_p, all_ts, all_sym = [], [], [], [], []
    garch = "--fast" not in sys.argv

    print("Fetching Fear & Greed history…")
    fg_history = fetch_fg_history(days=max(TRAIN_WINDOW_DAYS + 10, 100))
    print(f"  {len(fg_history)} days of F&G data" if fg_history else "  F&G fetch failed — training without it")

    for sym in CRYPTO_SYMBOLS:
        df = _load_symbol(sym)
        if df is None:
            print(f"  {sym}: no data — run: python3 collector.py --all")
            continue
        print(f"  {sym}: {len(df):,} candles", end="", flush=True)

        closes_s     = df["close"].astype(float).reset_index(drop=True)
        high_s       = df["high"].astype(float).reset_index(drop=True)   if "high"      in df.columns else None
        low_s        = df["low"].astype(float).reset_index(drop=True)    if "low"       in df.columns else None
        volume_s     = df["volume"].astype(float).reset_index(drop=True) if "volume"    in df.columns else None
        timestamps_s = df["timestamp"].astype(float).reset_index(drop=True) if "timestamp" in df.columns else None

        fg_s = None
        if fg_history and timestamps_s is not None:
            dates = pd.to_datetime(timestamps_s, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
            fg_s  = pd.Series([fg_history.get(d, 50) for d in dates], dtype=float)

        X_s  = build(closes_s, high=high_s, low=low_s, volume=volume_s,
                     timestamps=timestamps_s, fg_value=fg_s, garch=garch)
        y_s  = make_labels(closes_s).reindex(X_s.index).dropna()
        X_s  = X_s.reindex(y_s.index)
        p_s  = closes_s.reindex(y_s.index)
        ts_s = timestamps_s.reindex(y_s.index) if timestamps_s is not None else pd.Series(range(len(y_s)), index=y_s.index)
        X_s, y_s, p_s, ts_s = X_s.iloc[:-HORIZON], y_s.iloc[:-HORIZON], p_s.iloc[:-HORIZON], ts_s.iloc[:-HORIZON]

        all_X.append(X_s.reset_index(drop=True))
        all_y.append(y_s.reset_index(drop=True))
        all_p.append(p_s.reset_index(drop=True))
        all_ts.append(ts_s.reset_index(drop=True))
        all_sym.append(pd.Series([sym] * len(X_s)))
        print(f"  →  {len(X_s):,} samples")

    if not all_X:
        print("No data found. Run: python3 collector.py --all")
        sys.exit(1)

    X   = pd.concat(all_X,   ignore_index=True)
    y   = pd.concat(all_y,   ignore_index=True)
    p   = pd.concat(all_p,   ignore_index=True)
    ts  = pd.concat(all_ts,  ignore_index=True)
    sym = pd.concat(all_sym, ignore_index=True)
    print(f"\nCombined: {len(X):,} samples  |  {X.shape[1]} features  |  {y.mean():.1%} positive class")

    # Sort everything by timestamp so cross-validation splits are temporal.
    # Concatenating per-symbol blocks meant the previous CV trained on BTC
    # and tested on ETH at the same calendar time — leakage via cross-asset
    # correlation. Time-sorting puts all symbols on the same clock.
    order = ts.sort_values(kind="mergesort").index
    X, y, p, ts, sym = (
        X.iloc[order].reset_index(drop=True),
        y.iloc[order].reset_index(drop=True),
        p.iloc[order].reset_index(drop=True),
        ts.iloc[order].reset_index(drop=True),
        sym.iloc[order].reset_index(drop=True),
    )

    results = cross_validate(X, y, p, sym, fast)

    print(f"\nTraining final model on all {len(X):,} samples…")
    final = make_model(fast)
    final.fit(X, y)

    payload = {"model": final, "features": list(X.columns)}
    joblib.dump(payload, MODEL_FILE)
    print(f"Saved → {MODEL_FILE}")

    imp = pd.Series(final.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 15 features:")
    for feat, score in imp.head(15).items():
        bar = "█" * int(score * 300)
        print(f"  {feat:<26} {score:.4f}  {bar}")

    avg_auc    = float(np.mean([r["auc"]       for r in results]))
    avg_sharpe = float(np.mean([r["sharpe"]    for r in results]))
    avg_ret    = float(np.mean([r["total_ret"] for r in results]))
    meta = {
        "trained_at":         datetime.utcnow().isoformat(timespec="seconds"),
        "n_samples":          int(len(X)),
        "n_features":         int(X.shape[1]),
        "avg_auc":            avg_auc,
        "avg_sharpe":         avg_sharpe,
        "avg_ret":            avg_ret,
        "folds":              results,
        "feature_importance": [{"name": n, "score": float(s)} for n, s in imp.head(20).items()],
    }
    meta_file = os.path.join(DATA_DIR, "model_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata → {meta_file}")


if __name__ == "__main__":
    train()
