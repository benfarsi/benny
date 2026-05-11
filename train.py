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

def make_labels(closes: pd.Series, min_move: float = MIN_MOVE, horizon: int = HORIZON) -> pd.Series:
    """1 if price rises at least `min_move` in `horizon` candles, 0 otherwise."""
    future = closes.shift(-horizon)
    return ((future - closes) / closes >= min_move).astype(int)


# ── metrics ───────────────────────────────────────────────────────────────────

def simulate_strategy(proba: pd.Series, prices: pd.Series,
                      buy_conf: float = BUY_CONF, sell_conf: float = SELL_CONF) -> dict:
    position    = False
    entry_price = None
    pnl_list    = []
    usdt        = 1000.0
    btc         = 0.0
    equity_curve = []

    for i in range(len(proba) - HORIZON):
        p     = proba.iloc[i]
        price = prices.iloc[i]

        if not position and p > buy_conf:
            spent       = usdt * 0.95
            btc         = spent / price * (1 - FEE)
            usdt       -= spent
            entry_price = price
            cost_basis  = spent
            position    = True

        elif position and (p < sell_conf or
                           price >= entry_price * (1 + TAKE_PROFIT) or
                           price <= entry_price * (1 - STOP_LOSS)):
            proceeds = btc * price * (1 - FEE)
            pnl_list.append(proceeds / cost_basis - 1)
            usdt    += proceeds
            btc      = 0.0
            position = False

        equity_curve.append(usdt + btc * price)

    final  = usdt + btc * prices.iloc[-1]
    rets   = pd.Series(pnl_list)
    eq     = pd.Series(equity_curve) if equity_curve else pd.Series([1000.0])

    # ── strategy max drawdown ──
    peak     = eq.cummax()
    dd_curve = (eq - peak) / peak
    max_dd   = float(dd_curve.min()) * 100 if len(dd_curve) else 0.0

    # Annualize per-trade Sharpe by √(trades/year). TIMEFRAME=1m → 525,600 candles/year.
    # Cap absurd values: when stop-loss-only exits produce near-zero std,
    # |Sharpe| can blow up to thousands. Real-world Sharpe rarely exceeds ±5.
    if len(rets) > 1 and rets.std() > 0:
        trades_per_year = len(rets) * 525_600 / max(len(prices), 1)
        raw_sharpe = rets.mean() / rets.std() * np.sqrt(trades_per_year)
        sharpe     = float(np.clip(raw_sharpe, -10, 10))
        downside   = rets[rets < 0]
        if len(downside) > 1 and downside.std() > 0:
            raw_sortino = rets.mean() / downside.std() * np.sqrt(trades_per_year)
            sortino     = float(np.clip(raw_sortino, -10, 10))
        else:
            sortino = 0.0
    else:
        sharpe = sortino = 0.0

    total_ret_pct = (final / 1000 - 1) * 100
    calmar = (total_ret_pct / abs(max_dd)) if max_dd < 0 else 0.0

    # ── buy-and-hold baseline (same window, same starting capital) ──
    bh_ret_pct = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    bh_eq      = 1000 * (prices / prices.iloc[0])
    bh_peak    = bh_eq.cummax()
    bh_max_dd  = float(((bh_eq - bh_peak) / bh_peak).min()) * 100

    bh_rets = prices.pct_change().dropna()
    if len(bh_rets) > 1 and bh_rets.std() > 0:
        bh_sharpe = float(bh_rets.mean() / bh_rets.std() * np.sqrt(525_600))
        bh_sharpe = float(np.clip(bh_sharpe, -10, 10))
    else:
        bh_sharpe = 0.0

    return {
        "sharpe":    round(sharpe, 3),
        "sortino":   round(sortino, 3),
        "calmar":    round(calmar, 3),
        "total_ret": round(total_ret_pct, 2),
        "max_dd":    round(max_dd, 2),
        "n_trades":  len(pnl_list),
        "win_rate":  round((rets > 0).mean() * 100, 1) if len(rets) > 0 else 0.0,
        "bh_ret":    round(bh_ret_pct, 2),
        "bh_max_dd": round(bh_max_dd, 2),
        "bh_sharpe": round(bh_sharpe, 3),
        "alpha":     round(total_ret_pct - bh_ret_pct, 2),
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
    """Average per-symbol metrics (mean of capped Sharpes), sum trades, equal-weight returns."""
    if not sym_sims:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "total_ret": 0.0,
                "max_dd": 0.0, "n_trades": 0, "win_rate": 0.0,
                "bh_ret": 0.0, "bh_max_dd": 0.0, "bh_sharpe": 0.0, "alpha": 0.0}
    mean = lambda k: float(np.mean([s[k] for s in sym_sims]))
    return {
        "sharpe":    round(mean("sharpe"),    3),
        "sortino":   round(mean("sortino"),   3),
        "calmar":    round(mean("calmar"),    3),
        "total_ret": round(mean("total_ret"), 2),
        "max_dd":    round(mean("max_dd"),    2),
        "n_trades":  int(sum(s["n_trades"] for s in sym_sims)),
        "win_rate":  round(mean("win_rate"),  1),
        "bh_ret":    round(mean("bh_ret"),    2),
        "bh_max_dd": round(mean("bh_max_dd"), 2),
        "bh_sharpe": round(mean("bh_sharpe"), 3),
        "alpha":     round(mean("alpha"),     2),
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
    hdr = f"  {'Fold':>4}  {'AUC':>5}  {'Sharpe':>7}  {'Sortino':>7}  {'Calmar':>6}  {'Return':>8}  {'MaxDD':>7}  {'Alpha':>8}  {'BH Ret':>8}  {'BH DD':>7}  {'BH Shp':>7}  {'Trades':>6}  {'Win%':>5}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

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
            f"  {fold+1:>4}  {auc:>5.3f}  {sim['sharpe']:>+7.2f}  {sim['sortino']:>+7.2f}  "
            f"{sim['calmar']:>+6.2f}  {sim['total_ret']:>+7.2f}%  {sim['max_dd']:>+6.2f}%  "
            f"{sim['alpha']:>+7.2f}%  {sim['bh_ret']:>+7.2f}%  {sim['bh_max_dd']:>+6.2f}%  "
            f"{sim['bh_sharpe']:>+7.2f}  {sim['n_trades']:>6}  {sim['win_rate']:>4.1f}%"
        )

    print("  " + "─" * (len(hdr) - 2))
    avg_auc     = np.mean([r["auc"]       for r in results])
    avg_sharpe  = np.mean([r["sharpe"]    for r in results])
    avg_sortino = np.mean([r["sortino"]   for r in results])
    avg_calmar  = np.mean([r["calmar"]    for r in results])
    avg_ret     = np.mean([r["total_ret"] for r in results])
    avg_dd      = np.mean([r["max_dd"]    for r in results])
    avg_alpha   = np.mean([r["alpha"]     for r in results])
    avg_bh_ret  = np.mean([r["bh_ret"]    for r in results])
    avg_bh_dd   = np.mean([r["bh_max_dd"] for r in results])
    avg_bh_shp  = np.mean([r["bh_sharpe"] for r in results])
    print(
        f"  {'avg':>4}  {avg_auc:>5.3f}  {avg_sharpe:>+7.2f}  {avg_sortino:>+7.2f}  "
        f"{avg_calmar:>+6.2f}  {avg_ret:>+7.2f}%  {avg_dd:>+6.2f}%  "
        f"{avg_alpha:>+7.2f}%  {avg_bh_ret:>+7.2f}%  {avg_bh_dd:>+6.2f}%  {avg_bh_shp:>+7.2f}"
    )

    # Verdict line — does the strategy actually beat buy-and-hold?
    print()
    if avg_alpha > 0 and avg_sharpe > avg_bh_shp:
        print(f"  ✓  Strategy beats buy-and-hold by {avg_alpha:+.2f}% return and {avg_sharpe - avg_bh_shp:+.2f} Sharpe.")
    elif avg_alpha > 0:
        print(f"  ~  Strategy edges buy-and-hold on return ({avg_alpha:+.2f}%) but loses on risk-adjusted (Sharpe {avg_sharpe:+.2f} vs {avg_bh_shp:+.2f}).")
    else:
        print(f"  ✗  Strategy LOSES to buy-and-hold: alpha {avg_alpha:+.2f}%, Sharpe {avg_sharpe:+.2f} vs BH {avg_bh_shp:+.2f}. Don't deploy.")

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

    avg_auc     = float(np.mean([r["auc"]       for r in results]))
    avg_sharpe  = float(np.mean([r["sharpe"]    for r in results]))
    avg_sortino = float(np.mean([r["sortino"]   for r in results]))
    avg_calmar  = float(np.mean([r["calmar"]    for r in results]))
    avg_ret     = float(np.mean([r["total_ret"] for r in results]))
    avg_dd      = float(np.mean([r["max_dd"]    for r in results]))
    avg_alpha   = float(np.mean([r["alpha"]     for r in results]))
    avg_bh_ret  = float(np.mean([r["bh_ret"]    for r in results]))
    avg_bh_shp  = float(np.mean([r["bh_sharpe"] for r in results]))
    meta = {
        "trained_at":         datetime.utcnow().isoformat(timespec="seconds"),
        "n_samples":          int(len(X)),
        "n_features":         int(X.shape[1]),
        "avg_auc":            avg_auc,
        "avg_sharpe":         avg_sharpe,
        "avg_sortino":        avg_sortino,
        "avg_calmar":         avg_calmar,
        "avg_ret":            avg_ret,
        "avg_max_dd":         avg_dd,
        "avg_alpha":          avg_alpha,
        "avg_bh_ret":         avg_bh_ret,
        "avg_bh_sharpe":      avg_bh_shp,
        "folds":              results,
        "feature_importance": [{"name": n, "score": float(s)} for n, s in imp.head(20).items()],
    }
    meta_file = os.path.join(DATA_DIR, "model_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata → {meta_file}")


if __name__ == "__main__":
    train()
