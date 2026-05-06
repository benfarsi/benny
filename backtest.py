"""
Backtest the RSI+EMA strategy across a range of thresholds.
Auto-fetches 30 days of data from Binance if no local file exists.
Usage: python backtest.py
"""
import os
import time
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

from config import EMA_PERIOD, RSI_PERIOD, SYMBOL, TAKE_PROFIT, STOP_LOSS, TIMEFRAME, TRADE_SIZE

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_FILE = os.path.join(DATA_DIR, SYMBOL.replace("/", "_") + f"_{TIMEFRAME}.csv")
FEE      = 0.001   # 0.1% Binance taker fee — always include this
START    = 1000.0


# ── data ──────────────────────────────────────────────────────────────────────

def _quick_fetch(days: int = 30) -> pd.DataFrame:
    print(f"No local data — fetching {days} days from Binance…")
    exchange = ccxt.binance({"enableRateLimit": True})
    now_ms   = exchange.milliseconds()
    since_ms = now_ms - days * 24 * 60 * 60 * 1000
    rows = []
    while since_ms < now_ms - 60_000:
        candles = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since_ms, limit=1000)
        if not candles:
            break
        rows.extend(candles)
        since_ms = candles[-1][0] + 60_000
        print(f"\r  {len(rows):,} candles…", end="", flush=True)
        time.sleep(exchange.rateLimit / 1000)
    print()
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


# ── indicators — vectorized, computed once ────────────────────────────────────

def _indicators(closes: pd.Series):
    ema = closes.ewm(span=EMA_PERIOD, adjust=False).mean().values

    delta    = closes.diff()
    avg_gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = (100 - 100 / (1 + avg_gain / avg_loss)).values

    return ema, rsi


# ── simulation ────────────────────────────────────────────────────────────────

def simulate(prices: np.ndarray, ema: np.ndarray, rsi: np.ndarray,
             buy_thresh: int, sell_thresh: int = 65) -> dict:
    usdt, btc = START, 0.0
    buy_price = None
    n_buys = n_sells = wins = 0
    equity = np.full(len(prices), START, dtype=float)
    warm   = max(EMA_PERIOD, RSI_PERIOD) + 1

    for i in range(warm, len(prices)):
        p, r, e = prices[i], rsi[i], ema[i]

        if r < buy_thresh and p > e and btc == 0:
            spend     = usdt * TRADE_SIZE
            btc      += (spend / p) * (1 - FEE)
            usdt     -= spend
            buy_price = p
            n_buys   += 1

        else:
            # Option C: RSI sell OR take-profit OR stop-loss
            rsi_exit = r > sell_thresh
            tp_exit  = buy_price is not None and p >= buy_price * (1 + TAKE_PROFIT)
            sl_exit  = buy_price is not None and p <= buy_price * (1 - STOP_LOSS)

            if (rsi_exit or tp_exit or sl_exit) and btc > 0:
                if buy_price and p > buy_price:
                    wins += 1
                usdt      += btc * p * (1 - FEE)
                btc        = 0.0
                buy_price  = None
                n_sells   += 1

        equity[i] = usdt + btc * p

    final = usdt + btc * prices[-1]
    eq    = equity[warm:]

    # Sharpe (annualized from 1m returns)
    rets   = pd.Series(eq).pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_600)) if rets.std() > 0 else 0.0

    # max drawdown
    peaks = np.maximum.accumulate(eq)
    mdd   = ((peaks - eq) / np.where(peaks > 0, peaks, 1)).max() * 100

    return {
        "buy":      buy_thresh,
        "trades":   n_sells,           # completed round-trips
        "win_rate": f"{wins/n_sells*100:.0f}%" if n_sells else "—",
        "ret":      (final - START) / START * 100,
        "sharpe":   round(sharpe, 2),
        "mdd":      round(mdd, 1),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(OUT_FILE):
        df   = pd.read_csv(OUT_FILE)
        t0   = datetime.fromtimestamp(df["timestamp"].iloc[0]  / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        t1   = datetime.fromtimestamp(df["timestamp"].iloc[-1] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"Loaded {len(df):,} candles  ({t0} → {t1})")
    else:
        df = _quick_fetch(days=30)

    closes = df["close"]
    prices = closes.values
    ema, rsi = _indicators(closes)

    bh = (prices[-1] - prices[0]) / prices[0] * 100
    print(f"Buy & hold over this period: {bh:+.2f}%\n")

    thresholds = [30, 33, 35, 38, 40, 42, 45, 48, 50]
    results    = [simulate(prices, ema, rsi, b) for b in thresholds]

    # ── table ──
    print(f"  {'RSI<':>5}  {'Trades':>7}  {'Win%':>6}  {'Return':>8}  {'Sharpe':>7}  {'MaxDD':>7}  ")
    print("  " + "─" * 60)

    best = max(results, key=lambda r: r["sharpe"])

    for r in results:
        is_best    = r["buy"] == best["buy"]
        is_current = r["buy"] == 40
        note = ""
        if is_best and is_current: note = "  ◄ best Sharpe  (your setting)"
        elif is_best:              note = "  ◄ best Sharpe"
        elif is_current:           note = "  ← your current setting"

        print(
            f"  {r['buy']:>5}  {r['trades']:>7}  {r['win_rate']:>6}  "
            f"{r['ret']:>+7.2f}%  {r['sharpe']:>7.2f}  {r['mdd']:>6.1f}%"
            + note
        )

    print()
    cur = next(r for r in results if r["buy"] == 40)
    if best["buy"] == 40:
        print("RSI < 40 IS the optimal buy threshold. Good call.")
    else:
        print(f"Best threshold: RSI < {best['buy']}  "
              f"(Sharpe {best['sharpe']} vs {cur['sharpe']} at RSI 40)")
        if best["sharpe"] - cur["sharpe"] < 0.1:
            print("Difference is small — RSI 40 is fine to keep.")
        else:
            print(f"Consider changing RSI_BUY_THRESHOLD to {best['buy']} in config.py.")


if __name__ == "__main__":
    main()
