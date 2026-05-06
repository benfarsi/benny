"""
Download historical OHLCV candles from Binance and save locally.
Usage:
  python collector.py              # update BTC/USDT only (default)
  python collector.py 90           # BTC/USDT, last 90 days
  python collector.py --all        # all CRYPTO_SYMBOLS, default 365 days
  python collector.py --all 90     # all CRYPTO_SYMBOLS, last 90 days
"""
import os
import sys
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

from config import SYMBOL, TIMEFRAME, CRYPTO_SYMBOLS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _out_file(symbol: str) -> str:
    return os.path.join(DATA_DIR, symbol.replace("/", "_") + f"_{TIMEFRAME}.csv")


def collect(symbol: str, days: int = 365) -> None:
    exchange = ccxt.binance({"enableRateLimit": True})
    now_ms   = exchange.milliseconds()
    since_ms = now_ms - days * 24 * 60 * 60 * 1000
    out_file = _out_file(symbol)

    if os.path.exists(out_file):
        existing = pd.read_csv(out_file)
        last_ts  = int(existing["timestamp"].max())
        since_ms = last_ts + 60_000
        dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"[{symbol}] Resuming from {dt} UTC")
    else:
        existing = pd.DataFrame()
        print(f"[{symbol}] Fetching {days}d of {TIMEFRAME} candles from Binance…")

    rows     = []
    expected = max((now_ms - since_ms) // 60_000, 1)
    fetched  = 0

    while since_ms < now_ms - 60_000:
        candles = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since_ms, limit=1000)
        if not candles:
            break
        rows.extend(candles)
        fetched  += len(candles)
        since_ms  = candles[-1][0] + 60_000
        pct = min(fetched / expected * 100, 100)
        ts  = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"\r  [{pct:5.1f}%]  {fetched:>7,} candles  —  {ts} UTC", end="", flush=True)
        time.sleep(exchange.rateLimit / 1000)

    print()
    if not rows:
        print(f"[{symbol}] Already up to date.")
        return

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    new  = pd.DataFrame(rows, columns=cols)
    df   = (
        pd.concat([existing, new], ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    df.to_csv(out_file, index=False)
    print(f"[{symbol}] Saved {len(df):,} candles → {out_file}")


if __name__ == "__main__":
    args = sys.argv[1:]
    all_symbols = "--all" in args
    days_args   = [a for a in args if a.isdigit()]
    days        = int(days_args[0]) if days_args else 365

    symbols = CRYPTO_SYMBOLS if all_symbols else [SYMBOL]

    for sym in symbols:
        collect(sym, days)
