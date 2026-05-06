import os
import time
import ccxt
from config import SYMBOL, TIMEFRAME, SLEEP_SECONDS, CANDLE_LIMIT
from strategy import get_signal
from trader import PaperTrader


def fetch_ohlcv(exchange: ccxt.Exchange):
    ohlcv      = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
    closes     = [c[4] for c in ohlcv]
    high       = [c[2] for c in ohlcv]
    low        = [c[3] for c in ohlcv]
    volume     = [c[5] for c in ohlcv]
    timestamps = [c[0] for c in ohlcv]
    return closes, high, low, volume, timestamps


def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def main() -> None:
    exchange = ccxt.binance({"enableRateLimit": True})
    trader   = PaperTrader()

    print(f"Starting paper trader for {SYMBOL} — polling every {SLEEP_SECONDS}s.")
    print("Press Ctrl+C to stop.\n")
    time.sleep(2)

    while True:
        try:
            closes, high, low, volume, timestamps = fetch_ohlcv(exchange)
            price = closes[-1]
            signal, rsi, ema = get_signal(closes, high=high, low=low, volume=volume, timestamps=timestamps)

            if signal == "BUY":
                trader.buy(price)
            elif signal == "SELL":
                trader.sell(price)

            clear()
            trader.print_status(price, signal, rsi, ema)

        except ccxt.NetworkError as e:
            print(f"[Network error] {e} — retrying next cycle.")
        except ccxt.ExchangeError as e:
            print(f"[Exchange error] {e} — retrying next cycle.")
        except Exception as e:
            print(f"[Unexpected error] {e} — retrying next cycle.")

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
