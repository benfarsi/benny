SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
STARTING_BALANCE = 1000.0
EMA_PERIOD = 20
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 40
RSI_SELL_THRESHOLD = 65
TRADE_SIZE    = 0.95
SLEEP_SECONDS = 60
CANDLE_LIMIT  = 300

TAKE_PROFIT = 0.03
STOP_LOSS   = 0.015

# ML signal thresholds — raise BUY_CONF to only trade when very confident,
# so 0.1% fees don't eat the edge on marginal calls.
BUY_CONF  = 0.65
SELL_CONF = 0.35

# Focus on liquid crypto only — model trained on BTC/USDT crypto data.
# Stocks removed until a separate stock model is trained.
CRYPTO_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT",
]

STOCK_SYMBOLS = []
