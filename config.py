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

# ML signal thresholds.
# Recent BTC data: positive class rate ~20%, model max proba ~0.67,
# p90=0.28, p99=0.46. Old thresholds (0.65/0.35) were calibrated for a
# bull market and essentially never fire in a sideways/bearish regime.
BUY_CONF  = 0.30   # top ~8% of signals → fires roughly every 12–20 min
SELL_CONF = 0.08   # below median → exit when model is not confident

# Focus on liquid crypto only — model trained on BTC/USDT crypto data.
# Stocks removed until a separate stock model is trained.
CRYPTO_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT",
]

STOCK_SYMBOLS = []
