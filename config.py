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
# SELL_CONF disabled — exits handled by ATR stop-loss and take-profit only.
BUY_CONF  = 0.50   # high-conviction only — fires far less often
SELL_CONF = 0.0    # disabled

# ATR-based position sizing + dynamic stop.
RISK_PCT   = 0.01  # risk 1% of symbol portfolio per trade
ATR_PERIOD = 14    # ATR lookback
ATR_MULT   = 1.5   # stop = entry - ATR_MULT × ATR

# Focus on liquid crypto only — model trained on BTC/USDT crypto data.
# Stocks removed until a separate stock model is trained.
CRYPTO_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT",
]

STOCK_SYMBOLS = []

MAX_DRAWDOWN = 0.08   # halt all buys if portfolio drops 8% from its peak
