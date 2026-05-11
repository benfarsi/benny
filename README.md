# Benny

An ML-driven crypto paper trading bot. Watches six Binance pairs at 1-minute
resolution, runs each candle through an XGBoost classifier trained on ~50
features (multi-timescale returns, volatility regimes, RSI/MACD/Bollinger,
GARCH(1,1) conditional vol, Fear & Greed index, session-of-day, OBV, VWAP gap),
and asks: *does this look like a setup that goes up at least 0.3% in the next
hour?* If yes, it buys with ATR-sized risk and exits via take-profit,
stop-loss, or a portfolio-level drawdown halt.

**This is a learning project.** No real money is involved. The point was to
take crypto trading from "vibes" to a system I can audit end-to-end, and to
get my hands dirty with the math (Sharpe annualization, walk-forward CV
without leakage, GARCH conditional vol, ATR position sizing).

## Live dashboard

The live paper-trader runs at a public URL with `BENNY_PUBLIC=1` set, which
disables the train/resume POST endpoints. The dashboard is read-only — visitors
can browse trades, the equity curve, and the model report, but can't trigger
retraining or override the drawdown halt.

## What's in this repo

- `bot.py`, `web.py` — Flask app + live trading loop, polling Binance every 60s
- `trader.py` — paper-trading bookkeeping (per-symbol PaperTrader with USDT/asset balances and trade log)
- `strategy.py` — feature pipeline + model inference, ATR sizing, BUY/SELL/HOLD signal
- `features.py` — ~50 engineered features, all built with strict no-look-ahead guarantees
- `train.py` — XGBoost training with **time-sorted walk-forward CV** (5 folds, per-symbol simulation inside each fold)
- `backtest.py` — RSI threshold sweep over historical data (sanity check, not the live signal)
- `collector.py` — pulls historical OHLCV from Binance into CSVs
- `sentiment.py` — Fear & Greed index fetch (alternative.me, free)
- `report.py` — round-trip trade analysis (win rate, profit factor, expectancy, ML signal quality)
- `templates/index.html` — full dashboard (Trading / Model / Portfolio / Report)

**What's NOT in this repo:** `data/model.pkl`, `data/model_meta.json`, training
CSVs, and the Fear & Greed cache. The algorithm is fully here; the trained
weights are mine. Reproduce from scratch with the steps below.

## Run it yourself

```bash
pip install -r requirements.txt

# 1. Pull a year of 1m candles for all configured symbols (~10 min, ~500 MB)
python3 collector.py --all 365

# 2. Train (uses --fast for ~3 min instead of ~15 min)
python3 train.py --fast

# 3. Start the web dashboard + live paper-trading loop
python3 web.py

# To run as a public read-only demo:
BENNY_PUBLIC=1 python3 web.py
```

The dashboard binds to `0.0.0.0:5000`.

## Math notes (the part I'm proud of)

The first version of this project shipped two subtle math bugs that quietly
inflated the backtest numbers. Both are now fixed:

**1. Sharpe annualization.** Per-trade returns were being annualized with
`√252` (the daily-returns convention), but Benny doesn't trade once a day —
trade frequency varies by symbol and regime. The correct annualizer is
`√(trades_per_year)`, computed from the test-fold time span. See `train.py:simulate_strategy`.

**2. Cross-symbol CV leakage.** The original "walk-forward" CV concatenated
per-symbol blocks `[BTC, ETH, SOL, BNB, XRP, DOGE]` and split by row index.
Each fold trained on one symbol and tested on the next — at the **same
calendar time.** Crypto symbols are highly correlated intraday, so the model
was effectively peeking at concurrent market events. Fixed by sorting all
rows by timestamp before folding, then running the simulator *per symbol*
inside each fold so the equity curve isn't jumping between $60k BTC and
$0.11 DOGE prices. See `train.py:cross_validate`.

**3. GARCH look-ahead.** `_garch_vol` fit on data `[0, i)` and then assigned
the resulting sigma to **past** rows `[i-50, i)`, leaking 50 minutes of
future information into training samples. Inference was unaffected (only the
last row's value is consumed live), but training was. Fixed by applying the
fitted sigma to forward rows `[i, i+50)`. See `features.py:_garch_vol`.

The honest CV after the fixes is much worse than the leaky version — AUC
dropped from 0.66 to 0.59, average return from +30% to -3%. That's the point:
correct math first, optimize second.

## Caveats I want to be loud about

- **Two weeks of paper trading is not enough data to claim an edge.** Trust
  expectancy and profit factor over short-window Sharpe; the Sharpe on the
  Report tab is essentially noise at n<50 trades.
- **Backtest > live, almost always.** Even with proper CV, expect 30–60%
  shrinkage from slippage, regime drift, and the fact that the future doesn't
  obey the past.
- **No transaction-cost modeling beyond a flat 0.1% fee per side.** Real fills
  on small symbols (DOGE especially) will be worse than backtest assumes.

## License

MIT. Use it, fork it, learn from it — but don't run real money against the
weights I'm not publishing.
