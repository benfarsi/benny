import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import ccxt
import yfinance as yf
from flask import Flask, jsonify, render_template, request

from config import (
    CANDLE_LIMIT, CRYPTO_SYMBOLS, SLEEP_SECONDS, STARTING_BALANCE,
    STOCK_SYMBOLS, TAKE_PROFIT, STOP_LOSS, TIMEFRAME,
)
from strategy import get_signal
from trader import PaperTrader

app = Flask(__name__)

_HERE         = os.path.dirname(os.path.abspath(__file__))
_MODEL_META   = os.path.join(_HERE, "data", "model_meta.json")
_TRAIN_SCRIPT = os.path.join(_HERE, "train.py")

_train_state = {"status": "idle", "log": [], "error": None}
_train_lock  = threading.Lock()


def _run_training(fast: bool) -> None:
    with _train_lock:
        _train_state.update(status="training", log=[], error=None)
    cmd = [sys.executable, _TRAIN_SCRIPT] + (["--fast"] if fast else [])
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=_HERE)
        for line in proc.stdout:
            with _train_lock:
                _train_state["log"].append(line.rstrip())
        proc.wait()
        with _train_lock:
            if proc.returncode == 0:
                _train_state["status"] = "done"
                from strategy import reload_model
                reload_model()
            else:
                _train_state["status"] = "error"
                _train_state["error"]  = f"process exited with code {proc.returncode}"
    except Exception as exc:
        with _train_lock:
            _train_state["status"] = "error"
            _train_state["error"]  = str(exc)


ALL_SYMBOLS = CRYPTO_SYMBOLS + STOCK_SYMBOLS


def _blank_state():
    return {
        "price":         None,
        "rsi":           None,
        "ema":           None,
        "signal":        "—",
        "usdt":          STARTING_BALANCE,
        "asset":         0.0,
        "portfolio":     STARTING_BALANCE,
        "pnl_pct":       0.0,
        "trades_count":  0,
        "recent_trades": [],
        "status":        "starting",
        "timestamp":     "—",
        "error":         None,
    }


states     = {s: _blank_state() for s in ALL_SYMBOLS}
traders    = {s: PaperTrader()  for s in ALL_SYMBOLS}
buy_prices = {s: None           for s in ALL_SYMBOLS}
_lock      = threading.Lock()

_equity_history: list[dict] = []
_eq_lock = threading.Lock()


def _crypto_ohlcv(exchange, symbol):
    ohlcv      = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
    closes     = [c[4] for c in ohlcv]
    high       = [c[2] for c in ohlcv]
    low        = [c[3] for c in ohlcv]
    volume     = [c[5] for c in ohlcv]
    timestamps = [c[0] for c in ohlcv]   # ms since epoch
    return closes, high, low, volume, timestamps


def _stock_ohlcv(symbol):
    df = yf.download(symbol, period="5d", interval="1m", progress=False, auto_adjust=True)

    def _col(name):
        c = df[name]
        if c.ndim > 1:
            c = c.iloc[:, 0]
        return c.dropna()

    closes = _col("Close")
    high   = _col("High")
    low    = _col("Low")
    volume = _col("Volume")
    # convert DatetimeIndex to ms timestamps aligned with closes
    ts_ms  = (closes.index.astype("int64") // 1_000_000).tolist()
    n = CANDLE_LIMIT
    return closes.tolist()[-n:], high.tolist()[-n:], low.tolist()[-n:], volume.tolist()[-n:], ts_ms[-n:]


def bot_loop():
    exchange = ccxt.binance({"enableRateLimit": True})
    while True:
        for sym in ALL_SYMBOLS:
            try:
                closes, high, low, volume, timestamps = (
                    _crypto_ohlcv(exchange, sym)
                    if sym in CRYPTO_SYMBOLS
                    else _stock_ohlcv(sym)
                )
                if len(closes) < 20:
                    continue

                price            = closes[-1]
                signal, rsi, ema = get_signal(closes, high=high, low=low, volume=volume, timestamps=timestamps)
                tr               = traders[sym]

                bp = buy_prices[sym]
                if bp is not None and tr.btc > 0:
                    if price >= bp * (1 + TAKE_PROFIT):
                        signal = "SELL"
                    elif price <= bp * (1 - STOP_LOSS):
                        signal = "SELL"

                if signal == "BUY" and tr.btc == 0:
                    if tr.buy(price):
                        buy_prices[sym] = price
                elif signal == "SELL" and tr.btc > 0:
                    if tr.sell(price):
                        buy_prices[sym] = None

                total   = tr.portfolio_value(price)
                pnl_pct = (total - STARTING_BALANCE) / STARTING_BALANCE * 100

                with _lock:
                    states[sym].update(
                        price         = price,
                        rsi           = round(rsi,   2),
                        ema           = round(ema,   2),
                        signal        = signal,
                        usdt          = round(tr.usdt, 2),
                        asset         = round(tr.btc,  6),
                        portfolio     = round(total,   2),
                        pnl_pct       = round(pnl_pct, 2),
                        trades_count  = len(tr.trades),
                        recent_trades = list(reversed(tr.trades[-5:])),
                        status        = "live",
                        timestamp     = datetime.utcnow().isoformat(timespec="seconds") + " UTC",
                        error         = None,
                    )
            except Exception as e:
                with _lock:
                    states[sym]["status"] = "error"
                    states[sym]["error"]  = str(e)

        # snapshot total portfolio for equity curve
        with _lock:
            total_eq = sum(
                s["portfolio"] for s in states.values()
                if s["portfolio"] is not None
            )
        with _eq_lock:
            _equity_history.append({
                "t": datetime.utcnow().isoformat(timespec="seconds"),
                "v": round(total_eq, 2),
            })
            if len(_equity_history) > 500:
                _equity_history.pop(0)

        time.sleep(SLEEP_SECONDS)


# ── API routes ────────────────────────────────────────────────────────────────

def _sanitize(obj):
    """Replace NaN/Inf with None so the response is always valid JSON."""
    import math
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@app.route("/api/model")
def api_model():
    meta = {}
    if os.path.exists(_MODEL_META):
        try:
            with open(_MODEL_META) as f:
                meta = json.load(f)
        except Exception:
            pass
    with _train_lock:
        state = dict(_train_state)
    return jsonify(_sanitize({**meta, **state}))


@app.route("/api/train", methods=["POST"])
def api_train():
    with _train_lock:
        if _train_state["status"] == "training":
            return jsonify({"error": "already training"}), 409
    fast = request.args.get("fast", "0") == "1"
    threading.Thread(target=_run_training, args=(fast,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({s: dict(v) for s, v in states.items()})


@app.route("/api/portfolio")
def api_portfolio():
    with _lock:
        performers = []
        total_value    = 0.0
        total_invested = STARTING_BALANCE * len(ALL_SYMBOLS)
        for sym, s in states.items():
            pv = s.get("portfolio") or STARTING_BALANCE
            total_value += pv
            performers.append({
                "symbol":    sym,
                "portfolio": round(pv, 2),
                "pnl_pct":  s.get("pnl_pct") or 0.0,
                "trades":   s.get("trades_count") or 0,
                "signal":   s.get("signal") or "—",
                "price":    s.get("price"),
            })
        performers.sort(key=lambda x: x["pnl_pct"], reverse=True)
        total_pnl_pct = (total_value - total_invested) / max(total_invested, 1) * 100

    with _eq_lock:
        history = list(_equity_history)

    return jsonify({
        "total_value":    round(total_value, 2),
        "total_invested": round(total_invested, 2),
        "total_pnl_pct":  round(total_pnl_pct, 2),
        "n_symbols":      len(ALL_SYMBOLS),
        "performers":     performers,
        "equity_history": history,
    })


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    print(f"Benny is live at http://localhost:5000  —  {len(ALL_SYMBOLS)} symbols")
    app.run(host="0.0.0.0", port=5000, debug=False)
