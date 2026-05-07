import csv
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
from report import generate_report
from strategy import get_signal
from trader import PaperTrader

app = Flask(__name__)

_HERE         = os.path.dirname(os.path.abspath(__file__))
_MODEL_META   = os.path.join(_HERE, "data", "model_meta.json")
_TRAIN_SCRIPT = os.path.join(_HERE, "train.py")
_SNAPSHOT     = os.path.join(_HERE, "data", "snapshot.json")
_TRADE_LOG    = os.path.join(_HERE, "data", "trades.csv")

_LOG_FIELDS = ["time", "symbol", "side", "price", "qty", "usdt_after", "portfolio", "pnl_pct", "ml_proba", "trigger"]
_log_lock   = threading.Lock()


def _append_trade(*, symbol, side, price, qty, usdt_after, portfolio, pnl_pct, ml_proba, trigger):
    os.makedirs(os.path.dirname(_TRADE_LOG), exist_ok=True)
    write_header = not os.path.exists(_TRADE_LOG)
    with _log_lock:
        with open(_TRADE_LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow({
                "time":       datetime.utcnow().isoformat(timespec="seconds"),
                "symbol":     symbol,
                "side":       side,
                "price":      round(price, 6),
                "qty":        round(qty, 8),
                "usdt_after": round(usdt_after, 2),
                "portfolio":  round(portfolio, 2),
                "pnl_pct":    round(pnl_pct, 4),
                "ml_proba":   round(ml_proba, 4) if ml_proba is not None else "",
                "trigger":    trigger,
            })

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
        "ml_proba":      None,
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


def _save_snapshot() -> None:
    os.makedirs(os.path.dirname(_SNAPSHOT), exist_ok=True)
    payload = {
        "traders": {
            sym: {"usdt": tr.usdt, "btc": tr.btc, "trades": tr.trades}
            for sym, tr in traders.items()
        },
        "buy_prices":     buy_prices,
        "equity_history": _equity_history,
    }
    tmp = _SNAPSHOT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, _SNAPSHOT)  # atomic write


def _load_snapshot() -> None:
    if not os.path.exists(_SNAPSHOT):
        return
    try:
        with open(_SNAPSHOT) as f:
            payload = json.load(f)
        for sym, data in payload.get("traders", {}).items():
            if sym in traders:
                traders[sym].usdt   = data["usdt"]
                traders[sym].btc    = data["btc"]
                traders[sym].trades = data["trades"]
        for sym, bp in payload.get("buy_prices", {}).items():
            if sym in buy_prices:
                buy_prices[sym] = bp
        _equity_history.extend(payload.get("equity_history", []))
        print(f"[snapshot] restored from {_SNAPSHOT}")
    except Exception as exc:
        print(f"[snapshot] load failed, starting fresh: {exc}")


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
    _load_snapshot()
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

                price                    = closes[-1]
                signal, rsi, ema, proba  = get_signal(closes, high=high, low=low, volume=volume, timestamps=timestamps)
                tr                       = traders[sym]

                bp      = buy_prices[sym]
                trigger = "ml"
                if bp is not None and tr.btc > 0:
                    if price >= bp * (1 + TAKE_PROFIT):
                        signal  = "SELL"
                        trigger = "take_profit"
                    elif price <= bp * (1 - STOP_LOSS):
                        signal  = "SELL"
                        trigger = "stop_loss"

                if signal == "BUY" and tr.btc == 0:
                    qty_before = tr.btc
                    if tr.buy(price):
                        buy_prices[sym] = price
                        total   = tr.portfolio_value(price)
                        pnl_pct = (total - STARTING_BALANCE) / STARTING_BALANCE * 100
                        _append_trade(symbol=sym, side="BUY", price=price,
                                      qty=tr.btc - qty_before, usdt_after=tr.usdt,
                                      portfolio=total, pnl_pct=pnl_pct,
                                      ml_proba=proba, trigger=trigger)
                elif signal == "SELL" and tr.btc > 0:
                    qty_before = tr.btc
                    if tr.sell(price):
                        buy_prices[sym] = None
                        total   = tr.portfolio_value(price)
                        pnl_pct = (total - STARTING_BALANCE) / STARTING_BALANCE * 100
                        _append_trade(symbol=sym, side="SELL", price=price,
                                      qty=qty_before, usdt_after=tr.usdt,
                                      portfolio=total, pnl_pct=pnl_pct,
                                      ml_proba=proba, trigger=trigger)

                total   = tr.portfolio_value(price)
                pnl_pct = (total - STARTING_BALANCE) / STARTING_BALANCE * 100

                with _lock:
                    states[sym].update(
                        price         = price,
                        rsi           = round(rsi,   2),
                        ema           = round(ema,   2),
                        signal        = signal,
                        ml_proba      = round(proba, 4) if proba is not None else None,
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

        _save_snapshot()
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
                "ml_proba": s.get("ml_proba"),
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


@app.route("/api/report")
def api_report():
    return jsonify(_sanitize(generate_report()))


@app.route("/api/trades")
def api_trades():
    from flask import send_file, Response
    if not os.path.exists(_TRADE_LOG):
        return jsonify([])
    # return as downloadable CSV if ?download=1, otherwise JSON
    if request.args.get("download") == "1":
        return send_file(_TRADE_LOG, mimetype="text/csv",
                         as_attachment=True, download_name="benny_trades.csv")
    rows = []
    with open(_TRADE_LOG, newline="") as f:
        rows = list(csv.DictReader(f))
    return jsonify(rows)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    print(f"Benny is live at http://localhost:5000  —  {len(ALL_SYMBOLS)} symbols")
    app.run(host="0.0.0.0", port=5000, debug=False)
