import os
import numpy as np
import pandas as pd
import joblib
from config import EMA_PERIOD, RSI_PERIOD, RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD, BUY_CONF, SELL_CONF

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.pkl")
_ml         = None

def _load():
    global _ml
    if os.path.exists(_MODEL_PATH):
        try:
            _ml = joblib.load(_MODEL_PATH)
            print(f"ML model loaded  ({len(_ml['features'])} features)")
        except Exception as e:
            print(f"Warning: could not load ML model: {e}")

_load()


def calculate_ema(prices: list[float], period: int) -> float:
    s = pd.Series(prices)
    return s.ewm(span=period, adjust=False).mean().iloc[-1]


def calculate_rsi(prices: list[float], period: int) -> float:
    s        = pd.Series(prices)
    delta    = s.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def get_signal(
    closes:     list[float],
    high:       list[float] | None = None,
    low:        list[float] | None = None,
    volume:     list[float] | None = None,
    timestamps: list[float] | None = None,
) -> tuple[str, float, float]:
    ema = calculate_ema(closes, EMA_PERIOD)
    rsi = calculate_rsi(closes, RSI_PERIOD)

    if _ml is not None:
        try:
            from features import build
            h  = pd.Series(high)       if high       is not None else None
            l  = pd.Series(low)        if low        is not None else None
            v  = pd.Series(volume)     if volume     is not None else None
            ts = pd.Series(timestamps) if timestamps is not None else None
            X  = build(pd.Series(closes), high=h, low=l, volume=v, timestamps=ts, garch=True)
            if len(X) > 0:
                row   = X.reindex(columns=_ml["features"], fill_value=0).iloc[[-1]]
                proba = float(_ml["model"].predict_proba(row)[0, 1])

                if proba > BUY_CONF:
                    return "BUY", rsi, ema
                elif proba < SELL_CONF:
                    return "SELL", rsi, ema
                else:
                    return "HOLD", rsi, ema
        except Exception:
            pass

    # RSI + EMA fallback
    latest = closes[-1]
    if rsi < RSI_BUY_THRESHOLD and latest > ema:
        signal = "BUY"
    elif rsi > RSI_SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, rsi, ema


def reload_model():
    """Hot-swap the model after retraining without restarting."""
    _load()
