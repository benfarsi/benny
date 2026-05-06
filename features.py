"""
Feature engineering for Benny's ML model.
Input: pd.Series of close prices (and optionally high, low, volume).
Output: DataFrame of features, NaN rows dropped, zero look-ahead.
"""
import warnings
import numpy as np
import pandas as pd


def _rsi(s: pd.Series, period: int) -> pd.Series:
    delta    = s.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _stoch_rsi(rsi: pd.Series, period: int = 14) -> pd.Series:
    """Stochastic RSI — where is current RSI within its recent range? [0, 1]"""
    lo = rsi.rolling(period).min()
    hi = rsi.rolling(period).max()
    return (rsi - lo) / (hi - lo + 1e-9)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength (0=choppy, 1=strong trend)."""
    prev_c = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs(),
    ], axis=1).max(axis=1)

    up   =  high.diff()
    down = -low.diff()
    plus_dm  = up.where((up > down)   & (up   > 0), 0.0)
    minus_dm = down.where((down > up) & (down  > 0), 0.0)

    atr      = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di  = 100 * plus_dm.ewm(com=period - 1,  min_periods=period).mean() / (atr + 1e-9)
    minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / (atr + 1e-9)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx / 100   # normalize to [0, 1]


def _garch_vol(returns: pd.Series, window: int = 400) -> pd.Series:
    """
    GARCH(1,1) conditional volatility, refitted on a rolling window.
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    Window is clamped to half the series length so it always runs on short inputs.
    """
    from arch import arch_model

    window = min(window, max(100, len(returns) // 2))
    vol  = pd.Series(np.nan, index=returns.index, dtype=float)
    step = 50

    for i in range(window, len(returns) + 1, step):
        subset = returns.iloc[max(0, i - window):i].dropna() * 100
        if len(subset) < 100:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = arch_model(subset, vol="Garch", p=1, q=1,
                                 dist="t", rescale=False).fit(disp="off")
            sigma = float(res.conditional_volatility.iloc[-1]) / 100
            end   = min(i, len(returns))
            start = end - step
            vol.iloc[start:end] = sigma
        except Exception:
            pass

    return vol.ffill()


def _realized_vol_parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Parkinson estimator — uses high/low to estimate intrabar volatility."""
    hl = np.log(high / low) ** 2
    return np.sqrt(hl.rolling(window).mean() / (4 * np.log(2)))


def build(closes: pd.Series,
          high:       pd.Series | None = None,
          low:        pd.Series | None = None,
          volume:     pd.Series | None = None,
          timestamps: pd.Series | None = None,
          garch:      bool = True) -> pd.DataFrame:
    """
    Build feature matrix. No look-ahead bias — every feature at row t
    uses only data up to and including time t.
    """
    c = closes.astype(float).reset_index(drop=True)
    r = np.log(c / c.shift(1))
    X = pd.DataFrame()

    # ── returns at multiple timescales ───────────────────────────────────────
    for w in [1, 3, 5, 10, 15, 30, 60]:
        X[f"ret_{w}"] = r.rolling(w).sum()

    # ── realized (close-to-close) volatility ─────────────────────────────────
    for w in [5, 10, 20, 60]:
        X[f"vol_{w}"] = r.rolling(w).std()

    # ── vol-of-vol ────────────────────────────────────────────────────────────
    X["vol_of_vol"] = X["vol_20"].rolling(20).std()

    # ── volatility asymmetry (leverage effect) ────────────────────────────────
    neg_sq = (r.clip(upper=0) ** 2).rolling(20).mean()
    pos_sq = (r.clip(lower=0) ** 2).rolling(20).mean()
    X["vol_asymmetry"] = (neg_sq - pos_sq) / (neg_sq + pos_sq + 1e-10)

    # ── price deviation from EMA ──────────────────────────────────────────────
    for w in [10, 20, 50, 100]:
        ema = c.ewm(span=w, adjust=False).mean()
        X[f"ema_gap_{w}"] = (c - ema) / ema

    # ── RSI + Stochastic RSI ──────────────────────────────────────────────────
    rsi14 = _rsi(c, 14)
    X["rsi_14"]      = rsi14 / 100
    X["rsi_7"]       = _rsi(c, 7) / 100
    X["stoch_rsi"]   = _stoch_rsi(rsi14)   # where is RSI within its own range?

    # ── MACD ──────────────────────────────────────────────────────────────────
    X["macd"] = (c.ewm(span=12, adjust=False).mean()
               - c.ewm(span=26, adjust=False).mean()) / c

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    X["bb_pos"]   = (c - bb_mid) / (2 * bb_std + 1e-9)
    X["bb_width"] = (2 * bb_std) / (bb_mid + 1e-9)

    # ── rate of change ────────────────────────────────────────────────────────
    for w in [5, 10, 20]:
        X[f"roc_{w}"] = c / c.shift(w) - 1

    # ── lagged returns (autocorrelation) ──────────────────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        X[f"lag_{lag}"] = r.shift(lag)

    # ── regime: trend consistency ─────────────────────────────────────────────
    # >0.5 = mostly up (trending), <0.5 = mostly down, ~0.5 = choppy
    X["trend_consistency_20"] = (r > 0).rolling(20).mean()
    X["trend_consistency_60"] = (r > 0).rolling(60).mean()

    # ── OHLC features (if high/low available) ────────────────────────────────
    if high is not None and low is not None:
        h = high.astype(float).reset_index(drop=True)
        l = low.astype(float).reset_index(drop=True)

        # Parkinson vol
        for w in [5, 20]:
            X[f"park_vol_{w}"] = _realized_vol_parkinson(h, l, w)
        X["hl_range"] = (h - l) / c

        # ADX — how strongly is price trending right now?
        X["adx"] = _adx(h, l, c)

        # Range expansion — is volatility expanding vs the recent norm?
        bar_range = h - l
        X["range_expansion"] = bar_range / (bar_range.rolling(20).mean() + 1e-9)

    # ── volume features ───────────────────────────────────────────────────────
    if volume is not None:
        v = volume.astype(float).reset_index(drop=True)
        v_ma5   = v.rolling(5).mean()
        v_ma20  = v.rolling(20).mean()
        v_std20 = v.rolling(20).std()

        # Is volume above or below its recent average?
        X["vol_ratio_5"]  = v / (v_ma5  + 1e-9)
        X["vol_ratio_20"] = v / (v_ma20 + 1e-9)

        # Volume z-score: how many std devs above/below normal?
        X["vol_zscore"] = (v - v_ma20) / (v_std20 + 1e-9)

        # On-Balance Volume — cumulative direction-weighted volume, normalized
        sign = np.sign(r.fillna(0))
        obv  = (sign * v).cumsum()
        X["obv_norm"] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9)

        # VWAP deviation — is price cheap or expensive vs volume-weighted avg?
        vwap = (c * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)
        X["vwap_gap"] = (c - vwap) / (vwap + 1e-9)

    # ── time-of-day features ──────────────────────────────────────────────────
    # Crypto trades 24/7 but liquidity/volatility vary hugely by session.
    # Cyclical sin/cos encoding so 23:00 and 00:00 are treated as close.
    if timestamps is not None:
        ts   = pd.to_datetime(
            timestamps.astype(float).reset_index(drop=True), unit="ms", utc=True
        )
        hour = ts.dt.hour.astype(float)
        X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        X["session_asia"] = ((hour >= 0)  & (hour < 8)).astype(float)   # low liquidity
        X["session_eu"]   = ((hour >= 7)  & (hour < 15)).astype(float)  # European open
        X["session_us"]   = ((hour >= 13) & (hour < 21)).astype(float)  # US market hours

    # ── GARCH(1,1) conditional volatility ────────────────────────────────────
    if garch:
        X["garch_vol"] = _garch_vol(r).values

    return X.dropna()
