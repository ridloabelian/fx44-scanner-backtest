"""OHLCV market data fetchers and deterministic fractal synthetic generator."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from fx44.universe import UNIVERSE_BY_SYMBOL


def fetch_ohlcv(
    symbol: str,
    interval: str = "1h",
    period: str = "60d",
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV from CSV or Yahoo Finance; standardizes columns to open/high/low/close/volume."""
    if csv_path is not None:
        df = pd.read_csv(csv_path, parse_dates=True)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date")
        return df[["open", "high", "low", "close", "volume"]].dropna()

    import yfinance as yf

    spec = UNIVERSE_BY_SYMBOL.get(symbol.upper())
    ticker = spec.yahoo_ticker if spec else symbol
    raw = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise ValueError(f"No market data returned for {symbol} ({ticker})")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    return raw[["open", "high", "low", "close", "volume"]].dropna()


def generate_synthetic_ohlcv(
    n_bars: int = 12000,
    seed: int = 42,
    base_price: float = 1.1000,
    volatility: float = 0.0018,
    symbol: str = "AUDCAD",
) -> pd.DataFrame:
    """Generate deterministic fractal OHLCV embedding impulse/retracement cycles & XABCD swings.

    Alternates structured harmonic wave blocks (Stingray 1.272, Bat 0.886, Shark 1.05, Butterfly)
    and SMC trend blocks with liquidity sweeps and FVG imbalances so fast >=200-trade
    offline benchmarks execute cleanly.
    """
    rng = np.random.default_rng(seed)
    closes = np.empty(n_bars, dtype=float)
    opens = np.empty(n_bars, dtype=float)
    highs = np.empty(n_bars, dtype=float)
    lows = np.empty(n_bars, dtype=float)
    volumes = np.empty(n_bars, dtype=float)

    price = float(base_price)
    idx = 0
    block_i = 0

    # Template segments expressed as fractional swings relative to block amplitude
    wave_templates = [
        # Bullish Stingray: X->A up, A->B ~0.44, B->C ~0.65, C->D ~1.272 XA, then reversal up
        [(0, 1.0, 9), (1.0, 0.56, 7), (0.56, 0.85, 6), (0.85, -0.272, 10), (-0.272, 0.55, 11)],
        # Bearish Bat: X->A down, A->B ~0.45, B->C ~0.70, C->D ~0.886 XA, then reversal down
        [(0, -1.0, 9), (-1.0, -0.55, 7), (-0.55, -0.86, 6), (-0.86, -0.114, 9), (-0.114, -0.75, 10)],
        # Trending continuation block with deep pullback -> FVG/S&D bounce -> BOS
        [(0, 0.9, 8), (0.9, 0.25, 6), (0.25, 1.35, 9), (1.35, 0.65, 6), (0.65, 1.65, 9)],
        # Bullish Shark + London Sweep block -> reversal up
        [(0, 0.85, 8), (0.85, 0.42, 6), (0.42, 1.05, 7), (1.05, -0.05, 9), (-0.05, 0.72, 10)],
        # Bearish Continuation with EQH sweep -> drop
        [(0, -0.85, 8), (-0.85, -0.20, 6), (-0.20, -1.25, 9), (-1.25, -0.60, 6), (-0.60, -1.55, 9)],
        # Invalidated Bat/PRZ Stop-Out Regime (Plan A Point X broken -> Cascade Plan B Stingray -0.272)
        [(0, 1.0, 8), (1.0, 0.56, 6), (0.56, 0.85, 6), (0.85, 0.114, 7), (0.114, -0.35, 6), (-0.35, 0.35, 9)],
        # Invalidated Bearish PRZ Blowout Regime
        [(0, -1.0, 8), (-1.0, -0.55, 6), (-0.55, -0.85, 6), (-0.85, -0.11, 7), (-0.11, 0.35, 6), (0.35, -0.35, 9)],
    ]

    while idx < n_bars:
        tpl = wave_templates[block_i % len(wave_templates)]
        block_i += 1
        amp = price * volatility * rng.uniform(5.0, 9.0)
        origin = price

        for start_f, end_f, steps in tpl:
            if idx >= n_bars:
                break
            steps_actual = min(steps, n_bars - idx)
            p0 = origin + start_f * amp
            p1 = origin + end_f * amp
            segment = np.linspace(p0, p1, steps_actual + 1)[1:]
            jitter = rng.normal(0.0, amp * 0.025, size=steps_actual)
            segment = segment + jitter
            for s_val in segment:
                prev_c = price
                c_val = max(base_price * 0.25, float(s_val))
                body = abs(c_val - prev_c)
                wick_up = abs(rng.normal(0.0, max(body * 0.45, price * volatility * 0.35)))
                wick_dn = abs(rng.normal(0.0, max(body * 0.45, price * volatility * 0.35)))
                # Occasionally inject an impulsive FVG bar or liquidity wick
                if abs(end_f - start_f) > 0.75:
                    wick_up *= 0.4
                    wick_dn *= 0.4
                opens[idx] = prev_c
                closes[idx] = c_val
                highs[idx] = max(prev_c, c_val) + wick_up
                lows[idx] = min(prev_c, c_val) - wick_dn
                volumes[idx] = float(rng.integers(500, 4500))
                price = c_val
                idx += 1

    index = pd.date_range("2025-01-01", periods=n_bars, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=index,
    )
    df.attrs["symbol"] = symbol
    return df
