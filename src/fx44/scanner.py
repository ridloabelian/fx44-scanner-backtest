"""44-Pair multi-session scanner and Telegram/Discord webhook formatter."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from fx44.data import fetch_ohlcv, generate_synthetic_ohlcv
from fx44.patterns.harmonics import detect_harmonics
from fx44.patterns.smc import detect_london_asia_sweep, find_fresh_snd_zones
from fx44.universe import InstrumentSpec, filter_instruments_by_session


@dataclass(frozen=True)
class ScanHit:
    symbol: str
    asset_class: str
    engine: str  # HARMONIC_PRZ | SMC_SND | LONDON_SWEEP | GOLDEN_PULLBACK
    pattern: str
    direction: str
    price: float
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr: float
    score: float
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)


def scan_dataframe(df: pd.DataFrame, symbol: str, asset_class: str = "forex", recent_bars: int = 35) -> list[ScanHit]:
    """Scan a single OHLCV DataFrame for recent Harmonic PRZ, Fresh S&D, Golden Pullback & London sweeps."""
    hits: list[ScanHit] = []
    if len(df) < 55:
        return hits

    last_bar = len(df) - 1
    last_close = float(df["close"].iloc[-1])

    # 1. Harmonic & Stingray PRZ setups completed near recent bar
    for h in detect_harmonics(df):
        if last_bar - h.bar_d <= recent_bars:
            dist_pct = abs(last_close - h.price_d) / max(abs(h.price_d), 1e-9) * 100.0
            score = round(max(10.0, 95.0 - dist_pct * 12.0 + min(20.0, h.rr_tp2 * 4.0)), 1)
            hits.append(
                ScanHit(
                    symbol=symbol,
                    asset_class=asset_class,
                    engine="HARMONIC_PRZ",
                    pattern=h.pattern,
                    direction=h.direction,
                    price=round(last_close, 5),
                    entry=round(h.price_d, 5),
                    sl=round(h.invalidation_sl, 5),
                    tp1=round(h.tp1_382, 5),
                    tp2=round(h.tp2_500, 5),
                    rr=h.rr_tp2,
                    score=score,
                    notes=f"B/XA={h.b_xa}, D/XA={h.d_xa}, PointB_Trigger={h.point_b_triggered}",
                )
            )

    # 2. Fresh S&D + >=70% Imbalance fill
    for s in find_fresh_snd_zones(df):
        if last_bar - s.bar_idx <= recent_bars:
            risk = abs(s.entry_risk_limit - s.sl_risk)
            rew = abs(s.tp_target - s.entry_risk_limit)
            rr = round(rew / max(risk, 1e-9), 2)
            hits.append(
                ScanHit(
                    symbol=symbol,
                    asset_class=asset_class,
                    engine="SMC_SND",
                    pattern=f"{s.setup_type}_{s.pd_zone}",
                    direction=s.direction,
                    price=round(last_close, 5),
                    entry=round(s.entry_risk_limit, 5),
                    sl=round(s.sl_risk, 5),
                    tp1=round((s.entry_risk_limit + s.tp_target) * 0.5, 5),
                    tp2=round(s.tp_target, 5),
                    rr=rr,
                    score=round(min(98.0, 70.0 + rr * 5.0), 1),
                    notes=f"Imbalance filled {s.imbalance_filled_pct}%, confirm_SL={s.sl_confirm}",
                )
            )

    # 3. Golden Ratio 0.50-0.618 + MA50/200 + BB(20,2) Scalping Pullback
    closes = df["close"]
    ma50 = closes.rolling(50).mean().iloc[-1]
    bb_mid = closes.rolling(20).mean().iloc[-1]
    bb_std = closes.rolling(20).std().iloc[-1]
    swing_hi = float(df["high"].iloc[-45:].max())
    swing_lo = float(df["low"].iloc[-45:].min())
    rng = swing_hi - swing_lo
    if rng > 0 and not np.isnan(ma50) and not np.isnan(bb_mid):
        fib50 = swing_hi - rng * 0.50
        fib618 = swing_hi - rng * 0.618
        fib786 = swing_hi - rng * 0.786
        if last_close >= fib618 and last_close <= fib50 and last_close <= (bb_mid - bb_std * 0.8):
            risk = abs(last_close - fib786)
            rew = abs(swing_hi - last_close)
            rr = round(rew / max(risk, 1e-9), 2)
            if rr >= 1.5:
                hits.append(
                    ScanHit(
                        symbol=symbol,
                        asset_class=asset_class,
                        engine="GOLDEN_PULLBACK",
                        pattern="FIBO_0.5_0.618_BB",
                        direction="BULLISH",
                        price=round(last_close, 5),
                        entry=round(last_close, 5),
                        sl=round(fib786, 5),
                        tp1=round(bb_mid, 5),
                        tp2=round(swing_hi, 5),
                        rr=rr,
                        score=round(min(92.0, 62.0 + rr * 6.0), 1),
                        notes="Golden Ratio 0.50-0.618 + Lower BB + SL 0.786",
                    )
                )

    # 4. London Open Sweep of Asia Range
    sweeps = detect_london_asia_sweep(df.tail(48))
    if sweeps:
        last_sw = sweeps[-1]
        direction = "BEARISH" if "HIGH" in last_sw["signal"] else "BULLISH"
        hits.append(
            ScanHit(
                symbol=symbol,
                asset_class=asset_class,
                engine="LONDON_SWEEP",
                pattern=last_sw["signal"],
                direction=direction,
                price=round(last_close, 5),
                entry=round(last_close, 5),
                sl=round(last_sw["level"], 5),
                tp1=round(swing_lo if direction == "BEARISH" else swing_hi, 5),
                tp2=round(swing_lo if direction == "BEARISH" else swing_hi, 5),
                rr=3.0,
                score=81.0,
                notes="London swept Asia High/Low — drop to M10/M5 for Body Close BOS",
            )
        )

    return hits


def scan_universe(
    session: str = "all",
    offline_demo: bool = True,
    interval: str = "1h",
    period: str = "30d",
    limit: int | None = None,
) -> tuple[list[ScanHit], list[str]]:
    """Scan filtered 44-instrument universe; offline_demo uses deterministic synthetic feeds."""
    specs: list[InstrumentSpec] = filter_instruments_by_session(session)
    if limit is not None:
        specs = specs[:limit]

    all_hits: list[ScanHit] = []
    errors: list[str] = []

    for idx, spec in enumerate(specs):
        try:
            if offline_demo:
                base = 2650.0 if spec.symbol == "XAUUSD" else 90000.0 if spec.symbol == "BTCUSD" else 1.10
                df = generate_synthetic_ohlcv(n_bars=260, seed=100 + idx, base_price=base, symbol=spec.symbol)
            else:
                df = fetch_ohlcv(spec.symbol, interval=interval, period=period)
            all_hits.extend(scan_dataframe(df, symbol=spec.symbol, asset_class=spec.asset_class))
        except Exception as exc:
            errors.append(f"{spec.symbol}: {exc}")

    all_hits.sort(key=lambda h: h.score, reverse=True)
    return all_hits, errors


def format_webhook_message(hits: list[ScanHit], top_n: int = 10) -> str:
    """Format ranked scanner setups for Telegram/Discord markdown webhook alerts."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"**FX44 Scanner Alert** ({now_str})", ""]
    if not hits:
        lines.append("No active high-conviction PRZ / SMC / London Sweep setups — **DIAM (No-Trade)**.")
        return "\n".join(lines)
    for h in hits[:top_n]:
        lines.append(
            f"- **{h.symbol}** | `{h.engine}` ({h.pattern} {h.direction}) | "
            f"Entry `{h.entry}` | SL `{h.sl}` | TP2 `{h.tp2}` | RR `1:{h.rr}` | Score `{h.score}`"
        )
    return "\n".join(lines)
