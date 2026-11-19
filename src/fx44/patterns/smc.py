"""Pure Price Action & SMC detector (Market Structure, Fresh S&D, Imbalance Fill, Sweeps)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from fx44.patterns.harmonics import extract_alternating_pivots
from fx44.universe import get_wib_session


@dataclass(frozen=True)
class SmcSetup:
    setup_type: str  # CONTINUATION | LIQUIDITY_SWEEP | LONDON_SWEEP
    direction: str  # BULLISH | BEARISH
    bar_idx: int
    timestamp: str
    entry_risk_limit: float
    entry_ltf_confirm: float
    sl_risk: float
    sl_confirm: float
    tp_target: float
    pd_zone: str  # DISCOUNT | PREMIUM
    imbalance_filled_pct: float
    break_mode: str


def detect_market_structure(
    df: pd.DataFrame,
    break_mode: str = "body",
    pivot_len: int = 3,
) -> list[dict]:
    """Detect BOS (SB) & CHoCH/TC events under either 'body' close or 'wick' break rule."""
    if break_mode not in ("body", "wick"):
        raise ValueError("break_mode must be 'body' or 'wick'")

    pivots = extract_alternating_pivots(df, left_right=pivot_len)
    if len(pivots) < 2:
        return []

    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    events: list[dict] = []

    last_hi: float | None = None
    last_lo: float | None = None
    trend = 0
    piv_ptr = 0

    for i in range(pivot_len + 1, len(df)):
        while piv_ptr < len(pivots) and pivots[piv_ptr][0] + pivot_len <= i:
            _, px, pdir = pivots[piv_ptr]
            if pdir == 1:
                last_hi = px
            else:
                last_lo = px
            piv_ptr += 1

        up_val = closes[i] if break_mode == "body" else highs[i]
        dn_val = closes[i] if break_mode == "body" else lows[i]

        if last_hi is not None and up_val > last_hi:
            ev_kind = "TC" if trend == -1 else "SB"
            trend = 1
            events.append({"bar_idx": i, "direction": "BULLISH", "event": ev_kind, "level": last_hi})
            last_hi = highs[i]

        elif last_lo is not None and dn_val < last_lo:
            ev_kind = "TC" if trend == 1 else "SB"
            trend = -1
            events.append({"bar_idx": i, "direction": "BEARISH", "event": ev_kind, "level": last_lo})
            last_lo = lows[i]

    return events


def find_fresh_snd_zones(df: pd.DataFrame, lookahead_bars: int = 24) -> list[SmcSetup]:
    """Locate fresh impulsive S&D zones paired with >=70% Imbalance fill & BOS target."""
    setups: list[SmcSetup] = []
    if len(df) < 30:
        return setups

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    tr = np.maximum(highs - lows, np.abs(closes - opens))
    atr = pd.Series(tr).rolling(14, min_periods=5).mean().to_numpy()
    timestamps = df.index

    i = 16
    while i < len(df) - lookahead_bars - 2:
        cur_atr = atr[i] if not np.isnan(atr[i]) else tr[i]
        if cur_atr <= 1e-9:
            i += 1
            continue

        # Bullish Imbalance (3-candle FVG): low[i] > high[i-2] with strong body on i-1
        body_mid = abs(closes[i - 1] - opens[i - 1])
        if lows[i] > highs[i - 2] and closes[i - 1] > opens[i - 1] and body_mid > cur_atr * 0.85:
            zone_top = highs[i - 2]
            zone_bot = min(lows[i - 2], lows[i - 1])
            imp_hi = float(np.max(highs[i : i + 3]))
            imp_rng = imp_hi - zone_bot
            if imp_rng > cur_atr * 1.4:
                # Track first retest returning >= 70% into S&D
                for j in range(i + 2, min(i + lookahead_bars, len(df) - 2)):
                    retrace_pct = (imp_hi - lows[j]) / max(imp_rng, 1e-9)
                    if lows[j] <= zone_top and lows[j] > zone_bot and retrace_pct >= 0.70:
                        risk_sl = zone_bot - cur_atr * 0.25
                        conf_entry = zone_top + (zone_top - zone_bot) * 0.15
                        conf_sl = zone_bot + (zone_top - zone_bot) * 0.25
                        tp_tgt = imp_hi + imp_rng * 1.35
                        setups.append(
                            SmcSetup(
                                setup_type="CONTINUATION",
                                direction="BULLISH",
                                bar_idx=j,
                                timestamp=str(timestamps[j]),
                                entry_risk_limit=round(float(zone_top), 6),
                                entry_ltf_confirm=round(float(conf_entry), 6),
                                sl_risk=round(float(risk_sl), 6),
                                sl_confirm=round(float(conf_sl), 6),
                                tp_target=round(float(tp_tgt), 6),
                                pd_zone="DISCOUNT",
                                imbalance_filled_pct=round(float(min(100.0, retrace_pct * 100)), 1),
                                break_mode="body",
                            )
                        )
                        i = j + 3
                        break

        # Bearish Imbalance: high[i] < low[i-2]
        elif highs[i] < lows[i - 2] and closes[i - 1] < opens[i - 1] and body_mid > cur_atr * 0.85:
            zone_bot = lows[i - 2]
            zone_top = max(highs[i - 2], highs[i - 1])
            imp_lo = float(np.min(lows[i : i + 3]))
            imp_rng = zone_top - imp_lo
            if imp_rng > cur_atr * 1.4:
                for j in range(i + 2, min(i + lookahead_bars, len(df) - 2)):
                    retrace_pct = (highs[j] - imp_lo) / max(imp_rng, 1e-9)
                    if highs[j] >= zone_bot and highs[j] < zone_top and retrace_pct >= 0.70:
                        risk_sl = zone_top + cur_atr * 0.25
                        conf_entry = zone_bot - (zone_top - zone_bot) * 0.15
                        conf_sl = zone_top - (zone_top - zone_bot) * 0.25
                        tp_tgt = imp_lo - imp_rng * 1.35
                        setups.append(
                            SmcSetup(
                                setup_type="CONTINUATION",
                                direction="BEARISH",
                                bar_idx=j,
                                timestamp=str(timestamps[j]),
                                entry_risk_limit=round(float(zone_bot), 6),
                                entry_ltf_confirm=round(float(conf_entry), 6),
                                sl_risk=round(float(risk_sl), 6),
                                sl_confirm=round(float(conf_sl), 6),
                                tp_target=round(float(tp_tgt), 6),
                                pd_zone="PREMIUM",
                                imbalance_filled_pct=round(float(min(100.0, retrace_pct * 100)), 1),
                                break_mode="body",
                            )
                        )
                        i = j + 3
                        break
        i += 1
    return setups


def detect_eqh_eql_sweeps(df: pd.DataFrame, tol_atr: float = 0.25) -> list[dict]:
    """Detect swept Equal Highs (EQH) / Equal Lows (EQL) rejection bars."""
    pivots = extract_alternating_pivots(df, left_right=3)
    if len(pivots) < 4:
        return []
    tr = np.maximum(df["high"] - df["low"], np.abs(df["close"] - df["open"]))
    atr = tr.rolling(14, min_periods=4).mean().to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    sweeps: list[dict] = []
    high_pivs = [p for p in pivots if p[2] == 1]
    low_pivs = [p for p in pivots if p[2] == -1]

    for k in range(1, len(high_pivs)):
        _, p1, _ = high_pivs[k - 1]
        b2, p2, _ = high_pivs[k]
        cur_atr = atr[b2] if not np.isnan(atr[b2]) else tr.iloc[b2]
        if abs(p1 - p2) <= cur_atr * tol_atr:
            lvl = max(p1, p2)
            for j in range(b2 + 1, min(b2 + 20, len(df))):
                if highs[j] > lvl and closes[j] < lvl:
                    sweeps.append({"bar_idx": j, "kind": "EQH_SWEEP_BEARISH", "level": lvl})
                    break

    for k in range(1, len(low_pivs)):
        _, p1, _ = low_pivs[k - 1]
        b2, p2, _ = low_pivs[k]
        cur_atr = atr[b2] if not np.isnan(atr[b2]) else tr.iloc[b2]
        if abs(p1 - p2) <= cur_atr * tol_atr:
            lvl = min(p1, p2)
            for j in range(b2 + 1, min(b2 + 20, len(df))):
                if lows[j] < lvl and closes[j] > lvl:
                    sweeps.append({"bar_idx": j, "kind": "EQL_SWEEP_BULLISH", "level": lvl})
                    break
    return sweeps


def detect_london_asia_sweep(df: pd.DataFrame) -> list[dict]:
    """Identify London Open (15:00-19:00 WIB) sweeps of Asia session (06:00-14:00 WIB) extremes."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return []
    asia_hi = None
    asia_lo = None
    prev_sess = None
    alerts: list[dict] = []

    for i, ts in enumerate(df.index):
        sess = get_wib_session(ts.to_pydatetime())
        hi = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])
        cl = float(df["close"].iloc[i])
        if sess == "asia":
            if prev_sess != "asia":
                asia_hi, asia_lo = hi, lo
            else:
                asia_hi = max(asia_hi, hi) if asia_hi is not None else hi
                asia_lo = min(asia_lo, lo) if asia_lo is not None else lo
        elif sess == "london" and asia_hi is not None and asia_lo is not None:
            if hi > asia_hi and cl < asia_hi:
                alerts.append({"bar_idx": i, "timestamp": str(ts), "signal": "LONDON_SWEEP_ASIA_HIGH", "level": asia_hi})
                asia_hi = None
            elif lo < asia_lo and cl > asia_lo:
                alerts.append({"bar_idx": i, "timestamp": str(ts), "signal": "LONDON_SWEEP_ASIA_LOW", "level": asia_lo})
                asia_lo = None
        prev_sess = sess
    return alerts
