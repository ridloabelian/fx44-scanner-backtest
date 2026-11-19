"""Harmonic & Proprietary Stingray Pattern detector."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HarmonicSetup:
    pattern: str  # Stingray, Bat, Butterfly, Crab, Shark, ABCD
    direction: str  # BULLISH | BEARISH
    bar_d: int
    timestamp: str
    price_x: float
    price_a: float
    price_b: float
    price_c: float
    price_d: float
    prz_low: float
    prz_high: float
    invalidation_sl: float
    tp1_382: float
    tp2_500: float
    tp3_618: float
    point_b_triggered: bool
    plan_b_stingray_d: float | None
    rr_tp2: float
    b_xa: float
    d_xa: float


def extract_alternating_pivots(df: pd.DataFrame, left_right: int = 3) -> list[tuple[int, float, int]]:
    """Extract strictly alternating swing pivots: (bar_idx, price, dir: 1=High/-1=Low)."""
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = len(df)
    raw: list[tuple[int, float, int]] = []
    for i in range(left_right, n - left_right):
        w_hi = hi[i - left_right : i + left_right + 1]
        w_lo = lo[i - left_right : i + left_right + 1]
        if hi[i] >= np.max(w_hi) and hi[i] > hi[i - 1]:
            raw.append((i, float(hi[i]), 1))
        if lo[i] <= np.min(w_lo) and lo[i] < lo[i - 1]:
            raw.append((i, float(lo[i]), -1))

    pivots: list[tuple[int, float, int]] = []
    for bar_i, px, pdir in raw:
        if not pivots:
            pivots.append((bar_i, px, pdir))
            continue
        prev_bar, prev_px, prev_dir = pivots[-1]
        if pdir == prev_dir:
            if (pdir == 1 and px > prev_px) or (pdir == -1 and px < prev_px):
                pivots[-1] = (bar_i, px, pdir)
        else:
            pivots.append((bar_i, px, pdir))
    return pivots


def classify_xabcd(
    px: float,
    pa: float,
    pb: float,
    pc: float,
    pd_price: float,
    tolerance: float = 0.08,
) -> tuple[str, str, float] | None:
    """Classify 5-point swing into Stingray, Bat, Butterfly, Crab, Shark, or ABCD."""
    is_bull = pa > px and pb < pa and pc > pb and pd_price < pc
    is_bear = pa < px and pb > pa and pc < pb and pd_price > pc
    if not (is_bull or is_bear):
        return None
    direction = "BULLISH" if is_bull else "BEARISH"

    xa = abs(pa - px)
    ab = abs(pb - pa)
    bc = abs(pc - pb)
    cd = abs(pd_price - pc)
    xc = abs(pc - px)
    if min(xa, ab, bc, xc) <= 1e-9:
        return None

    b_xa = ab / xa
    c_ab = bc / ab
    d_xa = abs(pa - pd_price) / xa
    d_xc = abs(pc - pd_price) / xc
    cd_ab = cd / ab

    def in_range(v: float, lo_v: float, hi_v: float, tol: float = tolerance) -> bool:
        return (lo_v - tol) <= v <= (hi_v + tol)

    def near(v: float, target: float, tol: float = tolerance) -> bool:
        return abs(v - target) <= tol

    # Priority: Proprietary Stingray (B ~0.382-0.50 XA, D = 1.272 XA, Inval X)
    if in_range(b_xa, 0.36, 0.54) and in_range(c_ab, 0.35, 0.92) and near(d_xa, 1.272, tolerance * 1.4):
        return ("Stingray", direction, px)
    if in_range(b_xa, 0.36, 0.54) and in_range(c_ab, 0.35, 0.92) and near(d_xa, 0.886, tolerance):
        return ("Bat", direction, px)
    if near(b_xa, 0.786, tolerance) and in_range(c_ab, 0.382, 0.886) and near(d_xa, 1.272, tolerance * 1.4):
        inval = pd_price - xa * 0.14 if is_bull else pd_price + xa * 0.14
        return ("Butterfly", direction, inval)
    if in_range(b_xa, 0.382, 0.618) and near(d_xa, 1.618, tolerance * 1.8):
        inval = pd_price - xa * 0.16 if is_bull else pd_price + xa * 0.16
        return ("Crab", direction, inval)
    if in_range(d_xc, 0.886, 1.13, tolerance):
        inval = pc - xc * (1.13 + tolerance) if is_bull else pc + xc * (1.13 + tolerance)
        return ("Shark", direction, inval)
    if in_range(cd_ab, 0.95, 1.32, tolerance * 1.3):
        return ("ABCD", direction, pc)
    return None


def detect_harmonics(
    df: pd.DataFrame,
    pivot_len: int = 3,
    tolerance: float = 0.08,
) -> list[HarmonicSetup]:
    """Detect XABCD patterns across a DataFrame and compute PRZ, Plan B & C-D TPs."""
    pivots = extract_alternating_pivots(df, left_right=pivot_len)
    setups: list[HarmonicSetup] = []
    if len(pivots) < 5:
        return setups

    closes = df["close"].to_numpy()
    timestamps = df.index

    for idx in range(4, len(pivots)):
        _, px, _ = pivots[idx - 4]
        _, pa, _ = pivots[idx - 3]
        bar_b, pb, _ = pivots[idx - 2]
        _, pc, _ = pivots[idx - 1]
        bar_d, pd_price, _ = pivots[idx]

        classified = classify_xabcd(px, pa, pb, pc, pd_price, tolerance=tolerance)
        if classified is None:
            continue
        pat, direction, inval = classified
        xa = abs(pa - px)
        cd = abs(pc - pd_price)
        if cd <= 1e-9:
            continue

        is_bull = direction == "BULLISH"
        # Ensure realistic minimum stop distance (>= 25% of CD leg) and correct orientation beyond Point D
        min_sl_dist = max(cd * 0.25, xa * 0.14)
        if is_bull and (inval >= pd_price or (pd_price - inval) < min_sl_dist):
            inval = pd_price - min_sl_dist
        elif not is_bull and (inval <= pd_price or (inval - pd_price) < min_sl_dist):
            inval = pd_price + min_sl_dist

        prz_pad = cd * 0.05
        prz_low = pd_price - prz_pad
        prz_high = pd_price + prz_pad

        tp1 = pd_price + cd * 0.382 if is_bull else pd_price - cd * 0.382
        tp2 = pd_price + cd * 0.500 if is_bull else pd_price - cd * 0.500
        tp3 = pd_price + cd * 0.618 if is_bull else pd_price - cd * 0.618

        risk = abs(pd_price - inval)
        reward2 = abs(tp2 - pd_price)
        rr_tp2 = round(reward2 / max(risk, 1e-9), 2)

        # Check Point B trigger break before D completes
        seg_closes = closes[bar_b + 1 : min(bar_d + 1, len(closes))]
        point_b_triggered = bool(np.any(seg_closes < pb)) if is_bull else bool(np.any(seg_closes > pb))

        # Cascade Plan B for Bat pattern -> Stingray 1.272 XA
        plan_b = None
        if pat == "Bat":
            plan_b = pa - xa * 1.272 if is_bull else pa + xa * 1.272

        ts_str = str(timestamps[bar_d]) if bar_d < len(timestamps) else str(bar_d)
        setups.append(
            HarmonicSetup(
                pattern=pat,
                direction=direction,
                bar_d=bar_d,
                timestamp=ts_str,
                price_x=round(px, 6),
                price_a=round(pa, 6),
                price_b=round(pb, 6),
                price_c=round(pc, 6),
                price_d=round(pd_price, 6),
                prz_low=round(prz_low, 6),
                prz_high=round(prz_high, 6),
                invalidation_sl=round(inval, 6),
                tp1_382=round(tp1, 6),
                tp2_500=round(tp2, 6),
                tp3_618=round(tp3, 6),
                point_b_triggered=point_b_triggered,
                plan_b_stingray_d=round(plan_b, 6) if plan_b is not None else None,
                rr_tp2=rr_tp2,
                b_xa=round(abs(pb - pa) / xa, 3),
                d_xa=round(abs(pa - pd_price) / xa, 3),
            )
        )
    return setups
