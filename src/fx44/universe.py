"""44-instrument trading universe and WIB session filtering engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

WIB_TZ = timezone(timedelta(hours=7))


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    yahoo_ticker: str
    asset_class: str
    pip_size: float
    preferred_sessions: tuple[str, ...]


# 28 Major & Cross Forex Pairs + 16 Commodities / Crypto / Global Indices = 44 Instruments
INSTRUMENT_UNIVERSE: tuple[InstrumentSpec, ...] = (
    # 7 USD Majors
    InstrumentSpec("EURUSD", "EURUSD=X", "forex", 0.0001, ("london", "ny")),
    InstrumentSpec("GBPUSD", "GBPUSD=X", "forex", 0.0001, ("london", "ny")),
    InstrumentSpec("AUDUSD", "AUDUSD=X", "forex", 0.0001, ("asia", "london", "ny")),
    InstrumentSpec("NZDUSD", "NZDUSD=X", "forex", 0.0001, ("asia", "ny")),
    InstrumentSpec("USDJPY", "USDJPY=X", "forex", 0.01, ("asia", "london", "ny")),
    InstrumentSpec("USDCAD", "USDCAD=X", "forex", 0.0001, ("ny",)),
    InstrumentSpec("USDCHF", "USDCHF=X", "forex", 0.0001, ("london", "ny")),
    # 21 Forex Crosses
    InstrumentSpec("EURGBP", "EURGBP=X", "forex", 0.0001, ("london",)),
    InstrumentSpec("EURJPY", "EURJPY=X", "forex", 0.01, ("asia", "london")),
    InstrumentSpec("EURAUD", "EURAUD=X", "forex", 0.0001, ("asia", "london")),
    InstrumentSpec("EURNZD", "EURNZD=X", "forex", 0.0001, ("asia", "london")),
    InstrumentSpec("EURCAD", "EURCAD=X", "forex", 0.0001, ("london", "ny")),
    InstrumentSpec("EURCHF", "EURCHF=X", "forex", 0.0001, ("london",)),
    InstrumentSpec("GBPJPY", "GBPJPY=X", "forex", 0.01, ("asia", "london", "ny")),
    InstrumentSpec("GBPAUD", "GBPAUD=X", "forex", 0.0001, ("asia", "london")),
    InstrumentSpec("GBPNZD", "GBPNZD=X", "forex", 0.0001, ("asia", "london")),
    InstrumentSpec("GBPCAD", "GBPCAD=X", "forex", 0.0001, ("london", "ny")),
    InstrumentSpec("GBPCHF", "GBPCHF=X", "forex", 0.0001, ("london",)),
    InstrumentSpec("AUDJPY", "AUDJPY=X", "forex", 0.01, ("asia",)),
    InstrumentSpec("AUDCAD", "AUDCAD=X", "forex", 0.0001, ("asia", "ny")),
    InstrumentSpec("AUDCHF", "AUDCHF=X", "forex", 0.0001, ("asia", "london")),
    InstrumentSpec("AUDNZD", "AUDNZD=X", "forex", 0.0001, ("asia",)),
    InstrumentSpec("NZDJPY", "NZDJPY=X", "forex", 0.01, ("asia",)),
    InstrumentSpec("NZDCAD", "NZDCAD=X", "forex", 0.0001, ("asia", "ny")),
    InstrumentSpec("NZDCHF", "NZDCHF=X", "forex", 0.0001, ("asia", "london")),
    InstrumentSpec("CADJPY", "CADJPY=X", "forex", 0.01, ("asia", "ny")),
    InstrumentSpec("CADCHF", "CADCHF=X", "forex", 0.0001, ("london", "ny")),
    InstrumentSpec("CHFJPY", "CHFJPY=X", "forex", 0.01, ("asia", "london")),
    # Metals, Energy & Crypto (5)
    InstrumentSpec("XAUUSD", "GC=F", "commodity", 0.1, ("london", "ny")),
    InstrumentSpec("XAGUSD", "SI=F", "commodity", 0.01, ("london", "ny")),
    InstrumentSpec("USOIL", "CL=F", "commodity", 0.01, ("london", "ny")),
    InstrumentSpec("NATGAS", "NG=F", "commodity", 0.001, ("ny",)),
    InstrumentSpec("BTCUSD", "BTC-USD", "crypto", 1.0, ("asia", "london", "ny")),
    # Global Indices & Regional Proxy (11)
    InstrumentSpec("SPX500", "^GSPC", "index", 1.0, ("ny",)),
    InstrumentSpec("US100", "^NDX", "index", 1.0, ("ny",)),
    InstrumentSpec("US30", "^DJI", "index", 1.0, ("ny",)),
    InstrumentSpec("JP225", "^N225", "index", 1.0, ("asia",)),
    InstrumentSpec("GER40", "^GDAXI", "index", 1.0, ("london",)),
    InstrumentSpec("UK100", "^FTSE", "index", 1.0, ("london",)),
    InstrumentSpec("SPA35", "^IBEX", "index", 1.0, ("london",)),
    InstrumentSpec("HK50", "^HSI", "index", 1.0, ("asia",)),
    InstrumentSpec("AUS200", "^AXJO", "index", 1.0, ("asia",)),
    InstrumentSpec("KOSPI", "^KS11", "index", 1.0, ("asia",)),
    InstrumentSpec("IHSG", "^JKSE", "index", 1.0, ("asia",)),
)

UNIVERSE_BY_SYMBOL = {item.symbol: item for item in INSTRUMENT_UNIVERSE}


def get_wib_session(dt: datetime) -> str:
    """Return active WIB session ('asia', 'london', 'ny', or 'lockout')."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    wib_dt = dt.astimezone(WIB_TZ)
    hour = wib_dt.hour
    if 6 <= hour < 14:
        return "asia"
    if 14 <= hour < 15:
        return "transition"
    if 15 <= hour < 19:
        return "london"
    if 19 <= hour < 24:
        return "ny"
    return "lockout"


def filter_instruments_by_session(session: str | None = None) -> list[InstrumentSpec]:
    """Filter instruments preferred for a given session ('all' returns all 44)."""
    if not session or session.lower() == "all":
        return list(INSTRUMENT_UNIVERSE)
    sess = session.lower()
    if sess == "lockout":
        return []
    return [spec for spec in INSTRUMENT_UNIVERSE if sess in spec.preferred_sessions]
