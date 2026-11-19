from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fx44.backtest import backtest_break_mode, backtest_smc_strategy, run_200_trade_benchmark  # noqa: E402
from fx44.cli import main as cli_main  # noqa: E402
from fx44.data import generate_synthetic_ohlcv  # noqa: E402
from fx44.patterns.harmonics import classify_xabcd, detect_harmonics  # noqa: E402
from fx44.patterns.smc import detect_market_structure, find_fresh_snd_zones  # noqa: E402
from fx44.scanner import format_webhook_message, scan_universe  # noqa: E402
from fx44.universe import INSTRUMENT_UNIVERSE, filter_instruments_by_session, get_wib_session  # noqa: E402


def test_universe_has_44_instruments_and_session_filters():
    assert len(INSTRUMENT_UNIVERSE) == 44
    assert len(filter_instruments_by_session("all")) == 44
    assert any(s.symbol == "AUDJPY" for s in filter_instruments_by_session("asia"))
    assert any(s.symbol == "EURUSD" for s in filter_instruments_by_session("london"))
    assert filter_instruments_by_session("lockout") == []

    # 15:00 WIB = 08:00 UTC -> London
    assert get_wib_session(datetime(2026, 5, 1, 8, 30, tzinfo=timezone.utc)) == "london"
    # 01:00 WIB = 18:00 UTC -> Lockout after 00:00 WIB
    assert get_wib_session(datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc)) == "lockout"


def test_stingray_bat_and_shark_classification():
    # Bullish Stingray: X=100, A=200 (XA=100), B=155 (AB=45 => 0.45), C=185 (BC=30), D=72.8 (AD=127.2 => 1.272)
    res_sting = classify_xabcd(100.0, 200.0, 155.0, 185.0, 72.8)
    assert res_sting is not None and res_sting[0] == "Stingray" and res_sting[1] == "BULLISH"

    # Bearish Bat: X=200, A=100 (XA=100), B=144 (0.44), C=115, D=188.6 (0.886)
    res_bat = classify_xabcd(200.0, 100.0, 144.0, 115.0, 188.6)
    assert res_bat is not None and res_bat[0] == "Bat" and res_bat[1] == "BEARISH"

    df = generate_synthetic_ohlcv(n_bars=1200, seed=42)
    harmonics = detect_harmonics(df, pivot_len=2, tolerance=0.09)
    assert len(harmonics) >= 5
    assert any(h.pattern == "Stingray" for h in harmonics)


def test_smc_structure_and_zones():
    df = generate_synthetic_ohlcv(n_bars=1200, seed=7)
    body_events = detect_market_structure(df, break_mode="body")
    wick_events = detect_market_structure(df, break_mode="wick")
    assert len(body_events) > 10
    assert len(wick_events) >= len(body_events)
    zones = find_fresh_snd_zones(df)
    assert len(zones) >= 5


def test_scanner_offline_and_webhook_formatting():
    hits, errors = scan_universe(session="london", offline_demo=True, limit=8)
    assert errors == []
    assert len(hits) > 0
    msg = format_webhook_message(hits)
    assert "FX44 Scanner Alert" in msg


def test_200_trade_benchmark_exceeds_minimum(tmp_path):
    results, report_md = run_200_trade_benchmark(min_trades=205, seed=42)
    assert len(results) == 7
    for r in results:
        assert r.trades >= 205, f"{r.strategy_name} had only {r.trades} trades"
    assert "| Stingray |" in report_md or "Harmonic (Stingray)" in report_md

    out_file = tmp_path / "bench.md"
    rc = cli_main(["benchmark", "--min-trades", "205", "--out", str(out_file)])
    assert rc == 0
    assert out_file.exists() and out_file.stat().st_size > 200
