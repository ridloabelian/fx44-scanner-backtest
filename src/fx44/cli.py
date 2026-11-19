"""Command-line interface for fx44-scanner-backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from fx44.backtest import backtest_harmonic_pattern, backtest_smc_strategy, run_200_trade_benchmark
from fx44.data import fetch_ohlcv, generate_synthetic_ohlcv
from fx44.scanner import format_webhook_message, scan_universe


def cmd_scan(args: argparse.Namespace) -> int:
    hits, errors = scan_universe(
        session=args.session,
        offline_demo=args.offline,
        interval=args.interval,
        period=args.period,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps({"hits": [h.to_dict() for h in hits], "errors": errors}, indent=2))
    else:
        print(format_webhook_message(hits, top_n=args.top))
        if errors:
            print(f"\n[WARN] {len(errors)} feed errors occurred.", file=sys.stderr)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    if args.offline or not args.symbol:
        df = generate_synthetic_ohlcv(n_bars=args.bars, seed=args.seed, symbol=args.symbol or "AUDCAD")
    else:
        df = fetch_ohlcv(args.symbol, interval=args.interval, period=args.period, csv_path=args.csv)

    if args.strategy == "smc":
        res = backtest_smc_strategy(df, entry_mode=args.entry_mode)
    else:
        res = backtest_harmonic_pattern(df, pattern_filter=None if args.pattern == "all" else args.pattern)
    print(json.dumps(res.to_dict(), indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    results, report_md = run_200_trade_benchmark(min_trades=args.min_trades, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")
    print(f"Benchmark complete ({len(results)} variants, min_trades>={args.min_trades}) -> {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fx44", description="44-Pair Scanner & >=200-Trade Quant Backtest Lab")
    sub = p.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Scan 44-instrument universe for PRZ, S&D & London Sweeps")
    p_scan.add_argument("--session", default="all", choices=["all", "asia", "london", "ny"])
    p_scan.add_argument("--offline", action="store_true", help="Use deterministic synthetic OHLCV instead of live yfinance")
    p_scan.add_argument("--interval", default="1h")
    p_scan.add_argument("--period", default="30d")
    p_scan.add_argument("--limit", type=int, default=None)
    p_scan.add_argument("--top", type=int, default=10)
    p_scan.add_argument("--json", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    p_bt = sub.add_parser("backtest", help="Run single-strategy event-driven backtest")
    p_bt.add_argument("--strategy", default="smc", choices=["smc", "harmonic"])
    p_bt.add_argument("--entry-mode", default="risk_limit", choices=["risk_limit", "ltf_confirmation"])
    p_bt.add_argument("--pattern", default="Stingray", choices=["all", "Stingray", "Bat", "Butterfly", "Crab", "Shark"])
    p_bt.add_argument("--symbol", default="AUDCAD")
    p_bt.add_argument("--csv", default=None)
    p_bt.add_argument("--interval", default="1h")
    p_bt.add_argument("--period", default="60d")
    p_bt.add_argument("--bars", type=int, default=6000)
    p_bt.add_argument("--seed", type=int, default=42)
    p_bt.add_argument("--offline", action="store_true")
    p_bt.set_defaults(func=cmd_backtest)

    p_bm = sub.add_parser("benchmark", help="Run >=200-trade comparative benchmark and export Markdown report")
    p_bm.add_argument("--min-trades", type=int, default=220)
    p_bm.add_argument("--seed", type=int, default=42)
    p_bm.add_argument("--out", default="reports/benchmark_200_trades.md")
    p_bm.set_defaults(func=cmd_benchmark)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
