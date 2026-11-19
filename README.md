# FX44 Scanner & Quant Backtest Lab

Multi-session scanner across **44 trading instruments** (28 FX pairs + Gold/Silver/Oil/NatGas/BTC + 11 Global Indices) paired with an event-driven backtest engine built to validate rules over **>= 200–300 trades** instead of cherry-picked screenshots.

---

## Core Capabilities

1. **44-Instrument WIB Session Scanner (`fx44 scan`)**
   - Filters active instruments across **Asia (`06:00–14:00 WIB`)**, **London Sweep (`15:00–19:00 WIB`)**, **New York (`19:00–24:00 WIB`)**, and blocks after `00:00 WIB`.
   - Scans for **Harmonic & Stingray PRZ**, **Fresh S&D + >=70% Imbalance Fill**, **Golden Ratio (`0.50–0.618`) + BB(20,2) + SL `0.786`**, and **London Open Sweeps of Asia High/Low**.
2. **Event-Driven Simulator with Zero-Risk 50% BEP & BEP Slot Unlock (`fx44 backtest`)**
   - Compares **SMC Risk Entry (Limit)** vs **LTF Confirmation Entry**.
   - Compares **Wick Break** vs **Body Close** BOS validation.
   - Benchmarks **Stingray `1.272`** vs **Bat `0.886`** vs **Shark `0.886–1.13 XC`** using Fibo C–D targets (`0.382 / 0.50 / 0.618`).
   - Enforces `1%` risk per trade, `10%` max concurrent open risk, moves SL to BEP at `50%` floating profit, and unlocks additional slots whenever an existing trade reaches BEP (`0%` active risk).
3. **Deterministic >=200-Trade Benchmark (`fx44 benchmark`)**
   - Includes a fractal regime-switching OHLCV generator so 200+ trade ablation runs complete offline in seconds and export reproducible Markdown reports (see [`reports/benchmark_200_trades.md`](./reports/benchmark_200_trades.md)).

---

## Quickstart

```bash
pip install -e ".[dev]"

# Run offline demo scan across London pairs
fx44 scan --session london --offline

# Run live Yahoo Finance scan on New York pairs
fx44 scan --session ny --interval 1h --period 30d

# Run >=220-trade benchmark comparing Risk vs Confirmation, Wick vs Body, Stingray vs Bat vs Shark
fx44 benchmark --min-trades 220 --out reports/benchmark_200_trades.md

# Run tests
pytest tests -v
```
