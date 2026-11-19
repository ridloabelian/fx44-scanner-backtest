# FX44 Quant Benchmark Report (>=200 Trades per Variant)

Minimum target trades per strategy: `220` | Risk/trade: `1.0%` | Max Open Risk: `10.0%` with **Zero-Risk 50% BEP Slot Unlock**.

| Strategy Variant | Trades | Wins | BEP (`50%` Lock) | Losses | Win Rate | Avg RR | Profit Factor | Expectancy (R) | Total Return | Max DD | BEP Slots Unlocked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SMC Continuation (risk_limit) | 489 | 87 | 48 | 354 | 17.8% | 1:7.46 | 1.83 | +0.603R | +1359.1% | 36.6% | 136 |
| SMC Continuation (ltf_confirmation) | 397 | 179 | 107 | 111 | 45.1% | 1:6.85 | 11.06 | +2.799R | +34470.7% | 3.9% | 287 |
| Structure Break (wick_break) | 1105 | 167 | 249 | 689 | 15.1% | 1:3.00 | 0.73 | -0.170R | -86.3% | 86.7% | 416 |
| Structure Break (body_break) | 937 | 145 | 232 | 560 | 15.5% | 1:3.00 | 0.78 | -0.133R | -73.9% | 75.0% | 377 |
| Harmonic (Stingray) | 225 | 135 | 90 | 0 | 60.0% | 1:2.47 | 99.00 | +1.483R | +839.7% | 0.0% | 225 |
| Harmonic (Bat) | 220 | 220 | 0 | 0 | 100.0% | 1:2.47 | 99.00 | +2.472R | +871.8% | 0.0% | 220 |
| Harmonic (Shark) | 266 | 265 | 1 | 0 | 99.6% | 1:1.90 | 99.00 | +1.898R | +2243.6% | 0.0% | 239 |

## Key Analytical Takeaways
- **Zero-Risk 50% Trailing & BEP Unlock:** Moving stops to BEP once floating profit reaches 50% of target turns marginal reversals into risk-free exits and unlocks open-risk slots inside the 10% cap.
- **Risk Limit vs LTF Confirmation:** LTF Confirmation filters low-quality zone touches and improves realized R:R when combined with refined S&D stops, mirroring the Mancing Dollar FXReplay AUDCAD methodology.
- **Wick vs Body Close:** Consistency across ≥200 trades matters more than switching mid-stream; Body Close generally reduces false sweep triggers.
