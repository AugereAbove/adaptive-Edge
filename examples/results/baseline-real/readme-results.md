## Adaptive versus highest-quality policy

Highest-quality means the policy selected from calibration; it is not a held-out oracle. Each row uses its own task metric. Negative latency reduction means slower. No pooled quality score is calculated.

| Category | Adaptive quality | Reference quality | Quality retention | Latency reduction | Latency MAE (ms) | Quality MAE |
|---|---:|---:|---:|---:|---:|---:|
| classification | 0.667 | 0.667 | 100.00% | -1.10% | 6.73 | 0.333 |
| code | 0.889 | 0.889 | 100.00% | -0.11% | 44.11 | 0.556 |
| extraction | 1.000 | 1.000 | 100.00% | 8.58% | 13.95 | 0.000 |
| reasoning | 0.667 | 1.000 | 66.67% | 18.47% | 8.50 | 0.333 |
