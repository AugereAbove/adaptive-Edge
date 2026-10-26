# Preliminary real baseline

Generated from immutable SQLite observations by scripts/curate_results.py. No simulated observations are included.

Calibration: 108 observations across 12 tasks and 3 models. Held-out: 12 tasks, 3 repetitions, 180 served requests. Ablation: 36 observations.

Ollama 0.32.6; Windows; RAM 33.706 GB. Hardware snapshots, model digests, settings, timestamps and frozen training IDs are in manifest.json. GPU identity is a supplemental postrun snapshot.

Models: qwen2.5:0.5b, qwen3:4b, smollm2:1.7b.

## Adaptive versus highest-quality policy

Highest-quality means the policy selected from calibration; it is not a held-out oracle. Each row uses its own task metric. Negative latency reduction means slower. No pooled quality score is calculated.

| Category | Adaptive quality | Reference quality | Quality retention | Latency reduction | Latency MAE (ms) | Quality MAE |
|---|---:|---:|---:|---:|---:|---:|
| classification | 0.667 | 0.667 | 100.00% | -1.10% | 6.73 | 0.333 |
| code | 0.889 | 0.889 | 100.00% | -0.11% | 44.11 | 0.556 |
| extraction | 1.000 | 1.000 | 100.00% | 8.58% | 13.95 | 0.000 |
| reasoning | 0.667 | 1.000 | 66.67% | 18.47% | 8.50 | 0.333 |

## Request coverage

| Policy | Requested | Served | Abstentions | Failures |
|---|---:|---:|---:|---:|
| smallest | 36 | 36 | 0 | 0 |
| largest | 36 | 36 | 0 | 0 |
| fastest | 36 | 36 | 0 | 0 |
| highest_quality | 36 | 36 | 0 | 0 |
| lowest_memory | 36 | 0 | 36 | 0 |
| adaptive | 36 | 36 | 0 | 0 |

Model peak RAM/VRAM is unavailable; client RSS and resident allocation are not substituted. No model-memory reduction is claimed. Served-request constraint satisfaction is in research.json and is conditional on service; abstentions are separate.

## Equal-weight ablation

Same frozen calibration, held-out tasks, seeds and repetitions. Declared adaptive quality/latency weights are compared with equal weights; unavailable memory objectives are inactive. The ablation runs later, so latency changes are descriptive, not causal.

| Category | Original quality | Equal-weight quality | Original latency (ms) | Equal-weight latency (ms) |
|---|---:|---:|---:|---:|
| classification | 0.667 | 0.667 | 381.85 | 397.68 |
| code | 0.889 | 0.889 | 405.61 | 405.82 |
| extraction | 1.000 | 1.000 | 345.49 | 348.77 |
| reasoning | 0.667 | 0.667 | 347.87 | 353.36 |

## Uncertainty and interpretation

Tables use task means over successful repetitions. Percentile bootstrap: 2000 resamples, seed 42, empirical order statistics 50 and 1950 (one-based), 95% interval for the mean. Fewer than three task means suppresses intervals; fewer than ten is marked limited. Each category here has only three independent task instances. Intervals describe this tiny sample and are not population guarantees. No significance tests were performed.

Calibration and evaluation have disjoint prompts/IDs but share task templates. This is a preliminary in-template baseline, not evidence of general routing superiority. The reasoning result illustrates a quality loss; calibration medians can hide individual task failures. Background work, fixed order and prompt caching limit latency interpretation. Code quality is restricted-expression test success, not full Python pass@1.

See ../../../docs/methodology.md and ../../../research/threats_to_validity.md. Full raw text, SQLite and high-resolution SVG/PNG figures remain in ignored outputs/baseline-real/. Curated CSVs omit generated text; configurations preserve authored fixtures. Checksums detect accidental artifact changes, not scientific validity.
