# From artifact to paper

## Experimental setup

Use examples/results/baseline-real/manifest.json for digests, configurations, software and hardware. Archived calibration/evaluation configurations contain prompts, tests, generation settings and split identities. The authored suite is a small controlled testbed, not a representative benchmark. Distinguish independent tasks, repetitions and policy requests.

## Results

Use aggregate-results.csv for task-mean statistics and percentile bootstrap intervals; effects.csv for category-specific adaptive versus calibration-selected highest-quality comparisons; prediction-errors.csv for selected-model errors; research.json for abstentions, request denominators and unavailable measurements. Regenerate report.md rather than editing numbers by hand. Quality retention is a comparable category mean ratio, not a pooled ratio across unrelated metrics.

Map quality–latency/Pareto figures to tradeoffs, policy bars to baselines and prediction plots to calibration limits. Model RAM/VRAM figures remain unavailable; do not substitute client RSS or resident allocation.

## Ablation study

Compare ablation-research.json with research.json: equal quality/latency weights versus declared original weights, using identical frozen calibration, prompts, seeds and repetitions. The ablation runs later, so timing is confounded by order/cache/thermal state. Choices were frozen before evaluation; no validation set was used because there was no tuning.

## Threats to validity

Use [methodology](../docs/methodology.md) and [threats](threats_to_validity.md). This evidence cannot establish general superiority, population uncertainty, energy savings or model-memory savings.
