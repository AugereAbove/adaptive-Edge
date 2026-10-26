# Paper starter

Working title: **AdaptiveEdge: Inspectable Selection of Local Language Models Under Resource Constraints**.

1. Introduction: local selection question, competing objectives, reproducible evidence. Present infrastructure and a transparent baseline without claiming an optimal router.
2. Related work: verified references in [related_work.md](related_work.md); distinguish full-model selection, learned routing, cascades, MoE and speculative generation.
3. System: typed contracts, measurement scope, append-only provenance, calibration-only medians, constraints, deterministic ties and Pareto utility. Use the README architecture diagram.
4. Experimental setup: populate model digests, hardware, runtime, settings, tasks, splits, seeds, repetitions and warm-ups from manifests/configurations.
5. Results: generated category tables/figures in [baseline report](../examples/results/baseline-real/report.md). Include unfavorable outcomes, prediction errors, unserved requests and unavailable memory. Do not average incompatible quality metrics.
6. Ablation: equal weights versus configured weights on frozen evidence; temporal/order confounding prevents causal attribution.
7. Validity: [threats_to_validity.md](threats_to_validity.md), particularly the small template-based sample, runtime effects and restricted code metric.
8. Conclusion: only conclusions supported by generated outputs. Larger external benchmarks are needed for claims of general routing benefit.

[Results template](results_template.md) maps artifacts into the manuscript. No synthetic numerical conclusions belong in the paper.
