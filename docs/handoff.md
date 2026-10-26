# AdaptiveEdge 0.1.0 release-candidate handoff

> Historical source handoff: This document preserves the initial release handoff from e3712b6. Statements about untracked files, absent remotes, local verification, coverage and CI describe that source snapshot, not the current adaptive-Edge checkout. This staged publication excludes the source test suite and CI; transfer validation is limited to source comparison and static review.

## Delivered artifact

The benchmark, selection, storage and reporting paths remain compatible with the initial implementation. New work adds explicit data splits, a transactional schema migration, a capability-restricted executable code metric, task-level bootstrap uncertainty, prediction errors, descriptive policy effects, structured diagnostics, publication plots and a measured baseline with an equal-weight ablation.

The complete final verification record is `outputs/release/verification.json`, with individual command logs in the same directory. Coverage reports, dependency scan output, clean-install checks and live smoke evidence remain local under ignored outputs/release. The public-facing real artifact is [examples/results/baseline-real/report.md](../examples/results/baseline-real/report.md), with hashes, configurations, model digests, tables and selected SVG figures. Full raw output text, databases and PNG/SVG plot families remain in ignored outputs/baseline-real. No model weights or large databases belong in Git.

## Reproduce the workflow

Create a Python environment and install `python -m pip install -e ".[dev,ollama,plotting]"`. Run doctor and the functional benchmark described in README. For the real baseline, explicitly install qwen2.5:0.5b, smollm2:1.7b and qwen3:4b in Ollama, then run `python experiments/baseline_real.py` and `python scripts/curate_results.py`. New experiment IDs are created; original observations are never overwritten. The curator deliberately omits unrelated installed-model inventory and generated response text.

Calibration/validation/evaluation/functional/inference splits are visible in records, manifests, SQL and CSV. Predictions accept only compatible successful calibration observations without a policy. Policy evaluation freezes a calibration experiment and rejects exact prompt/ID overlap. Shared templates and model-training contamination still require methodological judgment. Legacy schema-1 data receives calibration defaults on migration; this does not retroactively establish prospective split declaration.

`adaptiveedge benchmark reproduce ID` restores benchmark tasks/settings, creates a new experiment and links reproduction_of provenance. Mutable tags/runtime/hardware can change; the command warns explicitly. Evaluation is rerun from its frozen training manifest, configuration, constraints and weights rather than through the benchmark-only reproduce command.

## Empirical state

The complete real study and ablation are recorded in the generated report; mock observations are excluded. All six policies were requested. Lowest-memory abstained because model peak memory was unavailable. Adaptive selection lost reasoning quality relative to the calibration-selected highest-quality policy; documentation retains that unfavorable finding. No general superiority, model-memory saving, significance or population-generalization claim is made.

Live verification covers discovery, validation, generation, usage, warm/cold behavior, timeouts, failure handling, persisted adaptive inference, exports and CLI reproduction. Default tests remain offline; live tests require ADAPTIVEEDGE_REAL_MODEL and explicit opt-in markers. Postrun process RSS/resident allocation diagnostics are retained separately from unsupported model peak RAM/VRAM.

## Verification and repository state

The final command logs establish lint, formatting, strict typing, offline tests and branch coverage, pre-commit, source/repository/claim audits, dependency scanning, wheel/sdist builds, clean package installation, CLI smoke and live checks. The prior 94.28% combined coverage is the enforced floor; selection, constraints, predictor and Pareto logic retain complete local coverage. The remote Windows/Linux Python 3.11–3.13 matrix is configured but has not run because no remote exists. Local verification uses Windows and Python 3.11.

Git remains on main with no commits, remote, push, public repository, package publication or release created. Project files are intentionally untracked pending the owner's first commit. MIT licensing is unchanged. CITATION.cff preserves the existing collective contributor credit, version 0.1.0 and no invented owner identity, repository URL, DOI or public release date.

## Remaining boundaries

Research limits: tiny authored in-template samples, one personal desktop with background activity, fixed order/prompt caching, quantization/family confounding, limited task bootstrap intervals, selected-candidate prediction errors and mutable artifacts. Read [methodology](methodology.md) and [threats](../research/threats_to_validity.md).

Engineering scope: model peak RAM/VRAM is unsupported; the code evaluator deliberately supports expression fixtures rather than general Python; Transformers/llama.cpp execution adapters are outside this release; exact digest-enforced replication is not implemented. Missing evidence is explicit and does not become a zero measurement.

The owner has not supplied personal author details or a private security endpoint. Retain collective credit or supply the intended identity, configure private vulnerability reporting on the actual host, and add the real repository URL only when it exists. No private reporting feature is claimed to be enabled.

After final local checks pass, the next action is to create the GitHub remote, make the initial commit and push. Inspect the first remote CI run before tagging a public release. No major implementation pass is required to publish this clearly scoped research artifact.
