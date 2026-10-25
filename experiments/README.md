# Experiment entry points

The reusable driver benchmarks a calibration configuration, freezes that experiment's evidence, evaluates all policies on a second configuration and produces separate training/evaluation reports:

```text
python experiments/run_experiment.py configs/benchmarks/basic.yaml configs/benchmarks/evaluation-smoke.yaml --smoke --db adaptiveedge.db --output outputs/experiment
```

This command uses simulated mock properties and overlapping functional tasks. It verifies the experiment pipeline only. For research, supply real installed model configurations and explicit disjoint held-out custom tasks, remove `--smoke`, archive configs and environment manifests, and use a justified evaluation metric. Task dataset names/versions and candidate hashes must match the calibration context.

Use the CLI `evaluate` command to set constraints/weights through JSON files. Reports retain actual compliance, failures, unknown compliance and abstentions. Current reports are descriptive statistics; perform declared paired task-level uncertainty analysis in a versioned study script before claiming research findings.

## Real baseline

Install the small tags qwen2.5:0.5b, smollm2:1.7b and qwen3:4b explicitly in Ollama. Run `python experiments/baseline_real.py`, then `python scripts/curate_results.py`. The script declares three repetitions on twelve calibration and twelve held-out tasks across classification, extraction, reasoning and code, then runs all six policies and one equal-weight adaptive ablation. Lowest-memory abstains if model peaks are unavailable. Do not interpret unserved requests as successful inference. No validation tuning is performed.

For optional live checks set ADAPTIVEEDGE_REAL_MODEL to an installed tag and run `pytest -m real_backend --no-cov`; otherwise the marked test skips. `python scripts/verify_live.py` checks persisted adaptive execution, CLI reproduction and exports after the baseline exists. Default CI remains offline.
