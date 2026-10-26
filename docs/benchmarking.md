# Running benchmarks

Install from the repository with `python -m pip install -e ".[dev,ollama,plotting]"`. Tests never download models. Each CLI command accepts its database path through `--db`; commands also accept `--json` where useful. `--debug` and `--verbose` are global options before the command.

The SQLite database parent directory must already exist. Export/report directories are created automatically. `python scripts/smoke.py` runs the complete functional CLI verification and retains its local evidence under ignored `outputs/smoke`.

```text
adaptiveedge doctor --db adaptiveedge.db --json
adaptiveedge hardware --json
adaptiveedge models --backend mock --json
adaptiveedge benchmark run configs/benchmarks/basic.yaml --db adaptiveedge.db --json
adaptiveedge benchmark list --db adaptiveedge.db --json
adaptiveedge benchmark show EXPERIMENT_ID --db adaptiveedge.db --json
adaptiveedge benchmark reproduce EXPERIMENT_ID --db adaptiveedge.db --json
```

The built-in suite has five controlled examples covering classification, extraction, reasoning, summarization and code. Three repetitions across two mock models give 30 raw observations. Warm-ups are excluded. All model speeds, sizes, token counts and memory properties of the mock backend are synthetic fixtures. Client RSS is still measured and labeled separately. This is a functional suite, not an academic dataset or performance result.

## Ollama

Install and start Ollama separately. Explicitly install models using Ollama's own tools; AdaptiveEdge never pulls them. Discover exact installed tags:

```text
adaptiveedge models --backend ollama --json
```

Copy `configs/benchmarks/ollama.template.yaml` to a new file. Replace both `REPLACE_WITH_INSTALLED_*_MODEL_TAG` strings with exact installed tags. Keep the resulting config alongside your experiment. Supply verified artifact `size_gb` values to enable size policies; omit rather than guess. Quantization and parameter counts are descriptive metadata supplied by the experimenter and are not inferred from a label.

```text
adaptiveedge benchmark run configs/benchmarks/ollama.local.yaml --db ollama.db --json
adaptiveedge select --task reasoning --db ollama.db --max-latency-ms 800 --json
adaptiveedge run --task reasoning --input-file configs/examples/input.txt --db ollama.db --json
```

The adapter follows the [Ollama generate API](https://docs.ollama.com/api/generate). It measures streamed request wall time and first nonempty output chunk, keeps thinking text out of the final answer, and records server loading/generation durations and token usage when available. Throughput is completion tokens divided by server generation seconds. TTFT refers to first answer text, not hidden thinking tokens. Missing usage remains unavailable. The availability manifest includes server version and model inventory/digests from `/api/tags`.

Set `OLLAMA_HOST` only when needed. The default is local HTTP. Remote service hardware is not detected by the client, so remote runs require a separately controlled environment and cannot currently support trustworthy hardware-conditioned selection. The backend uses server-managed device placement (`device: auto`). Reserved generation settings and seed cannot be overridden by runtime extras.

## Custom tasks and configs

YAML and TOML are supported. Use `suite: custom` and an explicit `tasks` sequence. Each task specifies a unique ID, category, prompt, optional reference, metric identifiers, dataset name and version. References can be absent; then quality is unmeasured, never zero. Effective generation settings come from the task when supplied, otherwise the model. Model configurations, full prompts and outputs are stored locally.

Experiment settings include `repetitions`, `warmup_runs`, `seed`, `timeout_seconds`, and `mode` (`warm` or `cold`). Seed is base seed plus repetition modulo 2^31, shared across configurations. Cold mode unloads the selected Ollama model before each request and requests unloading afterward; it skips warm-ups. Warm mode warms each configuration before measurement. OS filesystem caches are not flushed in either mode.

## Selection and evaluation

```text
adaptiveedge select --task reasoning --max-latency-ms 30 --db adaptiveedge.db --json
adaptiveedge select --task reasoning --constraints-file configs/examples/constraints.json --weights-file configs/examples/weights.json --db adaptiveedge.db --json
adaptiveedge run --task reasoning --input-file configs/examples/input.txt --max-latency-ms 30 --db adaptiveedge.db --json
adaptiveedge evaluate TRAINING_ID configs/benchmarks/evaluation-smoke.yaml --allow-overlap --db adaptiveedge.db --json
```

The last command is explicitly an overlapping functional smoke test. For research, supply distinct held-out task IDs and prompts and omit `--allow-overlap`. The evaluator freezes the specified training experiment, evaluates all six policies, records actual constraint compliance, and logs abstentions when no candidate is feasible. Warm-up runs occur immediately before each policy request. All evaluation rows are excluded from prediction history. Candidate configuration hashes must match those benchmarked during training.

Exit codes: 0 success; 1 recorded execution failure(s); 2 invalid input, configuration or infrastructure failure; 3 no feasible selection; 130 interruption. An all-abstention evaluation still completes successfully and records abstentions in its manifest; inspect that count before interpreting a report. Doctor reports optional runtime absence without failing the installation check.
