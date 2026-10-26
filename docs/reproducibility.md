# Reproducibility

Every experiment records a canonical configuration hash and a unique identifier (`hash-prefix` plus random UUID). Each observation has an independent run UUID, repetition index, seed, UTC timestamp, full model configuration, task/reference/metrics/dataset version, hardware snapshot, software version and inference measurements. The experiment manifest records runtime availability/version and Ollama model inventory including digests where the server reports them. Resolved functional tasks are archived so a later suite change does not silently change a benchmark rerun. Source prompts and generated text are local plaintext data.

```text
adaptiveedge benchmark show EXPERIMENT_ID --json
adaptiveedge benchmark reproduce EXPERIMENT_ID --json
adaptiveedge report EXPERIMENT_ID --output outputs/reproduction
```

Reproduce validates that model tags are installed, restores archived tasks and creates a new experiment. It never modifies the source experiment. The new configuration can have a different hash because resolved built-in tasks become an explicit custom suite. A provenance event links the original ID. The command reports failures and warns that hardware, runtime and mutable model tags may differ. It does not automatically install an old artifact or enforce digest equality. Before treating a rerun as a replication, compare inventory/digests, versions and hardware manifests. Export and pin model artifacts separately where licensing permits.

Policy evaluation freezes a specific training experiment and records its run IDs, weights and constraints. Adaptive requests store their structured selection and constraints. These rows never become benchmark training data. `benchmark reproduce` accepts benchmark experiments only; rerun policy evaluation with its original frozen training ID/config/weights/constraints. Save commands and configuration files with the research artifact.

For a reproducible environment, archive `python -m pip freeze`, the wheel/sdist, source commit, installed model manifest/digests, runtime build and driver versions, OS/hardware description and power settings alongside reports. Dependency ranges permit compatible upgrades; they are not a lockfile. Resolve and archive an environment per experiment, and rerun validation after upgrades. A local development dependency snapshot is not a cross-platform environment lock.

SQLite schema version is stored in `PRAGMA user_version`. Unknown versions fail rather than being modified. Commits are per observation and atomic across result tables. Raw records reject update/delete through database triggers. SQLite files should be backed up after closing the connection; a copy of an actively written database is not a backup protocol. Default use is one sequential writer, not distributed workers.

Reports include raw JSON/CSV, structured summary JSON, Markdown and manifest JSON. JSON preserves the full nested records. CSV uses empty cells for missing numeric measurements and status columns to distinguish causes; JSON is authoritative when CSV is ambiguous. Large model/dataset artifacts do not belong in the database or Git. Export destination paths are operator-controlled. Review all exported prompts and hardware information before publication.

## Release-candidate baseline

`python experiments/baseline_real.py` writes separate calibration/evaluation configurations, frozen model metadata, SQLite observations, reports and an equal-weight ablation under ignored outputs/baseline-real. Models must already be installed. `python scripts/curate_results.py` regenerates a small examples/results/baseline-real package and SHA-256 file checksums. It omits unrelated installed-model inventory, local paths, databases and generated text. The full private database retains raw outputs. Curated CSV run IDs trace back to immutable local observations.

Splits are explicit in configurations, manifests, records and CSV. A version 1 database migrates transactionally to schema 2 with legacy calibration defaults; it cannot retroactively establish prospective split declaration. Validate task-family overlap separately from exact prompt/ID checks. A different hardware fingerprint produces no compatible prediction rather than silently reusing another machine.

The original baseline hardware profile predates optional NVML installation; a postrun snapshot identifies the GPU separately. Background verification work was active. This and fixed order limit timing interpretation. Current device/free-memory snapshots must not overwrite the original hardware record.

Benchmark reproduction restores archived tasks and settings under a new experiment ID and records reproduction_of provenance. It warns about mutable tags/runtime/hardware. Policy evaluation is rerun with its recorded calibration ID, configuration, weights and constraints; the benchmark-only reproduce command rejects evaluation/inference records explicitly.

Curated artifact checksums use SHA-256 with CRLF normalized to LF. `.gitattributes` enforces LF text checkouts so Windows and Linux validate the same published artifact bytes. This normalization changes no experimental values.
