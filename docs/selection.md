# Selection mathematics and evidence

Given candidates, a task context and a hardware snapshot, the predictor considers only successful benchmark observations with the same complete model configuration hash, hardware fingerprint, task category, dataset name/version, effective generation settings and warm/cold measurement mode. It excludes adaptive/evaluation rows and requires simulated status to match the mock backend. If a primary metric is requested, rows must contain that metric. Each estimate is the median of available observations. Missing values remain missing; each candidate exposes its sample count.

The hardware fingerprint includes OS release, CPU description, logical cores, total RAM, GPU names and accelerator runtime. Available memory and timestamp are excluded. Driver changes, background load and mutable model tags are not automatically modeled; historical evidence must be regenerated when these change. Records from different servers must use distinct candidate configuration identities. Remote Ollama attribution requires a future server hardware profiler.

By default, input length is recorded but not used to extrapolate. `HistoricalMedianPredictor(input_length_ratio=r)` can restrict evidence to prompts satisfying `max(lengths)/max(1,min(lengths)) <= r`, with `r >= 1`. This is a character-length filter, not a token or latency model. No extrapolation is implied. Category-only API requests inherit the most recently benchmarked dataset/metric context; pass an explicit `TaskDefinition` when multiple datasets exist.

## Hard constraints

Maximum constraints compare prediction `<= limit`; minimum constraints compare prediction `>= limit`. Unknown required values are infeasible, including unknown model size. Empty allowlists allow nothing. Device preference requires an explicitly matching configuration and is treated as a hard compatibility requirement. Host available RAM is checked when both host availability and model memory are known. Model memory cannot be substituted with client RSS. Predictions describe historical behavior, so a feasible prediction is not a runtime guarantee.

No compatible evidence is infeasible for every policy. If no scored feasible candidate remains, `selected` is null and inference is not attempted. CLI exit code is 3. No infeasible fallback executes automatically.

## Policies

| Policy | Utility / ordering |
|---|---|
| smallest | Negative verified model artifact size in GB |
| largest | Verified model artifact size in GB |
| fastest | Negative predicted request latency in ms |
| highest_quality | Predicted primary task metric |
| lowest_memory | Negative predicted model-runtime peak RAM in GB |
| adaptive | Weighted normalized multi-objective utility on the feasible Pareto frontier |

Largest is a size baseline; it is not assumed to be the strongest model. Highest-quality is category-specific and empirical. Size and parameter count are distinct properties. Missing policy metrics prevent that candidate from being scored.

For adaptive selection, active objectives are positive-weight objectives observed for **every feasible candidate**. An objective missing for one feasible candidate is dropped for all, and the remaining weights are renormalized. This avoids comparing utilities on different scales. Hard constraints still require their measurements independently. The explanation lists active objectives; inspect it when evidence is incomplete.

A dominates B if A is no worse in every active objective and strictly better in at least one. Quality and throughput are maximized; latency, RAM, VRAM and model size are minimized. Missing objective values make a pair incomparable. Equal points are both nondominated. Report frontiers support selectable quality/latency/RAM/VRAM/throughput objectives within identical task/hardware/mode/simulation/split/policy groups. Publication projections show descriptive group means separately.

For objective j with feasible-set minimum a and maximum b, benefit normalization is `(x-a)/(b-a)` and cost normalization is `(b-x)/(b-a)`. Constant objectives receive 1 for all candidates. Utility is `sum(w_j * normalized_j) / sum(w_j)` over active objectives. Normalization includes the entire feasible set before dominance removal. Zero active objectives gives no selection. Ties sort by label/model name then canonical configuration hash, ascending; source input order never decides the winner.

The default weights are quality 0.6, latency 0.3 and RAM 0.1; all others are zero. These are explicit demonstration preferences, not learned or experimentally optimal constants. Configure them through `Weights` or `--weights-file`. Run sensitivity analyses across weights and constraints before reporting a conclusion.

This median baseline estimates performance conditional on successful runs; failures are separately retained and reported. It does not estimate success probability, uncertainty, failure risk, token-aware context fit, energy or scheduling contention. A later predictor should model these before deployment with strict service guarantees.

Only calibration-split observations without a policy can train the predictor. Validation/evaluation/functional/inference rows cannot silently enter history. The release keeps exact compatible-context matching rather than borrowing global or cross-hardware evidence; unknown predictions cause abstention. Selected-candidate errors on held-out observations are reported separately from the historical medians.
