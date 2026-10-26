# Measurement methodology

AdaptiveEdge asks whether local configuration selection can reduce request cost while retaining task quality. It supplies a transparent baseline and evidence, not an established claim of general benefit.

## Execution and scopes

Requests are sequential. Calibration warms the first task once per model; policy evaluation warms the actual task before every measured request. Warm-ups are excluded. Warm mode does not guarantee zero loading: inspect load_ms for eviction. Cold mode requests selected-model unload before and after generation, not filesystem/GPU-driver cache eviction. Unload preparation is outside measured latency; loading during generation is inside it.

Request wall time is measured with a monotonic counter around streamed inference, including HTTP transit/decoding. It excludes validation, hardware discovery, warm-up and quality scoring. Server generation time, load time and TTFT have distinct scopes. TTFT is the first nonempty answer chunk, potentially after hidden reasoning; chunks can contain multiple tokens. Generation throughput is backend completion tokens divided by backend generation seconds, not request throughput. Missing/zero duration cannot yield a finite throughput claim. Model artifact size is not RAM, parameters, FLOPs or energy.

Client peak RAM is sampled process RSS at 10 ms plus initial/final reads. It excludes Ollama server/runner memory and can miss short peaks. Model peak RAM/VRAM remains unsupported. Optional NVML describes device-wide capacity/free VRAM, not model peaks. Live verification records postrun Ollama process RSS and server-reported resident allocation separately; neither is inference peak or attributable memory reduction. Never substitute client RSS or resident allocation into model-memory constraints. Unknown required measurements fail constraints. Lowest-memory therefore abstains on the real baseline. Other GPU vendors degrade gracefully to explicit unavailable evidence.

## Splits and frozen evidence

ExperimentSettings.split and BenchmarkResult.split distinguish calibration, validation, evaluation, functional and inference. The database stores split columns and rejects a result that disagrees with its experiment. Policy evaluation requires a calibration experiment, freezes its run IDs and uses only calibration observations. The predictor also rejects every other split and policy/inference rows. Evaluation is automatically marked evaluation; explicit overlap mode is functional and is not research evidence. Validation can store tuning observations but is never silently folded into calibration.

Prompt/ID overlap is rejected unless functional smoke mode was requested. This cannot detect semantic paraphrases or pretraining contamination. The baseline has disjoint authored prompts/IDs with shared task templates: describe it as in-template held-out evaluation. No weights were tuned, so no validation subset was used. The equal-weight ablation was declared before reading held-out outcomes and reuses frozen calibration. It runs later, so timing effects are not causally attributable to weights.

Schema version 2 transactionally adds split columns to version 1 without rewriting raw JSON. Legacy observations default to calibration, matching the original benchmark role; historical policy rows still cannot train prediction. Legacy databases lack prospectively declared splits and must not be relabeled as a preregistered study. Inserts are transactional and UPDATE/DELETE triggers protect raw tables. Cancellation preserves completed observations and records interruption. Unknown schema versions fail explicitly.

## Metrics

Classification accuracy compares case-folded word-token labels against the reference. Extra explanation is penalized. Extraction uses normalized exact match and token-multiset F1: twice overlap divided by predicted plus reference token counts. Case/punctuation normalization can obscure distinctions. The primary metric controls selection; other metrics remain separate.

numeric_answer removes comma separators and compares the final signed decimal number in the response with the numeric reference. It checks final answers, not reasoning or faithfulness; unrelated final numbers can be false positives. No approximate numeric tolerance, scientific notation or locale-specific formatting is implied.

code_test_success is the fraction of controlled cases passed by a capability-restricted expression interpreter. It accepts one named Python function with one return expression, bounded scalar values, arithmetic/comparison/conditional expressions and abs/min/max/len calls. A standalone trusted subprocess has isolated Python flags, minimal environment, temporary working directory and a two-second deadline. No generated exec/eval/shell execution occurs. Imports, attributes, loops, assignment, recursion and arbitrary calls are rejected. Markdown code fences can be stripped; extra explanatory code/docstrings are unsupported. Malformed, unsupported or timed-out programs score zero. This is not general Python pass@1; see [security boundary](../SECURITY.md).

The functional suite's summarization token F1 is lexical overlap, not factuality or human judgment. The real baseline excludes summarization. Never pool unrelated metric scales into a generic quality average.

## Prediction and selection

Historical medians match exact configuration (including quantization/options/generation), hardware fingerprint, task category, dataset/version, effective generation, warm/cold mode, primary metric and simulated/real status. Only successful calibration rows without a policy are eligible. Optional character-length-ratio filtering is conservative; it is not token-aware prediction. There is no fallback across incompatible hardware, categories, generations or metrics. Missing evidence produces abstention rather than an unjustified global estimate. Tests preserve this boundary. Candidates, sample counts and frozen run IDs make the calculation inspectable.

Adaptive utility normalizes available objectives across feasible candidates, applies declared positive weights and excludes dominated candidates. Unavailable objectives are inactive; active weights renormalize. Hard constraints concern predictions, not future guarantees. Largest means largest verified artifact, not strongest quality. Highest-quality means the calibration-selected highest predicted quality, not a held-out oracle. Deterministic ties use label and configuration hash.

## Statistics and comparative quantities

Raw summary/policy tables report successful-run count, mean, median, sample standard deviation (n-1), minimum and maximum. They suppress confidence intervals because repetitions on the same task are not independent. Missing measurements have distinct smaller counts; failures are not zero quality or zero latency.

research.json and aggregate-results.csv first average successful repetitions within each task, then summarize independent task means within experiment/category/dataset/metric/hardware/mode/split/simulation/policy groups. Percentile bootstrap confidence intervals for the mean use 2000 resamples with replacement and seed 42. Sorted resample means at one-based positions 50 and 1950 define the 95% interval. Fewer than three task means suppresses CI; fewer than ten is marked limited. The baseline has three task instances per category, so every estimated interval is limited. Constant bootstrap scores do not imply future certainty. The estimand is conditional on successful service; failure and abstention denominators are reported separately. No hypothesis tests are used.

Effects compare identical task sets/context against highest_quality. Quality retention = policy task-mean quality / reference task-mean quality, only for a positive reference and complete comparable metric coverage. Absolute quality difference = policy minus reference. Latency/RAM/VRAM reduction percent = 100 * (1 - policy/reference). Throughput increase percent = 100 * (policy/reference - 1). Zero reference or missing values yield null, never zero savings. Negative reductions mean increases. Constraint violation difference = reference satisfaction minus policy satisfaction. No pooled cross-category quality effect is computed.

Prediction errors compare the selected candidate's frozen estimate with its successful held-out observation. Compute absolute errors within task, average over repetitions, then report task-level MAE and median absolute error. MAPE averages per-task mean absolute percentage errors over nonzero observed denominators; report its smaller task count. Quality MAPE is suppressed because bounded scores and zero failures make it misleading. Selected-candidate errors do not establish counterfactual accuracy for other candidates.

Constraint satisfaction uses known observed constraints among successful served requests; unknown compliance is excluded and counted. Failure rate uses attempted inferences. Request summaries additionally report requested, served and abstained counts/rates. Abstention is neither success nor an inexpensive completed request.

Pareto reporting accepts quality, latency, RAM, VRAM, throughput and size objectives, grouped by identical task/hardware/mode/simulation/split/policy context. Missing objectives stay incomparable. Publication plots use compatible category/context groups and optional two-dimensional projections. A plotted descriptive frontier of group means is not a significance statement or a per-task optimality claim.

## Validity and reproducibility

The real run used a personal desktop with background applications and development/verification activity. Order was fixed; temperature, scheduler and caches were not controlled. GPU support was installed during collection; the original hardware fingerprint stayed frozen and a later NVML snapshot supplements device identity. Exact prompts, configurations, digests, runtime versions, seeds, timestamps and training run IDs are archived. Tags and software can drift, and reproduction cannot promise bitwise identity.

See [threats to validity](../research/threats_to_validity.md) for small samples, prompt sensitivity, quantization confounding, training leakage, stochasticity, server behavior and memory limits. The curated artifact is preliminary evidence of the workflow and its failures, not general superiority.
