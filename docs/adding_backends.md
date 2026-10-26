# Adding a backend

Implement the asynchronous `Backend` protocol in `backends/base.py`: `availability`, `discover`, `validate` and `infer`. Register an instance through `BackendRegistry.register` and inject the registry into `AdaptiveEdge` or `BenchmarkRunner`. The selector consumes typed historical records and must not branch on runtime-specific HTTP or tensor details.

Return `Availability` with a status, explanation, optional version and structured metadata. Discovery should only enumerate already available models. Validation must fail informatively for missing models, unsupported devices and invalid settings. Do not download artifacts or upload user data during discovery, validation or import.

Inference receives a model config, task, repetition seed and cold flag. Return an `InferenceResult` with the original configuration, final answer text, actual token counts/durations where known, UTC boundaries, correctly scoped memory and explicit error/usage metadata. Do not infer missing token counts from characters and label them measured. Distinguish generation time, loading time, request latency and TTFT. Add structured runtime extension options through named `RuntimeOption` values; typed fields and seeds take precedence.

The runner applies `asyncio.timeout` to availability, validation, warm-up and inference. Adapters must cooperate with cancellation and close network/process resources on cancellation. Blocking synchronous work or a native call that ignores cancellation cannot be preempted by this deadline. A Transformers adapter must run in a terminable subprocess or use a defensible runtime cancellation strategy, not wrap blocking generation in an abandoned thread. This is why the optional Transformers environment currently has no working adapter.

Model RAM/VRAM require a process/device attribution method; client RSS is not model-server memory. Device-wide snapshots and per-process peaks need distinct scopes. For unsupported methods, return a `Measurement` with no value and the appropriate status. An actual zero remains a measured value. Backend-reported duration or tokens should retain source provenance in metadata.

Contract tests should cover deterministic fixtures, validation errors, no usage, truncated streams, malformed messages, backend failures, timeouts, cancellation and unknown memory. Use temporary resources and in-memory HTTP transports. Optional live tests must be explicitly invoked and must never pull multi-gigabyte models in default CI. Consult `tests/unit/test_ollama.py` for the implemented transport contract.

## Adding datasets, metrics and policies

Materialize versioned tasks through the dataset-provider protocol into a custom benchmark config. Download/import operations must be explicit and license-aware. Use stable IDs, canonical prompts, dataset versions and held-out splits. Keep train/test task-family variants together when deduplicating.

Metrics are callable functions `(output, reference) -> score` registered on `MetricRegistry`; scores in benchmark records must be finite in [0,1]. Task metric identifiers are explicit nonempty names. Register new metrics before running custom tasks; benchmark preflight rejects unregistered identifiers, preventing silent name typos. Document the scale and semantics of every extension.

Policies implement `score(candidates)` and register through `Selector.register_policy`. Core constraint filtering remains outside policy code. Return null for unscorable candidates. Preserve deterministic results and document objective normalization, missing data and tie-breaking. Predictors implement `predict(configuration, task, hardware, history)` and must explain grouping and any extrapolation assumptions.
