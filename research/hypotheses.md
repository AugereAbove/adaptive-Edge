# Preregistered study template

These hypotheses are untested. Complete the dataset, model inventory, hardware, split, metric, resource budgets and success criteria before collecting held-out evaluation data.

- **H1:** Adaptive selection reduces average latency relative to always choosing the empirically highest-quality model.
- **H2:** Adaptive selection reduces average model-runtime memory consumption relative to that baseline.
- **H3:** Adaptive selection retains a predeclared proportion of the highest-quality policy's task performance.
- **H4:** The usefulness of adaptive selection varies across task categories and resource constraints.

Define the primary estimand per category and quality metric. Declare the retained-quality threshold, latency/memory effect sizes, independent sampling unit, failure treatment, abstention treatment, multiple-comparison handling and uncertainty method before testing. Do not choose these after observing evaluation results.

Recommended first study: two or three installed model configurations, a fixed local machine, a licensed versioned set of answer-checkable classification/extraction/reasoning tasks, disjoint calibration and test splits, five warm repetitions per task, and all six policies. Counterbalance model/policy order across runs. Sweep latency budgets and report paired task-level cost/quality deltas, actual violation rate and coverage of served requests. Tune weights only on calibration/validation data.

Ollama server model memory is not measured by v0.1. H2 requires a separately validated server-process/GPU profiler before evaluation; client RSS cannot test H2. Start with H1/H3/H4, then add the profiler and preregister H2.
