# Threats to validity

The baseline has three artifacts and three tasks per category on one machine. Task-level bootstrap intervals are marked limited. Constant observed scores do not establish future certainty. Repetitions measure within-task variation and do not enlarge the independent task sample.

Authored prompts are disjoint by ID/text but deliberately share templates and skills. This is an in-template held-out check, not task-family generalization. Model training overlap cannot be ruled out. Leakage checks cannot detect semantic paraphrases, pretraining contamination or informal human tuning. Weights and fixture rules were declared before evaluation and not tuned on outcomes.

Metrics measure normalized labels, city extraction, final numeric answers and restricted-expression fixture success. They do not measure human preference, reasoning faithfulness, general coding competence or summarization. Last-number extraction can accept extraneous text or a coincidentally correct last number. Code scoring rejects correct programs outside its subset. Tokenization/output lengths differ; tokens/s is runtime- and tokenizer-specific.

Quantization, family, artifact size and prompt adherence vary jointly. The experiment cannot isolate scale from family/quantization effects. Seeds and zero temperature do not guarantee identical output across versions/hardware. Prompt wording, templates, context length and output caps matter. Mutable tags can drift; digests identify artifacts but cannot guarantee old artifacts remain downloadable.

Runtime, OS scheduling, other applications, thermal state, drivers, filesystem caches and model residency affect timings. Model/policy orders are fixed. Task-specific warm-ups can cache exact prompts. The equal-weight ablation runs later; timing differences are descriptive, not causal weight effects. Warm requests can still load. Cold checks unload the selected model without clearing all caches.

Client sampled RSS excludes server/runner memory. Process snapshots and resident GPU allocation are diagnostics, not model inference peaks. Model RAM/VRAM reductions are unavailable. Lowest-memory abstention is not low-cost service. Device-wide NVML usage includes other workloads. Neither energy nor FLOPs is measured.

Prediction conditions on successful calibration and uses category medians rather than calibrated probabilistic forecasts. Medians can hide task failures. Unknown required evidence causes abstention; report coverage alongside served-request quality. Prediction errors concern selected candidates, not unobserved counterfactuals. Ratios with zero reference means are undefined. Descriptive effects/intervals do not establish significance or broad superiority.
