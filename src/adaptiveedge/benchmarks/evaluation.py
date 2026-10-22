from collections.abc import Sequence
from uuid import uuid4

from adaptiveedge.backends.base import BackendRegistry
from adaptiveedge.benchmarks.runner import BenchmarkRunner, software_versions
from adaptiveedge.config.schema import BenchmarkConfiguration
from adaptiveedge.core.models import BenchmarkResult, Constraints, Estimates, HardwareSnapshot
from adaptiveedge.selection.engine import Selector, constraint_violations
from adaptiveedge.storage.repository import Repository

POLICIES = ("smallest", "largest", "fastest", "highest_quality", "lowest_memory", "adaptive")


async def evaluate_policies(
    repository: Repository,
    registry: BackendRegistry,
    hardware: HardwareSnapshot,
    training_id: str,
    evaluation: BenchmarkConfiguration,
    constraints: Constraints | None = None,
    selector: Selector | None = None,
    policies: Sequence[str] = POLICIES,
    allow_overlap: bool = False,
) -> str:
    if repository.configuration(training_id).experiment.split != "calibration":
        raise ValueError("training experiment must use the calibration split")
    history = tuple(r for r in repository.results(training_id) if r.split == "calibration")
    if not history:
        raise ValueError("training experiment has no observations")
    if evaluation.suite != "custom":
        raise ValueError("policy evaluation requires explicit held-out custom tasks")
    training_prompts = {r.task.prompt for r in history}
    training_ids = {r.task.id for r in history}
    if not allow_overlap and any(
        task.prompt in training_prompts or task.id in training_ids for task in evaluation.tasks
    ):
        raise ValueError(
            "evaluation tasks overlap training; use held-out tasks or explicit smoke mode"
        )
    settings = evaluation.experiment
    evaluation = evaluation.model_copy(
        update={
            "experiment": settings.model_copy(
                update={"split": "functional" if allow_overlap else "evaluation"}
            )
        }
    )
    if settings.mode != "warm":
        raise ValueError("policy evaluation currently supports warm mode only")
    selector = selector or Selector()
    constraints = constraints or Constraints()
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("policies must be nonempty and unique")
    if any(policy not in selector.policies for policy in policies):
        raise ValueError("unknown evaluation policy")
    identifier = f"{evaluation.identity[:12]}-{uuid4().hex}"
    software = software_versions()
    repository.create_experiment(
        identifier,
        evaluation,
        hardware,
        software,
        {
            "kind": "policy_evaluation",
            "training_experiment": training_id,
            "training_run_ids": [r.run_id for r in history],
            "policies": list(policies),
            "constraints": constraints.model_dump(mode="json"),
            "allow_overlap": allow_overlap,
            "weights": selector.weights.model_dump(mode="json"),
        },
    )
    runner = BenchmarkRunner(repository, registry, hardware)
    interrupted = True
    try:
        for task in evaluation.tasks:
            for repetition in range(settings.repetitions):
                seed = (settings.seed + repetition) % 2147483648
                for policy in policies:
                    selection = selector.select(
                        task, hardware, evaluation.models, history, constraints, policy
                    )
                    if selection.selected is None:
                        repository.event(
                            identifier,
                            "abstention",
                            {
                                "task_id": task.id,
                                "policy": policy,
                                "repetition": repetition,
                                "selection": selection.model_dump(mode="json"),
                            },
                        )
                        continue
                    model = selection.selected
                    warm_failure = None
                    for _ in range(settings.warmup_runs):
                        warm = await runner.measure(
                            model, task, seed, settings.timeout_seconds, False
                        )
                        if warm.error:
                            warm_failure = warm
                            break
                    result = warm_failure or await runner.measure(
                        model, task, seed, settings.timeout_seconds, False
                    )
                    result, quality = runner.score(task, result)
                    actual = Estimates(
                        quality=quality.get(task.metrics[0]) if task.metrics else None,
                        latency_ms=result.latency_ms,
                        ram_gb=result.peak_ram_gb.value,
                        vram_gb=result.peak_vram_gb.value,
                        throughput_tps=result.throughput_tps,
                        size_gb=model.size_gb,
                        samples=1,
                    )
                    violations = constraint_violations(model, actual, constraints)
                    measured = result.model_copy(
                        update={
                            "metadata": {
                                **result.metadata,
                                "actual_constraint_violations": [
                                    v.model_dump(mode="json") for v in violations
                                ],
                                "warmup_failed": warm_failure is not None,
                            }
                        }
                    )
                    repository.save_result(
                        BenchmarkResult(
                            run_id=uuid4().hex,
                            split=evaluation.experiment.split,
                            experiment_id=identifier,
                            configuration=model,
                            hardware=hardware,
                            task=task,
                            inference=measured,
                            quality=quality,
                            seed=seed,
                            repetition=repetition,
                            software=software,
                            policy=policy,
                            selection=selection.model_dump(mode="json"),
                        )
                    )
        interrupted = False
    finally:
        repository.event(identifier, "interrupted" if interrupted else "completed", {})
    return identifier
