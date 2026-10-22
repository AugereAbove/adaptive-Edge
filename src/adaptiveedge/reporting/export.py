import csv
import json
from collections.abc import Sequence
from pathlib import Path

from adaptiveedge.core.models import BenchmarkResult


def export_json(results: Sequence[BenchmarkResult], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([r.model_dump(mode="json") for r in results], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def export_csv(results: Sequence[BenchmarkResult], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "experiment_id",
        "split",
        "mode",
        "backend",
        "started_at",
        "finished_at",
        "run_id",
        "model",
        "configuration_id",
        "hardware_id",
        "task_id",
        "category",
        "policy",
        "repetition",
        "seed",
        "simulated",
        "success",
        "latency_ms",
        "generation_ms",
        "load_ms",
        "ttft_ms",
        "throughput_tps",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_characters",
        "peak_ram_gb",
        "ram_status",
        "peak_vram_gb",
        "vram_status",
        "client_peak_ram_gb",
        "quality_json",
        "error_json",
    )
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            inference = row.inference
            writer.writerow(
                {
                    "experiment_id": row.experiment_id,
                    "split": row.split,
                    "mode": inference.metadata.get("mode"),
                    "backend": row.configuration.backend,
                    "started_at": inference.started_at.isoformat(),
                    "finished_at": inference.finished_at.isoformat(),
                    "run_id": row.run_id,
                    "model": row.configuration.display_name,
                    "configuration_id": row.configuration.identity,
                    "hardware_id": row.hardware.fingerprint,
                    "task_id": row.task.id,
                    "category": row.task.category,
                    "policy": row.policy,
                    "repetition": row.repetition,
                    "seed": row.seed,
                    "simulated": inference.simulated,
                    "success": row.success,
                    "latency_ms": inference.latency_ms if row.success else None,
                    "generation_ms": inference.generation_ms,
                    "load_ms": inference.load_ms,
                    "ttft_ms": inference.ttft_ms,
                    "throughput_tps": inference.throughput_tps,
                    "prompt_tokens": inference.prompt_tokens,
                    "completion_tokens": inference.completion_tokens,
                    "total_tokens": inference.total_tokens,
                    "input_characters": len(row.task.prompt),
                    "peak_ram_gb": inference.peak_ram_gb.value,
                    "ram_status": inference.peak_ram_gb.status,
                    "peak_vram_gb": inference.peak_vram_gb.value,
                    "vram_status": inference.peak_vram_gb.status,
                    "client_peak_ram_gb": inference.client_peak_ram_gb.value,
                    "quality_json": json.dumps(row.quality, sort_keys=True),
                    "error_json": inference.error.model_dump_json() if inference.error else "",
                }
            )
