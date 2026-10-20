from __future__ import annotations

import json
import os
from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import Field, JsonValue

from adaptiveedge.backends.base import Availability
from adaptiveedge.core.models import (
    InferenceResult,
    ModelConfiguration,
    Record,
    TaskDefinition,
    utcnow,
)

if TYPE_CHECKING:
    import httpx


class GenerateResponse(Record):
    response: str = ""
    thinking: str = ""
    done: bool = False
    error: str | None = None
    model: str = ""
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    prompt_eval_duration: int | None = None
    context: list[int] = Field(default_factory=list)


class OllamaBackend:
    def __init__(
        self, endpoint: str | None = None, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.endpoint = (
            endpoint or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self.transport = transport

    def _client(self) -> httpx.AsyncClient:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install adaptiveedge[ollama] for the Ollama backend") from exc
        return httpx.AsyncClient(base_url=self.endpoint, timeout=10, transport=self.transport)

    async def availability(self) -> Availability:
        try:
            async with self._client() as client:
                response = await client.get("/api/version")
                response.raise_for_status()
                tags = await client.get("/api/tags")
                tags.raise_for_status()
                return Availability(
                    available=True,
                    detail=self.endpoint,
                    version=str(response.json()["version"]),
                    metadata={"models": tags.json()["models"]},
                )
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            return Availability(available=False, detail=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            if type(exc).__module__.startswith("httpx"):
                return Availability(available=False, detail=f"Ollama connection failed: {exc}")
            raise

    async def discover(self) -> tuple[str, ...]:
        async with self._client() as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            return tuple(sorted(str(model["name"]) for model in response.json()["models"]))

    async def validate(self, configuration: ModelConfiguration) -> None:
        if configuration.model not in await self.discover():
            raise ValueError(
                f"Ollama model is not installed: {configuration.model}; pull it explicitly"
            )

    async def infer(
        self, configuration: ModelConfiguration, task: TaskDefinition, seed: int, cold: bool = False
    ) -> InferenceResult:
        generation = task.generation or configuration.generation
        options: dict[str, JsonValue] = {o.name: o.value for o in configuration.options}
        think = options.pop("think", None)
        if think is not None and not isinstance(think, bool):
            raise ValueError("think must be boolean")
        reserved = {"temperature", "top_p", "num_predict", "num_ctx", "seed"}
        if reserved.intersection(options):
            raise ValueError("runtime options cannot override typed generation settings or seed")
        options.update(
            temperature=generation.temperature,
            top_p=generation.top_p,
            num_predict=generation.max_output_tokens,
            num_ctx=configuration.context_size,
            seed=seed,
        )
        if configuration.device != "auto":
            raise ValueError("Ollama device selection is managed by the server; use device=auto")
        async with self._client() as client:
            if cold:
                unloaded = await client.post(
                    "/api/generate", json={"model": configuration.model, "keep_alive": 0}
                )
                unloaded.raise_for_status()
            started = utcnow()
            clock = perf_counter()
            output: list[str] = []
            ttft: float | None = None
            final: GenerateResponse | None = None
            async with client.stream(
                "POST",
                "/api/generate",
                json={
                    "model": configuration.model,
                    "prompt": task.prompt,
                    "stream": True,
                    **({"think": think} if think is not None else {}),
                    "options": options,
                    "keep_alive": 0 if cold else "5m",
                },
                timeout=None,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    known = {k: v for k, v in payload.items() if k in GenerateResponse.model_fields}
                    chunk = GenerateResponse.model_validate(known)
                    if chunk.error:
                        raise RuntimeError(chunk.error)
                    if chunk.response and ttft is None:
                        ttft = (perf_counter() - clock) * 1000
                    output.append(chunk.response)
                    if chunk.done:
                        final = chunk
            if final is None:
                raise RuntimeError("Ollama stream ended without a completion record")
            latency = (perf_counter() - clock) * 1000
            generation_ms = final.eval_duration / 1e6 if final.eval_duration is not None else None
            throughput = (
                final.eval_count * 1000 / generation_ms
                if final.eval_count is not None and generation_ms is not None and generation_ms > 0
                else None
            )
            return InferenceResult(
                configuration=configuration,
                output="".join(output),
                prompt_tokens=final.prompt_eval_count,
                completion_tokens=final.eval_count,
                latency_ms=latency,
                ttft_ms=ttft,
                generation_ms=generation_ms,
                load_ms=final.load_duration / 1e6 if final.load_duration is not None else None,
                throughput_tps=throughput,
                started_at=started,
                finished_at=utcnow(),
                metadata={
                    "endpoint": self.endpoint,
                    "resolved_model": final.model,
                    "server_total_ns": final.total_duration,
                    "prompt_eval_ns": final.prompt_eval_duration,
                    "memory_notice": "Client RSS is not server model RAM/VRAM",
                },
            )
