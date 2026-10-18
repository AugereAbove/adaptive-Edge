import asyncio
import contextlib
import platform
from types import TracebackType

import psutil

from adaptiveedge.core.models import HardwareSnapshot, Measurement
from adaptiveedge.profiling.gpu import gpu_snapshot


def hardware_snapshot() -> HardwareSnapshot:
    memory = psutil.virtual_memory()
    gpu = gpu_snapshot()
    return HardwareSnapshot(
        os=f"{platform.system()} {platform.release()}",
        cpu=platform.processor() or platform.machine(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        total_ram_gb=memory.total / 1e9,
        available_ram_gb=memory.available / 1e9,
        gpu_models=gpu.names,
        gpu_count=gpu.count,
        total_vram_gb=gpu.total,
        available_vram_gb=gpu.available,
        accelerator=gpu.runtime,
    )


class ProcessSampler:
    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.peak = 0.0
        self.error: str | None = None
        self._worker: asyncio.Task[None] | None = None

    def sample(self) -> None:
        try:
            self.peak = max(self.peak, psutil.Process().memory_info().rss / 1e9)
        except psutil.Error as exc:
            self.error = type(exc).__name__

    async def _poll(self) -> None:
        while True:
            self.sample()
            await asyncio.sleep(self.interval)

    async def __aenter__(self) -> "ProcessSampler":
        self.sample()
        self._worker = asyncio.create_task(self._poll())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
        self.sample()

    def measurement(self) -> Measurement:
        if self.error:
            return Measurement(
                status="failed", unit="GB", scope="client_process", detail=self.error
            )
        return Measurement(
            value=self.peak,
            status="measured",
            unit="GB",
            scope="client_process",
            detail=f"Sampled RSS; interval {self.interval}s; may miss short peaks",
        )
