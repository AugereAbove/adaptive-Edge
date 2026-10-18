import importlib

from adaptiveedge.core.models import Measurement, Record


class GpuSnapshot(Record):
    names: tuple[str, ...] = ()
    count: int | None = None
    total: Measurement
    available: Measurement
    runtime: str | None = None


def gpu_snapshot() -> GpuSnapshot:
    try:
        nvml = importlib.import_module("pynvml")
    except ImportError:
        missing = Measurement(
            status="unsupported",
            unit="GB",
            scope="device_system",
            detail="Install the gpu extra for optional NVIDIA NVML detection",
        )
        return GpuSnapshot(total=missing, available=missing)
    initialized = False
    try:
        nvml.nvmlInit()
        initialized = True
        count = int(nvml.nvmlDeviceGetCount())
        names = []
        total = 0
        available = 0
        for index in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            name = nvml.nvmlDeviceGetName(handle)
            names.append(name.decode() if isinstance(name, bytes) else str(name))
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            total += memory.total
            available += memory.free
        return GpuSnapshot(
            names=tuple(names),
            count=count,
            runtime="NVIDIA NVML",
            total=Measurement(
                value=total / 1e9, status="measured", unit="GB", scope="device_system"
            ),
            available=Measurement(
                value=available / 1e9, status="measured", unit="GB", scope="device_system"
            ),
        )
    except nvml.NVMLError as exc:
        failed = Measurement(status="failed", unit="GB", scope="device_system", detail=str(exc))
        return GpuSnapshot(total=failed, available=failed, runtime="NVIDIA NVML")
    finally:
        if initialized:
            nvml.nvmlShutdown()
