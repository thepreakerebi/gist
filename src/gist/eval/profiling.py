"""Measured cost accounting: wall clock and peak GPU memory, per stage.

Gist's claim is structural — it selects *before* the encoders run, so it should
save encoder FLOPs and peak memory rather than only decoder work. Until now the
FLOPs half of that was derived analytically from the encoder configs
(``gist.eval.efficiency``) and the memory half was not measured at all. An
efficiency claim supported only by arithmetic is the one criticism the design
cannot argue its way out of, so this module supplies the instrument.

Four decisions worth stating, because each one changes the number:

*Synchronise before reading the clock.* CUDA kernels are queued asynchronously.
Timing a block without ``torch.cuda.synchronize()`` measures how long the work
took to enqueue, not how long it took to run, and the error is easily large
enough to invert a comparison between two conditions.

*Report the delta over resident memory, not the total.* Qwen2.5-Omni-7B's fp16
weights are roughly 16 GB and they sit in memory under every condition.
Reporting total peak would bury the quantity actually under study — the
activation footprint of encoding K items — beneath a constant that is identical
for the dense baseline and for Gist. Both are recorded; ``activation_gb`` is the
one that answers the question, ``peak_reserved_gb`` is the one that decides
whether the card OOMs.

*An out-of-memory error is a measurement, not a crash.* The OmniZip head to head
(``results/runpod-omnizip-h2h/``) already found that a post-encoder method
allocates about 16.8 GB on a full-length clip before it prunes anything, and
OOMs on a 24 GB card. That is the pre-encoder argument in its strongest form, so
the harness records the failure, frees the allocator and carries on, instead of
losing the rest of a paid run to it.

*Torch is optional here.* With no torch, or torch without CUDA, the timers still
work and the memory fields stay ``None``. The harness can therefore be exercised
end to end on a laptop before a pod is paid for.

Two later additions, for comparing methods whose costs fall in different places:

*Energy comes from the GPU's own counter.* NVML exposes the board's cumulative
energy in millijoules since driver load, so a stage's energy is the difference
across it, read after the synchronise. It covers everything the card did,
including work outside PyTorch such as CTranslate2's Whisper, which a PyTorch
FLOP counter cannot see. CPU energy is not available inside a container, so CPU
work is reported as time only.

*FLOPs are counted, not derived, when asked.* ``count_flops=True`` runs the
block under ``torch.utils.flop_counter.FlopCounterMode``, which tallies matrix
multiplies, convolutions and attention per dispatched op. The counter slows the
block down, so a counted pass is a separate pass and its timings are not used.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_BYTES_PER_GB = 1e9

#: Discriminator on the first line of a measurement log, so the JSONL is
#: self-describing. The citation audit's lesson was that a number whose
#: provenance was not written down at the time cannot be recovered later.
DEVICE_RECORD = "device"
STAGE_RECORD = "stage"


def _torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _round_gb(num_bytes: int | None) -> float | None:
    return None if num_bytes is None else round(num_bytes / _BYTES_PER_GB, 3)


_NVML_HANDLE: Any | None = None
_NVML_TRIED = False


def _nvml_handle() -> Any | None:
    """Device 0's NVML handle, or None where NVML or the energy counter is absent."""
    global _NVML_HANDLE, _NVML_TRIED
    if _NVML_TRIED:
        return _NVML_HANDLE
    _NVML_TRIED = True
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except Exception:  # noqa: BLE001 - no NVML, no GPU, or a board without the counter
        return None
    _NVML_HANDLE = handle
    return handle


def gpu_energy_mj() -> int | None:
    """Cumulative board energy in millijoules, or None if it cannot be read."""
    handle = _nvml_handle()
    if handle is None:
        return None
    import pynvml

    return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))


def idle_power_w(seconds: float = 10.0) -> float | None:
    """Mean board power while nothing runs, so stage energy can be read above idle."""
    start = gpu_energy_mj()
    if start is None:
        return None
    time.sleep(seconds)
    end = gpu_energy_mj()
    return round((end - start) / 1000.0 / seconds, 2)


def cuda_available() -> bool:
    torch = _torch()
    return bool(torch is not None and torch.cuda.is_available())


def is_out_of_memory(exc: BaseException) -> bool:
    """True for a CUDA OOM however this torch version chooses to raise it.

    Torch >= 2.5 exposes ``torch.OutOfMemoryError``; older versions raise
    ``torch.cuda.OutOfMemoryError`` or a bare ``RuntimeError``. The string check
    is the backstop and also catches the allocator's cuDNN variants.
    """
    torch = _torch()
    if torch is not None:
        oom_type = getattr(torch, "OutOfMemoryError", None) or getattr(
            torch.cuda, "OutOfMemoryError", None
        )
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    return "out of memory" in str(exc).lower()


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """What the numbers were measured on. Written once per log."""

    available: bool
    name: str | None = None
    total_memory_gb: float | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    device_count: int = 0

    @classmethod
    def capture(cls) -> DeviceInfo:
        torch = _torch()
        if torch is None:
            return cls(available=False)
        if not torch.cuda.is_available():
            return cls(available=False, torch_version=torch.__version__)
        props = torch.cuda.get_device_properties(0)
        return cls(
            available=True,
            name=props.name,
            total_memory_gb=_round_gb(props.total_memory),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            device_count=torch.cuda.device_count(),
        )

    def describe(self) -> str:
        if not self.available:
            return "cpu (no CUDA device; memory figures unavailable)"
        return (
            f"{self.name} {self.total_memory_gb:.0f} GB, torch {self.torch_version}, "
            f"CUDA {self.cuda_version}"
        )


@dataclass
class StageMeasurement:
    """One timed block. ``extra`` carries whatever the caller wants to join on."""

    stage: str
    wall_seconds: float = 0.0
    resident_gb: float | None = None
    peak_allocated_gb: float | None = None
    peak_reserved_gb: float | None = None
    activation_gb: float | None = None
    energy_j: float | None = None
    flops: int | None = None
    oom: bool = False
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["record"] = STAGE_RECORD
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> StageMeasurement:
        fields = {k: v for k, v in payload.items() if k != "record"}
        return cls(**fields)


@contextmanager
def measure(
    stage: str,
    *,
    suppress_oom: bool = True,
    count_flops: bool = False,
    sink: Callable[[StageMeasurement], Any] | None = None,
    **extra: Any,
) -> Iterator[StageMeasurement]:
    """Time a block and record its GPU high-water marks.

    The record is yielded so the caller can attach results to it inside the
    block. An OOM is recorded and suppressed by default, because losing the
    remaining questions of a paid run to one oversized input is worse than
    carrying a failed row; every other exception propagates after being
    recorded.

    ``sink`` is called once with the finished record, on both the success and
    the failure path, immediately before any exception is re-raised. It exists
    because wrapping this in an outer context manager does not work: the outer
    manager's ``finally`` fires while the exception is still travelling through
    this generator, so it would persist the record before the timings and the
    error had been written to it.
    """
    record = StageMeasurement(stage=stage, extra=dict(extra))
    torch = _torch()
    on_gpu = bool(torch is not None and torch.cuda.is_available())

    if on_gpu:
        torch.cuda.synchronize()
        record.resident_gb = _round_gb(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()

    counter = None
    if count_flops and torch is not None:
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        counter.__enter__()

    energy_start = gpu_energy_mj() if on_gpu else None
    start = time.perf_counter()
    failure: BaseException | None = None
    try:
        yield record
        if on_gpu:
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - recorded on the row, then re-raised
        failure = exc
        record.error = f"{type(exc).__name__}: {exc}"[:400]
        record.oom = is_out_of_memory(exc)
    finally:
        if counter is not None:
            counter.__exit__(None, None, None)
            record.flops = int(counter.get_total_flops())
    record.wall_seconds = round(time.perf_counter() - start, 4)
    if energy_start is not None:
        record.energy_j = round((gpu_energy_mj() - energy_start) / 1000.0, 3)

    if on_gpu:
        record.peak_allocated_gb = _round_gb(torch.cuda.max_memory_allocated())
        record.peak_reserved_gb = _round_gb(torch.cuda.max_memory_reserved())
        if record.resident_gb is not None and record.peak_allocated_gb is not None:
            record.activation_gb = round(record.peak_allocated_gb - record.resident_gb, 3)
        if record.oom:
            torch.cuda.empty_cache()

    if sink is not None:
        sink(record)
    if failure is not None and not (record.oom and suppress_oom):
        raise failure


class MeasurementLog:
    """Append-only JSONL of stage measurements, headed by the device it ran on."""

    def __init__(self, path: str | Path, *, device: DeviceInfo | None = None) -> None:
        self.path = Path(path)
        self.device = device or DeviceInfo.capture()
        self.records: list[StageMeasurement] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = {"record": DEVICE_RECORD, **asdict(self.device)}
        self.path.write_text(json.dumps(header) + "\n")

    def note(self, kind: str, **data: Any) -> None:
        """A non-stage line, e.g. idle power, kept beside the stages it explains."""
        with self.path.open("a") as handle:
            handle.write(json.dumps({"record": kind, **data}) + "\n")

    def append(self, record: StageMeasurement) -> StageMeasurement:
        self.records.append(record)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record.to_json()) + "\n")
        return record

    @contextmanager
    def measure(self, stage: str, **kwargs: Any) -> Iterator[StageMeasurement]:
        """``measure`` that persists the finished row, including on failure.

        A measurement harness that loses the record of the thing that went wrong
        is worse than useless on a paid run, because the failure is exactly what
        needs explaining afterwards.
        """
        with measure(stage, sink=self.append, **kwargs) as record:
            yield record


def load_measurements(path: str | Path) -> tuple[DeviceInfo | None, list[StageMeasurement]]:
    device: DeviceInfo | None = None
    stages: list[StageMeasurement] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        kind = payload.get("record", STAGE_RECORD)
        if kind == DEVICE_RECORD:
            device = DeviceInfo(**{k: v for k, v in payload.items() if k != "record"})
        elif kind == STAGE_RECORD:
            stages.append(StageMeasurement.from_json(payload))
    return device, stages
