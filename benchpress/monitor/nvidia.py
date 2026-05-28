from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

# Fields we pull every tick from nvidia-smi. Order matters — we parse positionally.
_FIELDS = [
    "index",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "clocks.current.graphics",
    "clocks.current.memory",
    "clocks.current.sm",
    "fan.speed",
    "pstate",
]


@dataclass
class _Reading:
    index: int
    temp: float | None
    util_gpu: float | None
    util_mem: float | None
    mem_used_mib: float | None
    mem_total_mib: float | None
    power_w: float | None
    clock_graphics_mhz: float | None
    clock_memory_mhz: float | None
    clock_sm_mhz: float | None
    fan_pct: float | None


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _to_float(s: str) -> float | None:
    s = s.strip()
    if not s or s in {"[N/A]", "[Not Supported]", "N/A"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


class NvidiaProbe:
    """Polls nvidia-smi --query-gpu and emits sample tuples ready for storage."""

    def __init__(self) -> None:
        self._cmd = [
            "nvidia-smi",
            f"--query-gpu={','.join(_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]

    def sample(self, ts: float) -> list[tuple]:
        try:
            out = subprocess.check_output(self._cmd, text=True, timeout=2.0)
        except (subprocess.SubprocessError, FileNotFoundError):
            return []
        rows: list[tuple] = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < len(_FIELDS):
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            r = _Reading(
                index=idx,
                temp=_to_float(parts[1]),
                util_gpu=_to_float(parts[2]),
                util_mem=_to_float(parts[3]),
                mem_used_mib=_to_float(parts[4]),
                mem_total_mib=_to_float(parts[5]),
                power_w=_to_float(parts[6]),
                clock_graphics_mhz=_to_float(parts[7]),
                clock_memory_mhz=_to_float(parts[8]),
                clock_sm_mhz=_to_float(parts[9]),
                fan_pct=_to_float(parts[10]),
            )
            label = f"gpu{r.index}"
            src = "nvidia"

            def add(metric: str, value: float | None, unit: str) -> None:
                if value is not None:
                    rows.append((ts, src, metric, label, value, unit))

            add("temp", r.temp, "C")
            add("util_gpu", r.util_gpu, "%")
            add("util_mem", r.util_mem, "%")
            add("mem_used", r.mem_used_mib, "MiB")
            add("mem_total", r.mem_total_mib, "MiB")
            add("power", r.power_w, "W")
            add("clock_graphics", r.clock_graphics_mhz, "MHz")
            add("clock_memory", r.clock_memory_mhz, "MHz")
            add("clock_sm", r.clock_sm_mhz, "MHz")
            add("fan", r.fan_pct, "%")
        return rows
