from __future__ import annotations

from pathlib import Path

_PROC_STAT = Path("/proc/stat")
_CPUINFO = Path("/proc/cpuinfo")


class CpuProbe:
    """Per-CPU utilisation via /proc/stat deltas. Also exposes current MHz from /proc/cpuinfo."""

    def __init__(self) -> None:
        self._prev: dict[str, tuple[int, int]] = {}  # cpu -> (idle, total)

    def _read_stat(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for line in _PROC_STAT.read_text().splitlines():
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            name = parts[0]
            vals = list(map(int, parts[1:]))
            # user nice system idle iowait irq softirq steal guest guest_nice
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
            total = sum(vals)
            out[name] = (idle, total)
        return out

    def _read_freqs_mhz(self) -> list[tuple[int, float]]:
        if not _CPUINFO.exists():
            return []
        out: list[tuple[int, float]] = []
        cur_cpu: int | None = None
        for line in _CPUINFO.read_text().splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if k == "processor":
                try:
                    cur_cpu = int(v)
                except ValueError:
                    cur_cpu = None
            elif k == "cpu MHz" and cur_cpu is not None:
                try:
                    out.append((cur_cpu, float(v)))
                except ValueError:
                    pass
        return out

    def sample(self, ts: float) -> list[tuple]:
        rows: list[tuple] = []
        cur = self._read_stat()
        for name, (idle, total) in cur.items():
            pidle, ptotal = self._prev.get(name, (idle, total))
            dt = total - ptotal
            di = idle - pidle
            if dt > 0:
                util = max(0.0, min(100.0, 100.0 * (1.0 - di / dt)))
                rows.append((ts, "cpu", "util", name, util, "%"))
        self._prev = cur

        for cpu_idx, mhz in self._read_freqs_mhz():
            rows.append((ts, "cpu", "clock", f"cpu{cpu_idx}", mhz, "MHz"))
        return rows
