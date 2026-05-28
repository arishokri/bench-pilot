from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _list_nvme_devices() -> list[str]:
    out: list[str] = []
    for entry in sorted(Path("/sys/block").glob("nvme*n*")):
        out.append(f"/dev/{entry.name}")
    return out


class DiskProbe:
    """smartctl -A on each NVMe namespace. Pulls composite temp + usage.

    Most consumer NVMes don't expose live util percentage, so we surface what
    SMART gives us: temperature(s), 'Available Spare' and 'Percentage Used'.
    """

    def __init__(self) -> None:
        self._devs = _list_nvme_devices() if shutil.which("smartctl") else []

    def sample(self, ts: float) -> list[tuple]:
        rows: list[tuple] = []
        for dev in self._devs:
            try:
                out = subprocess.check_output(
                    ["smartctl", "-A", "-j", dev], text=True, timeout=2.0,
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
            import json as _json
            try:
                d = _json.loads(out)
            except _json.JSONDecodeError:
                continue
            label = Path(dev).name
            src = "disk"
            health = d.get("nvme_smart_health_information_log", {})
            temp = health.get("temperature")
            if temp is not None:
                rows.append((ts, src, "temp", label, float(temp), "C"))
            spare = health.get("available_spare")
            if spare is not None:
                rows.append((ts, src, "spare", label, float(spare), "%"))
            used = health.get("percentage_used")
            if used is not None:
                rows.append((ts, src, "wear", label, float(used), "%"))
            # Some drives report extra sensors
            for i, t in enumerate(health.get("temperature_sensors", []) or []):
                rows.append((ts, src, "temp", f"{label}:s{i}", float(t), "C"))
        return rows
