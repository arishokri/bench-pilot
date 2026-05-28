from __future__ import annotations

from pathlib import Path

_HWMON = Path("/sys/class/hwmon")


def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def _read_float(p: Path) -> float | None:
    s = _read(p)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


class HwmonProbe:
    """Walks /sys/class/hwmon for temps, fans, voltage, current and power.

    Auto-discovers chips and channels on first sample(); no manual config needed.
    """

    def __init__(self) -> None:
        self._channels: list[tuple[str, str, str, str, Path, float]] = []
        self._discovered = False

    def _discover(self) -> None:
        if not _HWMON.exists():
            self._discovered = True
            return
        for chip in sorted(_HWMON.iterdir()):
            name = _read(chip / "name") or chip.name
            for entry in sorted(chip.iterdir()):
                fn = entry.name
                # Match patterns: tempN_input, fanN_input, inN_input, currN_input, powerN_input
                if not fn.endswith("_input"):
                    continue
                prefix = fn[:-len("_input")]
                kind = prefix.rstrip("0123456789")
                idx = prefix[len(kind):]
                if kind not in {"temp", "fan", "in", "curr", "power"}:
                    continue
                metric, unit, scale = {
                    "temp": ("temp", "C", 1 / 1000.0),
                    "fan": ("rpm", "rpm", 1.0),
                    "in": ("voltage", "V", 1 / 1000.0),
                    "curr": ("current", "A", 1 / 1000.0),
                    "power": ("power", "W", 1 / 1_000_000.0),
                }[kind]
                label_file = entry.with_name(f"{prefix}_label")
                label = _read(label_file) or f"{kind}{idx}"
                source = f"hwmon:{name}"
                self._channels.append((source, metric, label, unit, entry, scale))
        self._discovered = True

    def sample(self, ts: float) -> list[tuple]:
        if not self._discovered:
            self._discover()
        rows: list[tuple] = []
        for src, metric, label, unit, path, scale in self._channels:
            raw = _read_float(path)
            if raw is None:
                continue
            rows.append((ts, src, metric, label, raw * scale, unit))
        return rows
