from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


class ToolMissing(RuntimeError):
    pass


@dataclass
class ShellResult:
    returncode: int
    stdout: str
    stderr: str


def require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise ToolMissing(f"required tool '{tool}' not found on PATH — run ./scripts/install-system-deps.sh")
    return path


def run(cmd: list[str], *, timeout: float | None = None, check: bool = True) -> ShellResult:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd)} failed (rc={proc.returncode})\nstderr: {proc.stderr.strip()}"
        )
    return ShellResult(proc.returncode, proc.stdout, proc.stderr)
