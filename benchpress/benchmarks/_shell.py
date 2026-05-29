from __future__ import annotations

import shutil
import subprocess
import threading
import time
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


def run(
    cmd: list[str],
    *,
    timeout: float | None = None,
    check: bool = True,
    cancel: threading.Event | None = None,
) -> ShellResult:
    if cancel is None:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    else:
        rc, out, err = _run_cancellable(cmd, timeout=timeout, cancel=cancel)
    if check and rc != 0:
        raise RuntimeError(
            f"{' '.join(cmd)} failed (rc={rc})\nstderr: {err.strip()}"
        )
    return ShellResult(rc, out, err)


def _run_cancellable(
    cmd: list[str], *, timeout: float | None, cancel: threading.Event,
) -> tuple[int, str, str]:
    """Run `cmd`, polling so a set `cancel` event (or `timeout`) terminates the child promptly.

    Sends SIGTERM first (stress-ng exits cleanly on it), then SIGKILL as a last resort.
    """
    deadline = (time.monotonic() + timeout) if timeout is not None else None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        try:
            out, err = proc.communicate(timeout=0.5)
            return proc.returncode, out, err
        except subprocess.TimeoutExpired:
            expired = deadline is not None and time.monotonic() >= deadline
            if cancel.is_set() or expired:
                proc.terminate()
                try:
                    out, err = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out, err = proc.communicate()
                return proc.returncode, out, err
