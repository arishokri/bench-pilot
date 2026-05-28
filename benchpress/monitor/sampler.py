from __future__ import annotations

import threading
import time
from typing import Protocol

from benchpress.monitor.cpu import CpuProbe
from benchpress.monitor.disk import DiskProbe
from benchpress.monitor.hwmon import HwmonProbe
from benchpress.monitor.nvidia import NvidiaProbe, available as nvidia_available
from benchpress.storage import Storage


class _Probe(Protocol):
    def sample(self, ts: float) -> list[tuple]: ...


class Sampler:
    """Background thread that polls every probe every `interval` seconds and writes batches to SQLite.

    Designed to coexist with the benchmark process — sampling overhead is tiny (~5ms per tick).
    """

    def __init__(self, storage: Storage, run_id: int, interval: float = 1.0):
        self._storage = storage
        self._run_id = run_id
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._probes: list[_Probe] = [
            CpuProbe(),
            HwmonProbe(),
            DiskProbe(),
        ]
        if nvidia_available():
            self._probes.append(NvidiaProbe())

    def start(self) -> None:
        if self._thread is not None:
            return
        # Prime probes that need a previous tick (CpuProbe) so the first emitted
        # delta is non-zero.
        for p in self._probes:
            try:
                p.sample(time.time())
            except Exception:
                pass
        self._thread = threading.Thread(target=self._loop, name="benchpress-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            ts = time.time()
            batch: list[tuple] = []
            for probe in self._probes:
                try:
                    rows = probe.sample(ts)
                except Exception:
                    rows = []
                for r in rows:
                    # r: (ts, source, metric, label, value, unit)
                    batch.append((self._run_id, *r))
            if batch:
                try:
                    self._storage.insert_samples(batch)
                except Exception:
                    pass
            next_tick += self._interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self._stop.wait(timeout=sleep_for)
            else:
                next_tick = time.monotonic()
