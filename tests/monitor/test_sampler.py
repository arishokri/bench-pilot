import threading
import time

from benchpress.monitor.sampler import Sampler


class _StubProbe:
    """Returns one fixed row per sample()."""

    def __init__(self, source: str = "stub"):
        self.source = source
        self.calls = 0

    def sample(self, ts):
        self.calls += 1
        return [(ts, self.source, "metric", "lbl", float(self.calls), "u")]


def test_sampler_writes_rows_then_stops(temp_db, monkeypatch):
    # Disable nvidia/cpu/hwmon/disk probes and inject stubs
    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})

    s = Sampler(temp_db, run_id, interval=0.05)
    s._probes = [_StubProbe("stub-a"), _StubProbe("stub-b")]

    s.start()
    time.sleep(0.25)
    s.stop()

    rows = temp_db.samples_for_run(run_id)
    sources = {r["source"] for r in rows}
    assert sources == {"stub-a", "stub-b"}
    assert len(rows) >= 4  # at least 2 ticks × 2 probes


def test_sampler_idempotent_start_and_stop(temp_db):
    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})
    s = Sampler(temp_db, run_id, interval=0.05)
    s._probes = [_StubProbe()]
    s.start()
    s.start()  # second start should be a no-op
    s.stop()
    s.stop()   # second stop should be safe


def test_probe_exception_does_not_kill_sampler(temp_db):
    class BrokenProbe:
        def sample(self, ts): raise RuntimeError("boom")

    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})
    s = Sampler(temp_db, run_id, interval=0.05)
    good = _StubProbe("good")
    s._probes = [BrokenProbe(), good]
    s.start()
    time.sleep(0.15)
    s.stop()
    assert good.calls >= 2
    # good probe still wrote rows despite broken sibling
    rows = temp_db.samples_for_run(run_id)
    assert any(r["source"] == "good" for r in rows)


def test_sampler_thread_dies_cleanly(temp_db):
    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})
    s = Sampler(temp_db, run_id, interval=0.05)
    s._probes = [_StubProbe()]
    s.start()
    s.stop()
    # No daemon thread should still be running
    alive = [t for t in threading.enumerate() if t.name == "benchpress-sampler"]
    assert alive == []
