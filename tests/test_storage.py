import threading

import pytest

from benchpress.storage import Storage


def test_schema_creates_tables(temp_db):
    tables = {r[0] for r in temp_db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {"runs", "benchmarks", "samples"}.issubset(tables)


def test_start_and_end_run_roundtrip(temp_db):
    run_id = temp_db.start_run(mode="bench", label="t",
                                hostname="host", system_info={"cpu": "x"})
    got = temp_db.get_run(run_id)
    assert got["mode"] == "bench"
    assert got["label"] == "t"
    assert got["hostname"] == "host"
    assert got["system_info"] == {"cpu": "x"}
    assert got["ended_at"] is None
    temp_db.end_run(run_id)
    assert temp_db.get_run(run_id)["ended_at"] is not None


def test_list_runs_orders_descending(temp_db):
    ids = [temp_db.start_run(mode="bench", label=f"r{i}", hostname="h", system_info={})
           for i in range(3)]
    runs = temp_db.list_runs()
    assert [r["id"] for r in runs] == list(reversed(ids))


def test_get_run_missing_returns_none(temp_db):
    assert temp_db.get_run(9999) is None


def test_benchmark_lifecycle_ok(temp_db):
    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})
    b = temp_db.start_benchmark(run_id, name="cpu.sysbench", component="cpu",
                                  params={"threads": 4})
    temp_db.finish_benchmark(b, status="ok", results={"events_per_second": 12345})
    rows = temp_db.benchmarks_for_run(run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["params"] == {"threads": 4}
    assert row["results"] == {"events_per_second": 12345}
    assert row["ended_at"] is not None
    assert row["error"] is None


def test_benchmark_lifecycle_skipped(temp_db):
    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})
    b = temp_db.start_benchmark(run_id, name="gpu.x", component="gpu", params={})
    temp_db.finish_benchmark(b, status="skipped", results={"skipped": "no CUDA"})
    row = temp_db.benchmarks_for_run(run_id)[0]
    assert row["status"] == "skipped"
    assert row["results"] == {"skipped": "no CUDA"}


def test_benchmark_lifecycle_error(temp_db):
    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})
    b = temp_db.start_benchmark(run_id, name="cpu.x", component="cpu", params={})
    temp_db.finish_benchmark(b, status="error", error="boom")
    row = temp_db.benchmarks_for_run(run_id)[0]
    assert row["status"] == "error"
    assert row["error"] == "boom"
    # Storage only populates "results" when results_json is non-null
    assert row.get("results") is None
    assert row["results_json"] is None


def test_samples_round_trip_in_ts_order(temp_db):
    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})
    rows = [
        (run_id, 3.0, "nvidia", "temp", "gpu0", 50.0, "C"),
        (run_id, 1.0, "nvidia", "temp", "gpu0", 40.0, "C"),
        (run_id, 2.0, "nvidia", "temp", "gpu0", 45.0, "C"),
    ]
    temp_db.insert_samples(rows)
    got = temp_db.samples_for_run(run_id)
    assert [r["ts"] for r in got] == [1.0, 2.0, 3.0]
    assert [r["value"] for r in got] == [40.0, 45.0, 50.0]


def test_transaction_rollback(temp_db):
    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})
    try:
        with temp_db.transaction():
            temp_db._conn.execute(
                "INSERT INTO benchmarks(run_id, name, component, started_at, status) "
                "VALUES(?,?,?,?,?)",
                (run_id, "rollback.me", "cpu", "now", "running"),
            )
            raise RuntimeError("force rollback")
    except RuntimeError:
        pass
    assert temp_db.benchmarks_for_run(run_id) == []



def test_concurrent_writers_each_get_unique_ids_and_finish_ok(temp_db):
    """Spawn N threads, each starting + finishing K benchmarks. On a race-free
    Storage every row ends with status=ok and ended_at non-null, and every
    benchmark id is unique. On the racy implementation, some rows are stuck
    in 'running' because finish_benchmark UPDATEd the wrong row."""
    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})

    N_THREADS = 6
    K_PER_THREAD = 20
    seen_ids: list[int] = []
    seen_lock = threading.Lock()

    def worker(thread_idx: int):
        for k in range(K_PER_THREAD):
            bid = temp_db.start_benchmark(run_id, name=f"t{thread_idx}.{k}",
                                            component="cpu", params={"t": thread_idx})
            with seen_lock:
                seen_ids.append(bid)
            temp_db.finish_benchmark(bid, status="ok", results={"n": k})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = temp_db.benchmarks_for_run(run_id)
    assert len(rows) == N_THREADS * K_PER_THREAD
    assert len(set(seen_ids)) == N_THREADS * K_PER_THREAD, "duplicate bench_ids from start_benchmark"
    stuck = [r for r in rows if r["status"] != "ok" or r["ended_at"] is None]
    assert stuck == [], f"{len(stuck)} rows stuck mid-state (race)"
