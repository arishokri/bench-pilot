from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    label TEXT,
    mode TEXT NOT NULL,
    hostname TEXT,
    system_info_json TEXT
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    component TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    params_json TEXT,
    results_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_benchmarks_run ON benchmarks(run_id);

CREATE TABLE IF NOT EXISTS samples (
    run_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    source TEXT NOT NULL,
    metric TEXT NOT NULL,
    label TEXT,
    value REAL NOT NULL,
    unit TEXT
);
CREATE INDEX IF NOT EXISTS idx_samples_run_ts ON samples(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_samples_run_source ON samples(run_id, source, metric);
"""


def _iso(ts: float | None = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


class Storage:
    """Thin SQLite wrapper. One connection per process; WAL for concurrent sampler writes."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ---------- runs ----------
    def start_run(self, *, mode: str, label: str | None, hostname: str, system_info: dict) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs(started_at, label, mode, hostname, system_info_json) VALUES(?,?,?,?,?)",
            (_iso(), label, mode, hostname, json.dumps(system_info)),
        )
        return cur.lastrowid

    def end_run(self, run_id: int) -> None:
        self._conn.execute("UPDATE runs SET ended_at=? WHERE id=?", (_iso(), run_id))

    def list_runs(self) -> list[dict]:
        cur = self._conn.execute(
            "SELECT id, started_at, ended_at, label, mode, hostname FROM runs ORDER BY id DESC"
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_run(self, run_id: int) -> dict | None:
        cur = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [c[0] for c in cur.description]
        d = dict(zip(cols, row))
        if d.get("system_info_json"):
            d["system_info"] = json.loads(d["system_info_json"])
        return d

    # ---------- benchmarks ----------
    def start_benchmark(self, run_id: int, *, name: str, component: str, params: dict) -> int:
        cur = self._conn.execute(
            "INSERT INTO benchmarks(run_id, name, component, started_at, status, params_json) "
            "VALUES(?,?,?,?,?,?)",
            (run_id, name, component, _iso(), "running", json.dumps(params)),
        )
        return cur.lastrowid

    def finish_benchmark(
        self,
        bench_id: int,
        *,
        status: str,
        results: dict | None = None,
        error: str | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE benchmarks SET ended_at=?, status=?, results_json=?, error=? WHERE id=?",
            (_iso(), status, json.dumps(results) if results else None, error, bench_id),
        )

    def benchmarks_for_run(self, run_id: int) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM benchmarks WHERE run_id=? ORDER BY id", (run_id,)
        )
        cols = [c[0] for c in cur.description]
        out: list[dict] = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            for k in ("params_json", "results_json"):
                if d.get(k):
                    d[k.replace("_json", "")] = json.loads(d[k])
            out.append(d)
        return out

    # ---------- samples ----------
    def insert_samples(self, rows: Iterable[tuple]) -> None:
        # rows: (run_id, ts, source, metric, label, value, unit)
        self._conn.executemany(
            "INSERT INTO samples(run_id, ts, source, metric, label, value, unit) VALUES(?,?,?,?,?,?,?)",
            rows,
        )

    def samples_for_run(self, run_id: int) -> list[dict]:
        cur = self._conn.execute(
            "SELECT ts, source, metric, label, value, unit FROM samples WHERE run_id=? ORDER BY ts",
            (run_id,),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    @contextmanager
    def transaction(self):
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
