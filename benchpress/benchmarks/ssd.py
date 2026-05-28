from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchpress.benchmarks._shell import require, run
from benchpress.benchmarks.base import BenchmarkResult


_JOB_FILES = "fio.bench.tmp"


def _fio_run(target_dir: Path, *, rw: str, bs: str, size: str, iodepth: int, runtime: int, numjobs: int) -> dict:
    target_dir.mkdir(parents=True, exist_ok=True)
    name = f"bp_{rw}_{bs}"
    cmd = [
        require("fio"),
        f"--name={name}",
        f"--directory={target_dir}",
        f"--filename={_JOB_FILES}",
        f"--rw={rw}",
        f"--bs={bs}",
        f"--size={size}",
        f"--iodepth={iodepth}",
        f"--runtime={runtime}",
        "--time_based=1",
        f"--numjobs={numjobs}",
        "--group_reporting=1",
        "--ioengine=libaio",
        "--direct=1",
        "--output-format=json",
    ]
    sh = run(cmd, timeout=runtime + 60)
    try:
        return json.loads(sh.stdout)
    except json.JSONDecodeError:
        # fio sometimes mixes warnings before json — find the brace
        idx = sh.stdout.find("{")
        if idx < 0:
            raise
        return json.loads(sh.stdout[idx:])


def _extract(job: dict, side: str) -> dict:
    s = job[side]
    return {
        "iops": s.get("iops"),
        "bw_mib_s": (s.get("bw_bytes") or 0) / (1024 * 1024),
        "lat_ns_mean": (s.get("clat_ns") or {}).get("mean"),
        "lat_ns_p99": (s.get("clat_ns") or {}).get("percentile", {}).get("99.000000"),
    }


@dataclass
class FioSeqRead:
    name: str = "ssd.fio.seq_read"
    component: str = "ssd"
    target_dir: Path = Path.home() / ".benchpress_fio"
    runtime: int = 10
    size: str = "2G"
    bs: str = "1M"
    iodepth: int = 32
    numjobs: int = 1

    def params(self) -> dict:
        return {"runtime": self.runtime, "size": self.size, "bs": self.bs, "iodepth": self.iodepth, "numjobs": self.numjobs}

    def run(self) -> BenchmarkResult:
        out = _fio_run(self.target_dir, rw="read", bs=self.bs, size=self.size, iodepth=self.iodepth, runtime=self.runtime, numjobs=self.numjobs)
        job = out["jobs"][0]
        return BenchmarkResult(results=_extract(job, "read"))


@dataclass
class FioSeqWrite:
    name: str = "ssd.fio.seq_write"
    component: str = "ssd"
    target_dir: Path = Path.home() / ".benchpress_fio"
    runtime: int = 10
    size: str = "2G"
    bs: str = "1M"
    iodepth: int = 32
    numjobs: int = 1

    def params(self) -> dict:
        return {"runtime": self.runtime, "size": self.size, "bs": self.bs, "iodepth": self.iodepth, "numjobs": self.numjobs}

    def run(self) -> BenchmarkResult:
        out = _fio_run(self.target_dir, rw="write", bs=self.bs, size=self.size, iodepth=self.iodepth, runtime=self.runtime, numjobs=self.numjobs)
        job = out["jobs"][0]
        return BenchmarkResult(results=_extract(job, "write"))


@dataclass
class FioRandRead:
    name: str = "ssd.fio.rand_read"
    component: str = "ssd"
    target_dir: Path = Path.home() / ".benchpress_fio"
    runtime: int = 15
    size: str = "1G"
    bs: str = "4k"
    iodepth: int = 32
    numjobs: int = 4

    def params(self) -> dict:
        return {"runtime": self.runtime, "size": self.size, "bs": self.bs, "iodepth": self.iodepth, "numjobs": self.numjobs}

    def run(self) -> BenchmarkResult:
        out = _fio_run(self.target_dir, rw="randread", bs=self.bs, size=self.size, iodepth=self.iodepth, runtime=self.runtime, numjobs=self.numjobs)
        job = out["jobs"][0]
        return BenchmarkResult(results=_extract(job, "read"))


@dataclass
class FioRandWrite:
    name: str = "ssd.fio.rand_write"
    component: str = "ssd"
    target_dir: Path = Path.home() / ".benchpress_fio"
    runtime: int = 15
    size: str = "1G"
    bs: str = "4k"
    iodepth: int = 32
    numjobs: int = 4

    def params(self) -> dict:
        return {"runtime": self.runtime, "size": self.size, "bs": self.bs, "iodepth": self.iodepth, "numjobs": self.numjobs}

    def run(self) -> BenchmarkResult:
        out = _fio_run(self.target_dir, rw="randwrite", bs=self.bs, size=self.size, iodepth=self.iodepth, runtime=self.runtime, numjobs=self.numjobs)
        job = out["jobs"][0]
        return BenchmarkResult(results=_extract(job, "write"))


def cleanup_fio_files(target_dir: Path) -> None:
    """Best-effort removal of fio scratch files."""
    if not target_dir.exists():
        return
    for p in target_dir.glob(f"{_JOB_FILES}*"):
        try:
            p.unlink()
        except OSError:
            pass
