import pytest

from benchpress.monitor import cpu as cpu_mod
from benchpress.monitor.cpu import CpuProbe


# Deltas designed for clean assertions:
#   cpu0: idle Δ=50, total Δ=100  -> util = 50%
#   cpu1: idle Δ=75, total Δ=100  -> util = 25%
#   cpu (aggregate): idle Δ=125, total Δ=200 -> util = 37.5%
PROC_STAT_T1 = """\
cpu  0 0 0 0 0 0 0 0 0 0
cpu0 0 0 0 0 0 0 0 0 0 0
cpu1 0 0 0 0 0 0 0 0 0 0
intr 12345
"""

PROC_STAT_T2 = """\
cpu  75 0 0 125 0 0 0 0 0 0
cpu0 50 0 0 50  0 0 0 0 0 0
cpu1 25 0 0 75  0 0 0 0 0 0
intr 12345
"""


CPUINFO = """\
processor	: 0
model name	: 12th Gen Intel(R) Core(TM) i9-12900K
cpu MHz		: 3200.000
cache size	: 30720 KB

processor	: 1
model name	: 12th Gen Intel(R) Core(TM) i9-12900K
cpu MHz		: 3201.500
"""


@pytest.fixture
def fake_proc(tmp_path, monkeypatch):
    stat = tmp_path / "proc_stat"
    info = tmp_path / "proc_cpuinfo"
    monkeypatch.setattr(cpu_mod, "_PROC_STAT", stat)
    monkeypatch.setattr(cpu_mod, "_CPUINFO", info)
    return stat, info


def test_first_sample_produces_no_util(fake_proc):
    stat, info = fake_proc
    stat.write_text(PROC_STAT_T1)
    info.write_text(CPUINFO)
    rows = CpuProbe().sample(ts=1.0)
    # First call has no prior baseline → no util rows. Only clock rows.
    util_rows = [r for r in rows if r[2] == "util"]
    assert util_rows == []
    clock_rows = [r for r in rows if r[2] == "clock"]
    assert sorted(c[3] for c in clock_rows) == ["cpu0", "cpu1"]


def test_second_sample_computes_delta_util(fake_proc):
    stat, info = fake_proc
    info.write_text(CPUINFO)
    p = CpuProbe()
    stat.write_text(PROC_STAT_T1)
    p.sample(ts=1.0)
    stat.write_text(PROC_STAT_T2)
    rows = p.sample(ts=2.0)
    util_rows = {r[3]: r[4] for r in rows if r[2] == "util"}
    assert util_rows["cpu0"] == pytest.approx(50.0)
    assert util_rows["cpu1"] == pytest.approx(25.0)
    assert util_rows["cpu"] == pytest.approx(37.5)


def test_clock_parsing(fake_proc):
    stat, info = fake_proc
    stat.write_text(PROC_STAT_T1)
    info.write_text(CPUINFO)
    rows = CpuProbe().sample(ts=1.0)
    clocks = {r[3]: r[4] for r in rows if r[2] == "clock"}
    assert clocks["cpu0"] == 3200.0
    assert clocks["cpu1"] == 3201.5


def test_missing_cpuinfo_still_emits_util(fake_proc):
    stat, info = fake_proc
    stat.write_text(PROC_STAT_T1)
    # info does NOT exist
    p = CpuProbe()
    p.sample(ts=1.0)
    stat.write_text(PROC_STAT_T2)
    rows = p.sample(ts=2.0)
    assert any(r[2] == "util" for r in rows)
    assert all(r[2] != "clock" for r in rows)
