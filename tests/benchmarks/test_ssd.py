import json

from benchpress.benchmarks.ssd import (
    FioRandRead,
    FioSeqRead,
    FioSeqWrite,
    _extract,
    cleanup_fio_files,
)


def test_extract_pulls_iops_bw_latency():
    job = {
        "read": {
            "iops": 250000.0,
            "bw_bytes": 7 * 1024 * 1024 * 1024 + 0,
            "clat_ns": {"mean": 12345.0, "percentile": {"99.000000": 99999.0}},
        }
    }
    out = _extract(job, "read")
    assert out["iops"] == 250000.0
    assert abs(out["bw_mib_s"] - 7168.0) < 0.5
    assert out["lat_ns_mean"] == 12345.0
    assert out["lat_ns_p99"] == 99999.0


def test_extract_handles_missing_percentile():
    job = {"write": {"iops": 1.0, "bw_bytes": 1024 * 1024, "clat_ns": {"mean": 1.0}}}
    out = _extract(job, "write")
    assert out["lat_ns_p99"] is None


def _fio_json(rw: str, iops: float, bw_bytes: int) -> str:
    side = "read" if "read" in rw else "write"
    return json.dumps({
        "jobs": [{side: {"iops": iops, "bw_bytes": bw_bytes,
                          "clat_ns": {"mean": 5000, "percentile": {"99.000000": 80000}}}}]
    })


def test_fio_seq_read_constructs_expected_cmdline(mock_shell, tmp_path):
    captured, set_result = mock_shell
    set_result(stdout=_fio_json("read", iops=12000, bw_bytes=4 * 1024 * 1024 * 1024))
    r = FioSeqRead(target_dir=tmp_path, runtime=10, size="2G", bs="1M",
                   iodepth=32, numjobs=1).run().results
    assert "--rw=read" in captured.cmdline
    assert "--bs=1M" in captured.cmdline
    assert "--size=2G" in captured.cmdline
    assert f"--directory={tmp_path}" in captured.cmdline
    assert "--ioengine=libaio" in captured.cmdline
    assert "--direct=1" in captured.cmdline
    assert r["iops"] == 12000
    assert abs(r["bw_mib_s"] - 4096.0) < 0.5


def test_fio_seq_write_uses_write_op(mock_shell, tmp_path):
    captured, set_result = mock_shell
    set_result(stdout=_fio_json("write", iops=10, bw_bytes=2 * 1024 * 1024))
    FioSeqWrite(target_dir=tmp_path).run()
    assert "--rw=write" in captured.cmdline


def test_fio_rand_read_uses_randread(mock_shell, tmp_path):
    captured, set_result = mock_shell
    set_result(stdout=_fio_json("randread", iops=200000, bw_bytes=800 * 1024 * 1024))
    FioRandRead(target_dir=tmp_path).run()
    assert "--rw=randread" in captured.cmdline


def test_fio_tolerates_pre_json_noise(mock_shell, tmp_path):
    """fio sometimes prints warnings before the JSON brace; parser should strip them."""
    captured, set_result = mock_shell
    body = _fio_json("read", iops=1, bw_bytes=1024)
    set_result(stdout="WARN: io_uring not supported, falling back\n" + body)
    r = FioSeqRead(target_dir=tmp_path).run().results
    assert r["iops"] == 1


def test_cleanup_fio_files_removes_scratch(tmp_path):
    (tmp_path / "fio.bench.tmp").write_text("scratch")
    (tmp_path / "fio.bench.tmp.0.0").write_text("scratch")
    (tmp_path / "keep_me.txt").write_text("not scratch")
    cleanup_fio_files(tmp_path)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["keep_me.txt"]


def test_cleanup_fio_files_no_target_dir_is_safe(tmp_path):
    cleanup_fio_files(tmp_path / "does-not-exist")  # must not raise
