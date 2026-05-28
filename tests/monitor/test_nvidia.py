from benchpilot.monitor.nvidia import NvidiaProbe, _to_float


def test_to_float_handles_na_sentinels():
    assert _to_float("3.14") == 3.14
    assert _to_float(" 42 ") == 42.0
    assert _to_float("") is None
    assert _to_float("[N/A]") is None
    assert _to_float("[Not Supported]") is None
    assert _to_float("not-a-number") is None


CSV_SINGLE_GPU = (
    "0, 65, 88, 70, 8192, 16303, 250.5, 2415, 9501, 2415, 45, P0\n"
)


def test_sample_emits_all_metrics_for_single_gpu(mock_check_output):
    mock_check_output(CSV_SINGLE_GPU)
    rows = NvidiaProbe().sample(ts=1.0)
    # Every row is (ts, source, metric, label, value, unit)
    assert all(r[0] == 1.0 and r[1] == "nvidia" and r[3] == "gpu0" for r in rows)
    metrics = {r[2]: (r[4], r[5]) for r in rows}
    assert metrics["temp"] == (65.0, "C")
    assert metrics["util_gpu"] == (88.0, "%")
    assert metrics["util_mem"] == (70.0, "%")
    assert metrics["mem_used"] == (8192.0, "MiB")
    assert metrics["mem_total"] == (16303.0, "MiB")
    assert metrics["power"] == (250.5, "W")
    assert metrics["clock_graphics"] == (2415.0, "MHz")
    assert metrics["clock_memory"] == (9501.0, "MHz")
    assert metrics["clock_sm"] == (2415.0, "MHz")
    assert metrics["fan"] == (45.0, "%")


def test_sample_returns_empty_when_nvidia_smi_missing(mock_check_output):
    mock_check_output(FileNotFoundError("nvidia-smi"))
    assert NvidiaProbe().sample(ts=1.0) == []


def test_sample_skips_unparseable_index(mock_check_output):
    # First row has non-integer index, second is valid
    mock_check_output("notanint, 60, 50, 50, 1, 2, 100, 1500, 7000, 1500, 30, P0\n"
                     "0, 60, 50, 50, 1, 2, 100, 1500, 7000, 1500, 30, P0\n")
    rows = NvidiaProbe().sample(ts=1.0)
    assert {r[3] for r in rows} == {"gpu0"}


def test_sample_skips_na_metrics(mock_check_output):
    # power=[N/A] should be omitted from emitted rows
    mock_check_output("0, 60, 50, 50, 1, 2, [N/A], 1500, 7000, 1500, 30, P0\n")
    rows = NvidiaProbe().sample(ts=1.0)
    assert all(r[2] != "power" for r in rows)
    assert any(r[2] == "temp" for r in rows)
