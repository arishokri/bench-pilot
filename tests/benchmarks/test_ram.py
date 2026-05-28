from benchpress.benchmarks.ram import MbwBench, StreamBench, SysbenchMemory


MBW_OUTPUT = """\
0	Method: MEMCPY	Elapsed: 0.05421	MiB: 1024	Copy: 18890.45 MiB/s
1	Method: MEMCPY	Elapsed: 0.05418	MiB: 1024	Copy: 18901.31 MiB/s
2	Method: MEMCPY	Elapsed: 0.05408	MiB: 1024	Copy: 18935.20 MiB/s
AVG	Method: MEMCPY	Elapsed: 0.05416	MiB: 1024	Copy: 18908.99 MiB/s
0	Method: DUMB	Elapsed: 0.08412	MiB: 1024	Copy: 12175.20 MiB/s
AVG	Method: DUMB	Elapsed: 0.08412	MiB: 1024	Copy: 12175.20 MiB/s
AVG	Method: MCBLOCK	Elapsed: 0.04901	MiB: 1024	Copy: 20893.40 MiB/s
"""


def test_mbw_parser_captures_each_method(mock_shell):
    _captured, set_result = mock_shell
    set_result(stdout=MBW_OUTPUT)
    r = MbwBench(size_mib=1024, iterations=3).run().results
    bw = r["bandwidth_mib_s"]
    assert bw["memcpy"] == 18908.99
    assert bw["dumb"] == 12175.20
    assert bw["mcblock"] == 20893.40


STREAM_OUTPUT = """\
-------------------------------------------------------------
STREAM version $Revision: 5.10 $
-------------------------------------------------------------
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:           42345.6     0.000452     0.000378     0.000691
Scale:          41234.5     0.000463     0.000388     0.000702
Add:            38123.4     0.000625     0.000498     0.000800
Triad:          39234.7     0.000610     0.000487     0.000792
-------------------------------------------------------------
"""


def test_stream_parser(monkeypatch, mock_shell):
    # StreamBench uses shutil.which then subprocess directly via _shell.run
    _captured, set_result = mock_shell
    # Ensure StreamBench finds a path so it proceeds to the parser
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/stream")
    set_result(stdout=STREAM_OUTPUT)
    r = StreamBench().run().results
    assert r["copy_mb_s"] == 42345.6
    assert r["scale_mb_s"] == 41234.5
    assert r["add_mb_s"] == 38123.4
    assert r["triad_mb_s"] == 39234.7


def test_stream_skipped_when_binary_missing(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    r = StreamBench().run().results
    assert "skipped" in r


SYSBENCH_MEMORY_OUTPUT = """\
sysbench 1.0.20

Running the test with following options:
Number of threads: 24

8192.00 MiB transferred (16384.50 MiB/sec)

General statistics:
    total time:                          0.5000s
"""


def test_sysbench_memory_parser_and_cmdline(mock_shell):
    captured, set_result = mock_shell
    set_result(stdout=SYSBENCH_MEMORY_OUTPUT)
    r = SysbenchMemory(threads=4, total_gib=8, operation="read").run().results
    assert r["transferred_mib"] == 8192.00
    assert r["bandwidth_mib_s"] == 16384.50
    assert r["wall_seconds"] == 0.5
    assert "--memory-oper=read" in captured.cmdline
    assert "--memory-total-size=8G" in captured.cmdline
    assert "--threads=4" in captured.cmdline
