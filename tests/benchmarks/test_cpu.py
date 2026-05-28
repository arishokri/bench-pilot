from benchpress.benchmarks.cpu import SysbenchCpu, _parse_sysbench_cpu


SYSBENCH_OUTPUT = """\
sysbench 1.0.20 (using bundled LuaJIT 2.1.0-beta2)

Running the test with following options:
Number of threads: 24
Initializing random number generator from current time

CPU speed:
    events per second: 12345.67

General statistics:
    total time:                          15.0008s
    total number of events:              185190

Latency (ms):
         min:                                    1.20
         avg:                                    1.95
         max:                                    8.10
         95th percentile:                        2.10
         sum:                                  360187.50

Threads fairness:
    events (avg/stddev):           7716.2500/40.32
    execution time (avg/stddev):   15.0078/0.00
"""


def test_parse_sysbench_cpu_extracts_all_known_fields():
    r = _parse_sysbench_cpu(SYSBENCH_OUTPUT)
    assert r["events_per_second"] == 12345.67
    assert r["total_events"] == 185190
    assert r["wall_seconds"] == 15.0008
    assert r["latency_ms_avg"] == 1.95
    assert r["latency_ms_p95"] == 2.10


def test_parse_sysbench_cpu_tolerates_missing_lines():
    r = _parse_sysbench_cpu("nothing useful here")
    assert r == {}


def test_sysbench_cpu_constructs_expected_cmdline(mock_shell):
    captured, set_result = mock_shell
    set_result(stdout=SYSBENCH_OUTPUT)
    SysbenchCpu(threads=4, seconds=10, cpu_max_prime=5000).run()
    assert captured.cmdline[1:] == ["cpu", "--threads=4", "--time=10", "--cpu-max-prime=5000", "run"]
    # `timeout` is run-time + 30s grace
    assert captured.timeout == 10 + 30


def test_sysbench_cpu_threads_defaults_to_cpu_count():
    import os
    b = SysbenchCpu()
    assert b.threads == (os.cpu_count() or 1)
    assert "threads" in b.params() and b.params()["threads"] == b.threads
