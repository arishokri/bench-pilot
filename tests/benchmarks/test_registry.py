from benchpilot.benchmarks.cpu import SysbenchCpu
from benchpilot.benchmarks.gpu.tensor import MatmulSuite
from benchpilot.benchmarks.ram import MbwBench, StreamBench, SysbenchMemory
from benchpilot.benchmarks.registry import build_benchmark_plan, build_stress_plan
from benchpilot.benchmarks.ssd import FioRandRead, FioRandWrite, FioSeqRead, FioSeqWrite
from benchpilot.benchmarks.stress import CpuStress, GpuStress, RamStress, SsdStress
from benchpilot.config import RunConfig, StressConfig


def _names(plan):
    return [b.name for b in plan]


def test_build_benchmark_plan_cpu_only():
    plan = build_benchmark_plan(RunConfig(components=("cpu",)))
    assert all(b.component == "cpu" for b in plan)
    assert isinstance(plan[0], SysbenchCpu)


def test_build_benchmark_plan_ram_class_order():
    plan = build_benchmark_plan(RunConfig(components=("ram",)))
    classes = [type(b) for b in plan]
    assert classes == [MbwBench, StreamBench, SysbenchMemory, SysbenchMemory]
    assert plan[2].operation == "read"
    assert plan[3].operation == "write"


def test_build_benchmark_plan_ssd_class_order():
    plan = build_benchmark_plan(RunConfig(components=("ssd",)))
    classes = [type(b) for b in plan]
    assert classes == [FioSeqRead, FioSeqWrite, FioRandRead, FioRandWrite]


def test_full_mode_lengthens_cpu_ram_ssd_runtimes():
    quick = build_benchmark_plan(RunConfig(components=("cpu", "ram", "ssd"), quick=True))
    full = build_benchmark_plan(RunConfig(components=("cpu", "ram", "ssd"), quick=False))
    # sysbench cpu seconds
    cpu_quick = next(b for b in quick if isinstance(b, SysbenchCpu))
    cpu_full = next(b for b in full if isinstance(b, SysbenchCpu))
    assert cpu_full.seconds > cpu_quick.seconds
    # fio runtime
    fio_quick = next(b for b in quick if isinstance(b, FioRandRead))
    fio_full = next(b for b in full if isinstance(b, FioRandRead))
    assert fio_full.runtime > fio_quick.runtime


def test_quick_vs_full_changes_gpu_budgets():
    quick = build_benchmark_plan(RunConfig(components=("gpu",), quick=True))
    full = build_benchmark_plan(RunConfig(components=("gpu",), quick=False))
    mm_quick = next(b for b in quick if isinstance(b, MatmulSuite))
    mm_full = next(b for b in full if isinstance(b, MatmulSuite))
    assert mm_full.seconds_per_dtype > mm_quick.seconds_per_dtype


def test_image_gen_skipped_when_flag_off():
    plan = build_benchmark_plan(RunConfig(components=("gpu",), include_image_gen=False))
    assert all("image_gen" not in b.name for b in plan)


def test_build_stress_plan_one_per_component(tmp_path):
    cfg = StressConfig(components=("cpu", "ram", "ssd", "gpu"),
                       duration_seconds=42, ssd_target_dir=tmp_path)
    plan = build_stress_plan(cfg)
    assert [type(b) for b in plan] == [CpuStress, RamStress, SsdStress, GpuStress]
    assert all(getattr(b, "duration_seconds", None) == 42 for b in plan)
