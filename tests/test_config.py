from benchpilot.config import COMPONENTS, RunConfig, StressConfig


def test_runconfig_defaults():
    cfg = RunConfig()
    assert cfg.components == COMPONENTS
    assert cfg.quick is True
    assert cfg.include_image_gen is True
    assert cfg.ollama_models == ()


def test_stressconfig_defaults():
    cfg = StressConfig()
    assert cfg.components == ("cpu",)
    assert cfg.duration_seconds == 120


def test_gpu_budget_quick_vs_full():
    assert RunConfig(quick=True).gpu_budget == 8.0
    assert RunConfig(quick=False).gpu_budget == 20.0


def test_gpu_budget_explicit_overrides_quick():
    assert RunConfig(quick=True, gpu_seconds_per_test=3.0).gpu_budget == 3.0
    assert RunConfig(quick=False, gpu_seconds_per_test=3.0).gpu_budget == 3.0


def test_db_path_derives_from_data_dir(tmp_path):
    cfg = RunConfig(data_dir=tmp_path)
    assert cfg.db_path == tmp_path / "benchpilot.db"
