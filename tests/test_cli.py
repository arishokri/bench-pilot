import pytest
from typer.testing import CliRunner

import benchpress.cli as cli_mod
from benchpress.cli import _parse_components, _parse_duration, app


runner = CliRunner()


# ---------- _parse_components ----------
def test_parse_components_default_is_all():
    from benchpress.config import COMPONENTS
    assert _parse_components(None) == COMPONENTS


def test_parse_components_list():
    assert _parse_components("cpu,gpu") == ("cpu", "gpu")


def test_parse_components_strips_whitespace():
    assert _parse_components(" cpu , ram ") == ("cpu", "ram")


def test_parse_components_rejects_unknown():
    import typer
    with pytest.raises(typer.BadParameter):
        _parse_components("cpu,fpga")


# ---------- _parse_duration ----------
@pytest.mark.parametrize("inp,expected", [
    ("90", 90),
    ("90s", 90),
    ("5m", 300),
    ("1h", 3600),
])
def test_parse_duration_valid(inp, expected):
    assert _parse_duration(inp) == expected


def test_parse_duration_rejects_bad_input():
    import typer
    with pytest.raises(typer.BadParameter):
        _parse_duration("not-a-number")


# ---------- CLI smoke ----------
def test_list_on_empty_db_creates_table(tmp_path):
    r = runner.invoke(app, ["list", "--data-dir", str(tmp_path)])
    assert r.exit_code == 0
    assert "Runs" in r.stdout


def test_report_errors_when_db_missing(tmp_path):
    r = runner.invoke(app, ["report",
                            "--data-dir", str(tmp_path),
                            "--report-dir", str(tmp_path / "reports")])
    assert r.exit_code != 0
    assert "no database" in r.stdout.lower() or "no database" in r.stderr.lower() \
        or "no database" in (r.output or "").lower()


def test_run_invalid_component(tmp_path):
    r = runner.invoke(app, ["run",
                            "--components", "fpga",
                            "--data-dir", str(tmp_path)])
    assert r.exit_code != 0


def test_stress_invalid_duration(tmp_path):
    r = runner.invoke(app, ["stress",
                            "--duration", "nonsense",
                            "--data-dir", str(tmp_path)])
    assert r.exit_code != 0


def test_run_forwards_flags_to_runner(monkeypatch, tmp_path):
    """Patch run_benchmarks and assert the CLI builds the right RunConfig."""
    captured = {}

    def fake_run_benchmarks(cfg, console=None):
        captured["cfg"] = cfg
        from benchpress.runner import RunSummary
        return RunSummary(run_id=1, benchmarks_ok=0, benchmarks_failed=0, benchmarks_skipped=0)

    monkeypatch.setattr(cli_mod, "run_benchmarks", fake_run_benchmarks)
    r = runner.invoke(app, [
        "run",
        "--full",
        "--components", "cpu,gpu",
        "--label", "x",
        "--no-image-gen",
        "--data-dir", str(tmp_path),
    ])
    assert r.exit_code == 0, r.output
    cfg = captured["cfg"]
    assert cfg.quick is False
    assert cfg.components == ("cpu", "gpu")
    assert cfg.label == "x"
    assert cfg.include_image_gen is False
    assert cfg.data_dir == tmp_path
    assert cfg.warmup is True            # on by default
    assert cfg.warmup_seconds is None    # derive from quick/full


def test_run_warmup_flags(monkeypatch, tmp_path):
    """--no-warmup and --warmup-duration reach the RunConfig."""
    captured = {}

    def fake_run_benchmarks(cfg, console=None):
        captured["cfg"] = cfg
        from benchpress.runner import RunSummary
        return RunSummary(run_id=1, benchmarks_ok=0, benchmarks_failed=0, benchmarks_skipped=0)

    monkeypatch.setattr(cli_mod, "run_benchmarks", fake_run_benchmarks)
    r = runner.invoke(app, [
        "run", "--no-warmup", "--warmup-duration", "90s",
        "--data-dir", str(tmp_path),
    ])
    assert r.exit_code == 0, r.output
    assert captured["cfg"].warmup is False
    assert captured["cfg"].warmup_seconds == 90


def test_stress_forwards_flags_to_runner(monkeypatch, tmp_path):
    captured = {}

    def fake_run_stress(cfg, console=None):
        captured["cfg"] = cfg
        from benchpress.runner import RunSummary
        return RunSummary(run_id=1, benchmarks_ok=0, benchmarks_failed=0, benchmarks_skipped=0)

    monkeypatch.setattr(cli_mod, "run_stress", fake_run_stress)
    r = runner.invoke(app, [
        "stress",
        "--duration", "3m",
        "--components", "cpu,gpu",
        "--data-dir", str(tmp_path),
    ])
    assert r.exit_code == 0, r.output
    cfg = captured["cfg"]
    assert cfg.duration_seconds == 180
    assert cfg.components == ("cpu", "gpu")
