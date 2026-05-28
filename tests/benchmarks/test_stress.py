from benchpilot.benchmarks.stress import CpuStress, GpuStress, RamStress, SsdStress


def test_cpu_stress_cmdline(mock_shell):
    captured, set_result = mock_shell
    set_result(stderr="...stress-ng output...")
    CpuStress(duration_seconds=30, threads=8).run()
    cmd = captured.cmdline
    assert cmd[0].endswith("stress-ng")
    assert "--cpu=8" in cmd
    assert "--timeout=30s" in cmd
    assert "--cpu-method=all" in cmd
    # timeout for the subprocess.run call is duration+60s
    assert captured.timeout == 90


def test_cpu_stress_threads_default(mock_shell):
    import os
    captured, _set = mock_shell
    CpuStress(duration_seconds=5).run()
    assert f"--cpu={os.cpu_count()}" in captured.cmdline


def test_ram_stress_cmdline(mock_shell):
    captured, _set = mock_shell
    RamStress(duration_seconds=10, workers=2, bytes_per_worker="1G").run()
    cmd = captured.cmdline
    assert "--vm=2" in cmd
    assert "--vm-bytes=1G" in cmd
    assert "--vm-keep" in cmd
    assert "--timeout=10s" in cmd


def test_ssd_stress_cmdline_and_cleanup(mock_shell, tmp_path):
    captured, _set = mock_shell
    # Pre-create a fake leftover scratch file to confirm cleanup pass
    (tmp_path / "stress-ng-leftover").write_text("garbage")
    SsdStress(duration_seconds=15, target_dir=tmp_path, workers=3).run()
    cmd = captured.cmdline
    assert "--hdd=3" in cmd
    assert "--timeout=15s" in cmd
    assert any(arg.startswith("--temp-path=") and str(tmp_path) in arg for arg in cmd)
    # post-run glob deletion removed our pre-seeded scratch file
    assert not (tmp_path / "stress-ng-leftover").exists()


def test_gpu_stress_skipped_without_torch(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("simulated")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = GpuStress(duration_seconds=1).run().results
    assert "skipped" in r and "torch" in r["skipped"]


def test_gpu_stress_skipped_when_no_cuda(monkeypatch):
    import torch  # noqa: F401  — let real import happen
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    r = GpuStress(duration_seconds=1).run().results
    assert r.get("skipped") == "no CUDA device"
