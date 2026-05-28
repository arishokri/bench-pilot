import json

import pytest

from benchpilot.monitor.disk import DiskProbe


@pytest.fixture
def patched_disk_probe(monkeypatch, mock_check_output):
    """Force DiskProbe to discover a fixed set of NVMe devices and route smartctl
    output through mock_check_output."""
    from benchpilot.monitor import disk as disk_mod

    def fake_devices(_self=None):
        return None  # we'll set after construction

    monkeypatch.setattr(disk_mod, "_list_nvme_devices", lambda: ["/dev/nvme0n1", "/dev/nvme1n1"])
    # smartctl probe also gates on shutil.which("smartctl"); make it appear present
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    return mock_check_output


def _smart_payload(temp=42, spare=99, used=1, extra_sensors=None):
    return json.dumps({
        "nvme_smart_health_information_log": {
            "temperature": temp,
            "available_spare": spare,
            "percentage_used": used,
            **({"temperature_sensors": extra_sensors} if extra_sensors is not None else {}),
        }
    })


def test_emits_temp_spare_wear_per_device(patched_disk_probe):
    payload = _smart_payload(temp=45, spare=98, used=2)
    patched_disk_probe(payload)
    rows = DiskProbe().sample(ts=1.0)
    # Two devices, three metrics each
    by_label_metric = {(r[3], r[2]): (r[4], r[5]) for r in rows}
    assert by_label_metric[("nvme0n1", "temp")] == (45.0, "C")
    assert by_label_metric[("nvme0n1", "spare")] == (98.0, "%")
    assert by_label_metric[("nvme0n1", "wear")] == (2.0, "%")
    assert by_label_metric[("nvme1n1", "temp")] == (45.0, "C")


def test_extra_sensor_temperatures(patched_disk_probe):
    patched_disk_probe(_smart_payload(extra_sensors=[40, 50]))
    rows = DiskProbe().sample(ts=1.0)
    extras = [r for r in rows if r[3].endswith(":s0") or r[3].endswith(":s1")]
    assert {r[3] for r in extras} == {"nvme0n1:s0", "nvme0n1:s1", "nvme1n1:s0", "nvme1n1:s1"}
    assert {r[4] for r in extras} == {40.0, 50.0}


def test_malformed_json_is_skipped(patched_disk_probe):
    patched_disk_probe("not valid json")
    assert DiskProbe().sample(ts=1.0) == []


def test_missing_smart_fields_are_omitted(patched_disk_probe):
    patched_disk_probe(json.dumps({"nvme_smart_health_information_log": {}}))
    assert DiskProbe().sample(ts=1.0) == []
