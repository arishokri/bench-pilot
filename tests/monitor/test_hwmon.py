import pytest

from benchpilot.monitor.hwmon import HwmonProbe


def test_temp_scales_milli_to_celsius(fake_hwmon):
    fake_hwmon({
        "hwmon0": {
            "name": "coretemp",
            "temp1_input": "42500",
            "temp1_label": "Package",
        },
    })
    rows = HwmonProbe().sample(ts=1.0)
    assert len(rows) == 1
    ts, src, metric, label, value, unit = rows[0]
    assert src == "hwmon:coretemp"
    assert metric == "temp"
    assert label == "Package"
    assert value == 42.5
    assert unit == "C"


def test_fan_unscaled_rpm(fake_hwmon):
    fake_hwmon({
        "hwmon0": {
            "name": "nct6798",
            "fan1_input": "1200",
            "fan1_label": "CPU_FAN",
        },
    })
    rows = HwmonProbe().sample(ts=1.0)
    assert rows == [(1.0, "hwmon:nct6798", "rpm", "CPU_FAN", 1200.0, "rpm")]


def test_voltage_scales_milli_to_volts(fake_hwmon):
    fake_hwmon({
        "hwmon0": {
            "name": "nct6798",
            "in1_input": "3300",
        },
    })
    rows = HwmonProbe().sample(ts=1.0)
    ts, src, metric, label, value, unit = rows[0]
    assert (metric, label, unit) == ("voltage", "in1", "V")
    assert value == pytest.approx(3.3)


def test_power_scales_micro_to_watts(fake_hwmon):
    fake_hwmon({
        "hwmon0": {
            "name": "fakecpu",
            "power1_input": "65000000",  # 65 W
        },
    })
    rows = HwmonProbe().sample(ts=1.0)
    assert rows[0][4] == 65.0
    assert rows[0][5] == "W"


def test_default_label_when_no_label_file(fake_hwmon):
    fake_hwmon({"hwmon0": {"name": "x", "temp3_input": "0"}})
    rows = HwmonProbe().sample(ts=1.0)
    assert rows[0][3] == "temp3"


def test_discovery_skips_unknown_kinds(fake_hwmon):
    fake_hwmon({
        "hwmon0": {
            "name": "x",
            "temp1_input": "1000",
            "freq1_input": "9999",       # not a recognised kind
            "weird_input": "42",         # no numeric suffix
        },
    })
    rows = HwmonProbe().sample(ts=1.0)
    metrics = {r[2] for r in rows}
    assert metrics == {"temp"}


def test_no_hwmon_returns_empty(tmp_path, monkeypatch):
    from benchpilot.monitor import hwmon as hwmod
    monkeypatch.setattr(hwmod, "_HWMON", tmp_path / "does-not-exist")
    assert HwmonProbe().sample(ts=1.0) == []
