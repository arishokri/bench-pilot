import json
import re

from benchpress.report.generator import generate_report
from benchpress.storage import Storage


def _seed_run(db: Storage, *, label: str) -> int:
    run_id = db.start_run(mode="bench", label=label, hostname="h",
                          system_info={"cpu": "Test CPU"})
    bid = db.start_benchmark(run_id, name="cpu.bench", component="cpu",
                              params={"threads": 4})
    db.finish_benchmark(bid, status="ok", results={"events_per_second": 1234.5})
    bid2 = db.start_benchmark(run_id, name="gpu.bench", component="gpu", params={})
    db.finish_benchmark(bid2, status="skipped", results={"skipped": "no CUDA"})

    samples = [
        # GPU temp
        (run_id, 1.0, "nvidia", "temp", "gpu0", 50.0, "C"),
        (run_id, 2.0, "nvidia", "temp", "gpu0", 55.0, "C"),
        (run_id, 3.0, "nvidia", "temp", "gpu0", 60.0, "C"),
        # GPU power
        (run_id, 1.0, "nvidia", "power", "gpu0", 100.0, "W"),
        (run_id, 2.0, "nvidia", "power", "gpu0", 120.0, "W"),
        # CPU util (per-core)
        (run_id, 1.0, "cpu", "util", "cpu0", 5.0, "%"),
        (run_id, 2.0, "cpu", "util", "cpu0", 90.0, "%"),
    ]
    db.insert_samples(samples)
    db.end_run(run_id)
    return run_id


def _extract_data_json(html: str) -> dict:
    m = re.search(r'<script id="bench-data" type="application/json">(.+?)</script>',
                  html, re.DOTALL)
    assert m is not None, "report missing bench-data script block"
    return json.loads(m.group(1))


def test_generate_report_writes_html_and_latest(tmp_path):
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    _seed_run(storage, label="t")
    storage.close()

    out_dir = tmp_path / "reports"
    out = generate_report(db_path=db_path, out_dir=out_dir)
    assert out.exists()
    assert (out_dir / "latest.html").exists()
    # The bench-data script must be present
    html = out.read_text()
    assert '<script id="bench-data"' in html


def test_report_json_payload_shape(tmp_path):
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    run_id = _seed_run(storage, label="payload")
    storage.close()

    out = generate_report(db_path=db_path, out_dir=tmp_path / "reports")
    data = _extract_data_json(out.read_text())

    assert len(data["runs"]) == 1
    run = data["runs"][0]
    assert run["id"] == run_id
    assert run["label"] == "payload"

    # Benchmarks survive with their structured results
    by_name = {b["name"]: b for b in run["benchmarks"]}
    assert by_name["cpu.bench"]["status"] == "ok"
    assert by_name["cpu.bench"]["results"] == {"events_per_second": 1234.5}
    assert by_name["gpu.bench"]["status"] == "skipped"

    # Series pivoted to source -> metric -> label -> {t, v, unit}
    nv = run["series"]["nvidia"]
    assert nv["temp"]["gpu0"]["t"] == [1.0, 2.0, 3.0]
    assert nv["temp"]["gpu0"]["v"] == [50.0, 55.0, 60.0]
    assert nv["temp"]["gpu0"]["unit"] == "C"
    assert nv["power"]["gpu0"]["v"] == [100.0, 120.0]

    assert run["series"]["cpu"]["util"]["cpu0"]["v"] == [5.0, 90.0]


def test_generate_report_only_subset(tmp_path):
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    rid1 = _seed_run(storage, label="r1")
    rid2 = _seed_run(storage, label="r2")
    storage.close()

    out = generate_report(db_path=db_path, out_dir=tmp_path / "reports",
                          only_run_ids=[rid2])
    data = _extract_data_json(out.read_text())
    assert [r["id"] for r in data["runs"]] == [rid2]
