from __future__ import annotations

import json
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from benchpilot.storage import Storage

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _gather(storage: Storage, run_ids: list[int]) -> dict:
    runs_out: list[dict] = []
    for run_id in run_ids:
        run = storage.get_run(run_id)
        if run is None:
            continue
        benchmarks = storage.benchmarks_for_run(run_id)
        samples = storage.samples_for_run(run_id)
        # Compress samples for the dashboard: pivot to series.
        # series[source][metric][label] = {"t": [...], "v": [...], "unit": ...}
        series: dict = {}
        for s in samples:
            src = s["source"]
            met = s["metric"]
            lbl = s["label"] or ""
            unit = s["unit"] or ""
            t = s["ts"]
            v = s["value"]
            series.setdefault(src, {}).setdefault(met, {}).setdefault(lbl, {"t": [], "v": [], "unit": unit})
            series[src][met][lbl]["t"].append(t)
            series[src][met][lbl]["v"].append(v)
        runs_out.append({
            "id": run["id"],
            "started_at": run["started_at"],
            "ended_at": run["ended_at"],
            "label": run["label"],
            "mode": run["mode"],
            "hostname": run["hostname"],
            "system_info": run.get("system_info") or {},
            "benchmarks": [
                {
                    "id": b["id"],
                    "name": b["name"],
                    "component": b["component"],
                    "status": b["status"],
                    "started_at": b["started_at"],
                    "ended_at": b["ended_at"],
                    "params": b.get("params") or {},
                    "results": b.get("results") or {},
                    "error": b.get("error"),
                }
                for b in benchmarks
            ],
            "series": series,
        })
    return {"runs": runs_out, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def generate_report(*, db_path: Path, out_dir: Path, only_run_ids: list[int] | None = None) -> Path:
    storage = Storage(db_path)
    if only_run_ids is None:
        only_run_ids = [r["id"] for r in storage.list_runs()]
    data = _gather(storage, only_run_ids)
    storage.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    tpl = env.get_template("dashboard.html.j2")
    html = tpl.render(
        data_json=json.dumps(data),
        generated_at=data["generated_at"],
        run_count=len(data["runs"]),
    )
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"benchpilot-report-{ts}.html"
    out.write_text(html)
    # Also write a stable "latest.html" symlink-like copy
    (out_dir / "latest.html").write_text(html)
    return out
