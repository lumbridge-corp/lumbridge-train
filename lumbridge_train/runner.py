"""The learning loop: collect → synthesize → train → eval gate → promote.

A run lives in `<workdir>/runs/<run-id>/`. `<workdir>/current` points at the promoted weights,
and `<workdir>/promotions.jsonl` records every decision, so a rollback is re-pointing a link.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from . import cloud, dataset, sources, synth, train
from .config import DAYS, Job


def new_run_dir(job: Job) -> Path:
    rid = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = job.workdir / "runs" / rid
    d.mkdir(parents=True, exist_ok=False)
    return d


def build(job: Job, run_dir: Path) -> dict:
    real = dataset.dedupe(sources.collect(job))
    print(f"build: {len(real)} real examples ({sum(e.eval_only for e in real)} eval-only)")
    tr, ev = dataset.split(job, real)
    extra = synth.top_up(job, tr)
    if not ev:
        # no real data yet (a fresh install): hold out part of the synthetic set as the eval set
        extra, ev = dataset.split(job, extra)
    tr = dataset.balance(job, dataset.dedupe(tr + extra))
    stats = dataset.write(job, run_dir, tr, ev)
    print("build:", json.dumps(stats))
    return stats


def current(job: Job) -> Path | None:
    link = job.workdir / "current"
    return link.resolve() if link.exists() else None


def gate(job: Job, base: dict, new: dict) -> tuple[bool, str]:
    metric = job.gate.get("metric", f"{job.primary.name}_acc")
    min_gain = float(job.gate.get("min_gain", 0.0))
    min_valid = float(job.gate.get("min_valid_json", 0.98))
    gain = new[metric] - base[metric]
    if new["valid_json"] < min_valid:
        return False, f"valid_json {new['valid_json']} < {min_valid}"
    if gain <= min_gain:
        return False, f"{metric} {new[metric]} vs {base[metric]} (gain {gain:+.4f} <= {min_gain})"
    return True, f"{metric} {new[metric]} vs {base[metric]} (gain {gain:+.4f})"


def promote(job: Job, run_dir: Path, weights: Path, reason: str, scores: dict) -> None:
    link = job.workdir / "current"
    tmp = job.workdir / ".current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(weights)
    os.replace(tmp, link)
    _record(job, {"action": "promote", "run": run_dir.name, "weights": str(weights), "reason": reason, **scores})


def _record(job: Job, entry: dict) -> None:
    entry = {"at": dt.datetime.now().isoformat(timespec="seconds"), "job": job.name, **entry}
    with open(job.workdir / "promotions.jsonl", "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def run(job: Job, *, promote_ok: bool = True) -> dict:
    run_dir = new_run_dir(job)
    print(f"run: {run_dir}")
    stats = build(job, run_dir)
    eval_file = run_dir / "eval.jsonl"
    baseline_model = str(current(job) or job.train["student"])
    if cloud.wanted(job):
        # rented GPU: train and both evals run there; weights and scores come back here
        weights, base, new = cloud.train_and_eval(job, run_dir, baseline_model)
    else:
        base = train.run_eval(job, baseline_model, eval_file, run_dir / "eval-baseline.json")
        weights = train.run_sft(job, run_dir)
        new = train.run_eval(job, str(weights), eval_file, run_dir / "eval-trained.json")
    ok, reason = gate(job, base, new)
    metric = job.gate.get("metric", f"{job.primary.name}_acc")
    scores = {"baseline": baseline_model, "baseline_score": base[metric], "trained_score": new[metric],
              "metric": metric, "data": stats}
    if ok and promote_ok:
        promote(job, run_dir, weights, reason, scores)
    else:
        _record(job, {"action": "reject" if not ok else "hold", "run": run_dir.name, "reason": reason, **scores})
    report = {"run": str(run_dir), "promoted": ok and promote_ok, "gate": reason, **scores}
    (run_dir / "report.json").write_text(json.dumps(report, indent=1))
    print("run:", json.dumps(report, indent=1))
    return report


def due(job: Job, today: dt.date | None = None) -> bool:
    today = today or dt.date.today()
    return DAYS[today.weekday()] == job.schedule["training_day"]
