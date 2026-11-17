"""Turn examples into a frozen eval set and an SFT training set (prompt/completion, TRL format)."""

from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

from .config import Job
from .sources import Example


def answer(job: Job, labels: dict) -> str:
    return json.dumps({f.name: labels[f.name] for f in job.fields}, separators=(",", ":"))


def prompt_messages(job: Job, text: str) -> list[dict]:
    limit = int(job.train.get("max_input_chars", 2000))
    return [{"role": "system", "content": job.system_prompt()}, {"role": "user", "content": text[:limit]}]


def _key(e: Example) -> str:
    return hashlib.sha1(" ".join(e.input.split()).lower().encode()).hexdigest()


def dedupe(rows: list[Example]) -> list[Example]:
    seen, out = set(), []
    for e in rows:
        k = _key(e)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def split(job: Job, real: list[Example]) -> tuple[list[Example], list[Example]]:
    """Newest `eval_fraction` of real rows (by ts) is the eval set; eval_only rows always go to eval."""
    frac = float(job.split.get("eval_fraction", 0.15))
    trainable = sorted([e for e in real if not e.eval_only], key=lambda e: (e.ts, _key(e)))
    cut = int(len(trainable) * (1 - frac))
    train, ev = trainable[:cut], trainable[cut:] + [e for e in real if e.eval_only]
    cap = job.split.get("max_eval")
    if cap and len(ev) > cap:
        ev = random.Random(0).sample(ev, cap)
    return train, ev


def balance(job: Job, train: list[Example]) -> list[Example]:
    """Cap any one answer at `max_per_choice` so a flood of one label does not drown the rest."""
    cap = job.train.get("max_per_choice")
    if not cap:
        return train
    f = job.primary
    by = collections.defaultdict(list)
    for e in train:
        by[e.labels[f.name]].append(e)
    rnd = random.Random(0)
    out = []
    for c, rows in by.items():
        out.extend(rnd.sample(rows, cap) if len(rows) > cap else rows)
    return out


def write(job: Job, out_dir: Path, train: list[Example], ev: list[Example]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_dir = out_dir / "sft"
    sft_dir.mkdir(exist_ok=True)
    rnd = random.Random(0)
    train = train[:]
    rnd.shuffle(train)
    with open(sft_dir / "train.jsonl", "w") as fh:
        for e in train:
            fh.write(json.dumps({
                "prompt": prompt_messages(job, e.input),
                "completion": [{"role": "assistant", "content": answer(job, e.labels)}],
            }) + "\n")
    with open(out_dir / "eval.jsonl", "w") as fh:
        for e in ev:
            fh.write(json.dumps({"input": e.input, "labels": e.labels, "origin": e.origin}) + "\n")
    f = job.primary
    stats = {
        "train": len(train),
        "eval": len(ev),
        "train_by_choice": dict(collections.Counter(e.labels[f.name] for e in train)),
        "eval_by_choice": dict(collections.Counter(e.labels[f.name] for e in ev)),
        "train_by_origin": dict(collections.Counter(e.origin.split(":")[0] for e in train)),
    }
    (out_dir / "data-stats.json").write_text(json.dumps(stats, indent=1))
    return stats
