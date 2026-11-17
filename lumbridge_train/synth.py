"""Synthetic examples: a teacher writes made-up inputs for rare answers, then labels them blind.

Two passes keep the data honest. The writer is asked for an input of a given kind; a labeler
(the same or another teacher) then labels it without seeing the intended kind. Only rows where
the two agree are kept, so the student learns from inputs that really read like their label.
"""

from __future__ import annotations

import collections
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

from .config import Job
from .sources import Example, _coerce
from .teacher import chat, parse_json

WRITE_PROMPT = """Write one realistic but entirely made-up example input for this task.

Task: {instructions}

The correct answer for your example must be {field} = "{choice}"{desc}.
Invent every name, company, address, number and link. Never use real people or real accounts.
Vary length, tone and wording; style hint: {hint}.
Return only the input text itself, no commentary, no labels."""

HINTS = [
    "short and terse", "long and detailed", "casual", "formal", "automated system message",
    "with a list", "with a date or deadline", "from a company", "from a person", "slightly ambiguous",
]


def label(job: Job, teacher_name: str, text: str) -> dict | None:
    t = job.teachers[teacher_name]
    reply = chat(t, [{"role": "system", "content": job.system_prompt()}, {"role": "user", "content": text}],
                 temperature=0.0, max_tokens=200)
    raw = parse_json(reply)
    return _coerce(job, raw) if raw else None


def _one(job: Job, writer: str, labeler: str, choice: str, seed: int) -> Example | None:
    f = job.primary
    rnd = random.Random(seed)
    desc = f.descriptions.get(choice)
    prompt = WRITE_PROMPT.format(
        instructions=job.instructions.strip(), field=f.name, choice=choice,
        desc=f" ({desc})" if desc else "", hint=rnd.choice(HINTS),
    )
    try:
        text = chat(job.teachers[writer], [{"role": "user", "content": prompt}], temperature=1.0, max_tokens=500)
        text = text.strip().strip("`").strip()
        if len(text) < 20:
            return None
        labels = label(job, labeler, text)
    except Exception as e:  # a teacher hiccup costs one row, not the run
        print(f"  synth: {choice}: {type(e).__name__}: {e}")
        return None
    if not labels or labels.get(f.name) != choice:
        return None
    return Example(input=text, labels=labels, ts=time.time(), origin=f"synth:{writer}>{labeler}")


def _cache_path(job: Job):
    return job.workdir / "synth-cache.jsonl"


def load_cache(job: Job) -> list[Example]:
    """Synthetic rows made by earlier runs. They are reused, so a stopped run loses nothing."""
    p = _cache_path(job)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        d = json.loads(line)
        labels = _coerce(job, d["labels"])
        if labels:
            out.append(Example(d["input"], labels, d.get("ts", 0), d.get("origin", "synth")))
    return out


def top_up(job: Job, real: list[Example]) -> list[Example]:
    cfg = job.synth
    if not cfg or not cfg.get("enabled", True):
        return []
    f = job.primary
    cached = load_cache(job)
    if cached:
        print(f"synth: {len(cached)} cached rows from earlier runs")
    writer = cfg["writer"]
    labeler = cfg.get("labeler", writer)
    target = int(cfg.get("per_choice_min", 0))
    max_tries = float(cfg.get("max_tries_factor", 2.5))
    have = collections.Counter(e.labels[f.name] for e in real + cached if not e.eval_only)
    wanted = {c: max(0, target - have[c]) for c in f.choices if c not in cfg.get("skip_choices", [])}
    wanted = {c: n for c, n in wanted.items() if n}
    if not wanted:
        return cached
    print(f"synth: topping up {wanted} with writer={writer} labeler={labeler}")
    jobs = [(c, i) for c, n in wanted.items() for i in range(int(n * max_tries))]
    out: list[Example] = list(cached)
    got = collections.Counter()
    job.workdir.mkdir(parents=True, exist_ok=True)
    cache = open(_cache_path(job), "a")
    conc = job.teachers[writer].concurrency
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(_one, job, writer, labeler, c, hash((job.name, c, i))) for c, i in jobs]
        for fut, (c, _) in zip(futs, jobs):
            if got[c] >= wanted[c]:
                fut.cancel()
                continue
            ex = fut.result()
            if ex is not None and got[c] < wanted[c]:
                out.append(ex)
                got[c] += 1
                cache.write(json.dumps({"input": ex.input, "labels": ex.labels, "ts": ex.ts, "origin": ex.origin}) + "\n")
                cache.flush()
                if sum(got.values()) % 10 == 0:
                    print(f"synth: {dict(got)}", flush=True)
    cache.close()
    print(f"synth: kept {dict(got)} of wanted {wanted}")
    return out
