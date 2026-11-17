"""Where real examples come from.

Every source yields Example rows: the input text the model sees, the labels, a timestamp
(for a time-based train/eval split) and where the row came from. Rows whose labels were
written by a model whose license or terms forbid training on its outputs are marked
`eval_only`: they can grade a model but never teach one.
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import sqlite3
import subprocess
from dataclasses import dataclass, field
from typing import Iterator

from .config import Job, expand

# Outputs of these models may not be used to train a model (provider terms). Override per source.
DEFAULT_EVAL_ONLY = ["claude-*", "gpt-*", "o[0-9]*", "gemini-*", "<synthetic>"]

# Set by collect(keep_unlabeled=True): sources also yield rows with no (valid) labels, for labeling.
KEEP_UNLABELED = "_keep_unlabeled"


@dataclass
class Example:
    input: str
    labels: dict
    ts: float = 0.0
    origin: str = ""
    eval_only: bool = False
    meta: dict = field(default_factory=dict)


def _match(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name or "", p) for p in patterns)


def _coerce(job: Job, raw: dict) -> dict | None:
    """Keep only the job's fields, normalised; None if a field is missing or invalid."""
    out = {}
    for f in job.fields:
        v = raw.get(f.name)
        if f.type == "bool":
            if isinstance(v, str):
                v = v.strip().lower() in ("1", "true", "yes")
            elif v is None:
                return None
            out[f.name] = bool(v)
        else:
            if not isinstance(v, str) or v not in f.choices:
                return None
            out[f.name] = v
    return out


def sqlite_source(job: Job, cfg: dict) -> Iterator[Example]:
    """`query` must return columns: input, ts, origin (optional) and one column per job field."""
    db = expand(cfg["path"])
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        for row in con.execute(cfg["query"]):
            d = dict(row)
            labels = _coerce(job, d)
            if not d.get("input") or (labels is None and not cfg.get(KEEP_UNLABELED)):
                continue
            yield Example(
                input=str(d["input"]),
                labels=labels or {},
                ts=float(d.get("ts") or 0),
                origin=f"{cfg.get('name', 'sqlite')}:{d.get('origin') or 'row'}",
            )
    finally:
        con.close()


def jsonl_source(job: Job, cfg: dict) -> Iterator[Example]:
    """Plain JSONL with {"input": ..., "<field>": ..., "ts": ...} per line."""
    for p in sorted(glob.glob(str(expand(cfg["path"])))):
        with open(p) as fh:
            for line in fh:
                d = json.loads(line)
                labels = _coerce(job, d)
                if d.get("input") and (labels is not None or cfg.get(KEEP_UNLABELED)):
                    origin = f"{cfg.get('name', 'jsonl')}:{d['origin']}" if d.get("origin") else cfg.get("name", "jsonl")
                    yield Example(d["input"], labels or {}, float(d.get("ts", 0)), origin)


def _content_text(c) -> str:
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return "" if c is None else str(c)


def _maybe_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return x
    return x


def langfuse_archive_source(job: Job, cfg: dict) -> Iterator[Example]:
    """Langfuse v4 event exports as `<root>/<project>/<YYYY-MM>/<date>.jsonl.zst`.

    Keeps GENERATION rows from the listed callers (Langfuse user_id before the '@'), takes the
    last user message as the input and the assistant's JSON reply as the labels. Exports can be
    re-written for the same day, so rows are de-duplicated by span id.
    """
    root = expand(cfg["root"])
    callers = cfg.get("callers")
    eval_only_models = cfg.get("eval_only_models", DEFAULT_EVAL_ONLY)
    seen: set[str] = set()
    for project in cfg.get("projects", ["*"]):
        for path in sorted(glob.glob(f"{root}/{project}/*/*.jsonl.zst")):
            proc = subprocess.run(["zstd", "-dc", path], capture_output=True, check=True)
            for line in proc.stdout.splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") != "GENERATION" or r.get("span_id") in seen:
                    continue
                caller = (r.get("user_id") or "").split("@")[0]
                if callers and caller not in callers:
                    continue
                seen.add(r.get("span_id"))
                msgs = _maybe_json(r.get("input"))
                if isinstance(msgs, dict):
                    msgs = msgs.get("messages")
                if not isinstance(msgs, list):
                    continue
                users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
                if not users:
                    continue
                out = _maybe_json(r.get("output"))
                text = _content_text(out.get("content") if isinstance(out, dict) else out)
                labels = _coerce(job, _maybe_json(text.strip()) if text else {})
                if labels is None:
                    continue
                model = r.get("provided_model_name") or ""
                yield Example(
                    input=_content_text(users[-1].get("content")),
                    labels=labels,
                    ts=0.0,
                    origin=f"langfuse:{caller}:{model}",
                    eval_only=_match(model, eval_only_models),
                    meta={"start_time": r.get("start_time")},
                )


SOURCES = {
    "sqlite": sqlite_source,
    "jsonl": jsonl_source,
    "langfuse_archive": langfuse_archive_source,
}


def example_key(text: str) -> str:
    """Stable id of an input: the same text, ignoring case and spacing, is the same example."""
    return hashlib.sha1(" ".join(text.split()).lower().encode()).hexdigest()


def human_labels(job: Job) -> dict[str, dict]:
    """Labels a person set in the labeling page (`<workdir>/labels.jsonl`), last write per example wins.

    Each value is {"labels": {...}} or {"drop": True} for rows marked as not a real example.
    """
    p = job.workdir / "labels.jsonl"
    out: dict[str, dict] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("undo"):
                out.pop(d["key"], None)
            elif d.get("drop"):
                out[d["key"]] = {"drop": True, "by": d.get("by", "human")}
            else:
                labels = _coerce(job, d.get("labels") or {})
                if labels is not None:
                    out[d["key"]] = {"labels": labels, "by": d.get("by", "human")}
    return out


def apply_human_labels(job: Job, rows: list[Example]) -> list[Example]:
    """A person's label replaces the source's; rows they marked as not real are dropped."""
    human = human_labels(job)
    if not human:
        return rows
    out = []
    for e in rows:
        h = human.get(example_key(e.input))
        if h is None:
            out.append(e)
        elif not h.get("drop"):
            e.labels = h["labels"]
            e.meta = {**e.meta, "human": h["by"], "source_origin": e.origin}
            e.origin = f"human:{h['by']}"
            e.eval_only = False  # the answer is now the person's, not the model's
            out.append(e)
    return out


def collect(job: Job, keep_unlabeled: bool = False, human: bool = True) -> list[Example]:
    rows: list[Example] = []
    for cfg in job.sources:
        if keep_unlabeled:
            cfg = {**cfg, KEEP_UNLABELED: True}
        fn = SOURCES[cfg["type"]]
        n0 = len(rows)
        rows.extend(fn(job, cfg))
        limit = cfg.get("limit")
        if limit and len(rows) - n0 > limit:
            # keep the newest rows of this source
            mine = sorted(rows[n0:], key=lambda e: e.ts, reverse=True)[:limit]
            rows = rows[:n0] + mine
    return apply_human_labels(job, rows) if human else rows
