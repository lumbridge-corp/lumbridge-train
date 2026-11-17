"""Score a model on a frozen eval set: per-field accuracy, per-answer recall, valid-JSON rate.

Run inside an environment with torch + transformers:
    python -m lumbridge_train.evaluate <job.toml> <model dir or HF id> <eval.jsonl> <out.json>
"""

from __future__ import annotations

import collections
import json
import sys
import time

from .config import load
from .dataset import prompt_messages
from .sources import _coerce
from .teacher import parse_json


def main(job_path: str, model: str, eval_file: str, out: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    job = load(job_path)
    rows = [json.loads(line) for line in open(eval_file)]
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda").eval()

    # Render prompts exactly as training did: the same renderer (with its template controls,
    # e.g. Qwen3 thinking off prefills an empty think block), falling back to the chat template.
    renderer = None
    if job.train.get("renderer"):
        try:
            from pydantic import TypeAdapter
            from renderers import RendererConfig, create_renderer

            renderer = create_renderer(tok, TypeAdapter(RendererConfig).validate_python(job.train["renderer"]))
        except Exception as e:
            print(f"eval: renderer unavailable ({type(e).__name__}: {e}); using the chat template")

    def render_ids(text: str) -> list[int]:
        msgs = prompt_messages(job, text)
        if renderer is not None:
            return list(renderer.render_ids(msgs, add_generation_prompt=True))
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True)

    preds = []
    bs = int(job.train.get("eval_batch_size", 16))
    max_new = int(job.train.get("eval_max_new_tokens", 64))
    t0 = time.time()
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        ids = [render_ids(r["input"]) for r in batch]
        width = max(len(x) for x in ids)
        input_ids = torch.tensor([[tok.pad_token_id] * (width - len(x)) + x for x in ids], device="cuda")
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], device="cuda")
        with torch.no_grad():
            gen = m.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for r, g in zip(batch, gen[:, width:]):
            text = tok.decode(g, skip_special_tokens=True)
            raw = parse_json(text)
            preds.append((r, _coerce(job, raw) if raw else None, text))

    n = len(preds)
    fields = [f.name for f in job.fields]
    correct = collections.Counter()
    valid = 0
    prim = job.primary.name
    per = collections.defaultdict(lambda: [0, 0])  # choice -> [hit, total]
    confusion = collections.Counter()
    for r, p, _ in preds:
        gold = r["labels"]
        per[gold[prim]][1] += 1
        if p is None:
            confusion[(gold[prim], "<invalid>")] += 1
            continue
        valid += 1
        confusion[(gold[prim], p[prim])] += 1
        for f in fields:
            correct[f] += int(p[f] == gold[f])
        per[gold[prim]][0] += int(p[prim] == gold[prim])
    res = {
        "model": model,
        "n": n,
        "seconds": round(time.time() - t0, 1),
        "valid_json": round(valid / n, 4) if n else 0,
        **{f"{f}_acc": round(correct[f] / n, 4) if n else 0 for f in fields},
        "recall": {c: round(h / t, 3) for c, (h, t) in sorted(per.items())},
        "support": {c: t for c, (h, t) in sorted(per.items())},
        "top_confusions": [[g, p, k] for (g, p), k in confusion.most_common(12) if g != p],
        "samples": [{"gold": r["labels"], "pred": p, "raw": t[:160]} for r, p, t in preds[:5]],
    }
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("samples",)}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
