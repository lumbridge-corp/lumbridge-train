"""What is in your traces? Counts per project, caller and model, and which rows can teach a model.

Reads a Langfuse v4 event archive (`<root>/<project>/<YYYY-MM>/<date>.jsonl.zst`) and prints
aggregates only, never message text.
"""

from __future__ import annotations

import collections
import glob
import json
import subprocess

from .sources import DEFAULT_EVAL_ONLY, _match


def main(root: str) -> None:
    out = {}
    for proj_dir in sorted(glob.glob(f"{root}/*/")):
        proj = proj_dir.rstrip("/").rsplit("/", 1)[1]
        seen: set[str] = set()
        types, callers, models = collections.Counter(), collections.Counter(), collections.Counter()
        trainable = eval_only = 0
        days = set()
        for path in sorted(glob.glob(f"{proj_dir}*/*.jsonl.zst")):
            days.add(path.rsplit("/", 1)[1][:10])
            for line in subprocess.run(["zstd", "-dc", path], capture_output=True).stdout.splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("span_id") in seen:
                    continue
                seen.add(r.get("span_id"))
                types[r.get("type")] += 1
                if r.get("type") != "GENERATION":
                    continue
                model = r.get("provided_model_name") or "?"
                callers[(r.get("user_id") or "?").split("@")[0]] += 1
                models[model] += 1
                if _match(model, DEFAULT_EVAL_ONLY):
                    eval_only += 1
                else:
                    trainable += 1
        out[proj] = {
            "days": f"{min(days)}..{max(days)}" if days else "",
            "types": dict(types.most_common()),
            "generations_trainable": trainable,
            "generations_eval_only": eval_only,
            "models": dict(models.most_common(8)),
            "callers": dict(callers.most_common(15)),
        }
    print(json.dumps(out, indent=1))
