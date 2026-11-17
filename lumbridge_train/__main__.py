"""lumbridge-train CLI.

    python -m lumbridge_train build   <job.toml>        # collect + synth + write a dataset run dir
    python -m lumbridge_train run     <job.toml> [--no-promote]
    python -m lumbridge_train due     <job.toml>        # exit 0 if today is the job's training day
    python -m lumbridge_train profile <langfuse-archive-root>
"""

from __future__ import annotations

import argparse
import sys

from . import profile, runner
from .config import load


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lumbridge-train")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("job")
    r = sub.add_parser("run")
    r.add_argument("job")
    r.add_argument("--no-promote", action="store_true")
    d = sub.add_parser("due")
    d.add_argument("job")
    p = sub.add_parser("profile")
    p.add_argument("root")
    a = ap.parse_args(argv)

    if a.cmd == "profile":
        profile.main(a.root)
        return 0
    job = load(a.job)
    if a.cmd == "build":
        runner.build(job, runner.new_run_dir(job))
    elif a.cmd == "run":
        runner.run(job, promote_ok=not a.no_promote)
    elif a.cmd == "due":
        ok = runner.due(job)
        print(f"{job.name}: training day is {job.schedule['training_day']}; today {'yes' if ok else 'no'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
