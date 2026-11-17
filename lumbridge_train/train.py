"""Train the student with lumbridge-rl (prime-rl) and export HF weights."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .config import Job, expand


def rl_dir(job: Job) -> Path:
    return expand(job.train.get("rl_dir", "~/src/lumbridge-rl"))


def uv() -> str:
    return os.environ.get("LT_UV", str(expand("~/.local/bin/uv")))


def sft_toml(job: Job, run_dir: Path, remote_dir: Path | None = None) -> Path:
    """Writes the lumbridge-rl SFT config. With `remote_dir`, paths point there (a rented machine)."""
    t = job.train
    renderer = t.get("renderer")
    base = remote_dir or run_dir
    lines = [
        'mode = "sft"',
        f"max_steps = {int(t.get('max_steps', 60))}",
        f'output_dir = "{base / "rl"}"',
        "clean = true",
        "",
        "[run]",
        'name = "sft"',
        "",
        "[ckpt]",
        "",
        "[model]",
        f'name = "{t["student"]}"',
        "",
        "[data]",
        f'name = "{base / "sft"}"',
        f"seq_len = {int(t.get('seq_len', 2048))}",
        f"batch_size = {int(t.get('batch_size', 16))}",
        "",
        "[optim]",
        f"lr = {float(t.get('lr', 2e-5))}",
    ]
    if renderer:
        lines += ["", "[renderer]"] + [f"{k} = {v!r}".replace("'", '"').replace("False", "false").replace("True", "true")
                                     for k, v in renderer.items()]
    lines += ["", "[deployment]", "num_train_gpus = 1"]
    path = run_dir / ("sft-remote.toml" if remote_dir else "sft.toml")
    path.write_text("\n".join(lines) + "\n")
    return path


def run_sft(job: Job, run_dir: Path) -> Path:
    """Returns the exported HF weights directory."""
    cfg = sft_toml(job, run_dir)
    log = run_dir / "sft.log"
    cmd = ["nice", "-n", "10", uv(), "run", "--no-sync", "sft", "@", str(cfg), "--no-monitors.prime"]
    print(f"train: {' '.join(cmd)}  (log {log})")
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=rl_dir(job), stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"sft failed rc={rc}, see {log}")
    ckpts = sorted((run_dir / "rl" / "sft" / "checkpoints").glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if not ckpts:
        raise RuntimeError(f"no checkpoint under {run_dir / 'rl' / 'sft' / 'checkpoints'}")
    weights = run_dir / "weights"
    cmd = [uv(), "run", "--no-sync", "python", "tools/convert_dcp_to_bf16.py", str(ckpts[-1]), str(weights)]
    with open(run_dir / "export.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=rl_dir(job), stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"export failed rc={rc}, see {run_dir / 'export.log'}")
    return weights


def run_eval(job: Job, model: str, eval_file: Path, out: Path) -> dict:
    """Runs evaluate.py inside the lumbridge-rl environment (it has torch + transformers)."""
    import json

    pkg_root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": f"{pkg_root}:{os.environ.get('PYTHONPATH', '')}"}
    cmd = [uv(), "run", "--no-sync", "python", "-m", "lumbridge_train.evaluate",
           str(job.path), model, str(eval_file), str(out)]
    print(f"eval: {model}")
    rc = subprocess.run(["nice", "-n", "10", *cmd], cwd=rl_dir(job), env=env).returncode
    if rc != 0:
        raise RuntimeError(f"eval failed rc={rc} for {model}")
    return json.loads(out.read_text())
