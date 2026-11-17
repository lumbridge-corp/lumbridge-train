"""Train and evaluate on a rented GPU (Prime Intellect), then delete it.

A job opts in with a `[train.cloud]` table. The runner picks the cheapest available offer from the job's GPU list
(spot first, then on-demand if allowed), creates a pod with our SSH key, copies lumbridge-rl, this package, the
run's dataset and the baseline weights to it, installs, trains, exports, scores the baseline and the new weights on
the eval set, copies the weights and scores back, and deletes the pod. The pod is deleted in a `finally`, and every
create and delete is logged to `<workdir>/cloud-pods.jsonl`; `python -m lumbridge_train cloud-reap` deletes any of
our pods left running (for a timer, as a second safety net).

Your dataset leaves your machine for the rented one. A job must say so: `data_leaves_fleet = true`. Without it the
runner refuses, so a job with private data never goes to the cloud by accident.

    [train.cloud]
    data_leaves_fleet = true
    gpus = ["RTX_PRO_6000B_96GB", "A100_80GB", "H100_80GB"]   # in order of preference
    spot = true                 # spot is half price but can be stopped without warning (the run retries)
    on_demand_fallback = true
    max_price_hr = 3.5
    max_hours = 2.0             # hard limit for the remote work
    disk_gb = 200               # pass it: without diskSize some providers give a 50 GB root disk
    api_key_file = "~/.config/lumbridge-train/prime.key"   # or api_key_env = "PRIME_API_KEY"
    ssh_key = "~/.ssh/lumbridge_train_pi"                   # its .pub must be registered with Prime Intellect
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import Job, expand

API = "https://api.primeintellect.ai/api/v1"
POD_PREFIX = "lt-"
REMOTE = Path("/root")
RL, PKG, RUN = REMOTE / "lumbridge-rl", REMOTE / "lt", REMOTE / "run"

DEFAULTS = {
    "data_leaves_fleet": False,
    "gpus": ["RTX_PRO_6000B_96GB", "A100_80GB", "H100_80GB"],
    "spot": True,
    "on_demand_fallback": True,
    "max_price_hr": 3.5,
    "max_hours": 2.0,
    "disk_gb": 200,
    "image": "ubuntu_22_cuda_12",
    "api_key_file": "~/.config/lumbridge-train/prime.key",
    "api_key_env": "PRIME_API_KEY",
    "ssh_key": "~/.ssh/lumbridge_train_pi",
    "retries": 1,
}


class Preempted(RuntimeError):
    """The pod went away under us (spot capacity reclaimed)."""


def settings(job: Job) -> dict:
    return {**DEFAULTS, **job.train.get("cloud", {})}


def wanted(job: Job) -> bool:
    return bool(job.train.get("cloud"))


# --- Prime Intellect API -------------------------------------------------------------------------------------


class PrimeIntellect:
    def __init__(self, cfg: dict):
        key = os.environ.get(cfg["api_key_env"] or "")
        path = expand(cfg["api_key_file"]) if cfg.get("api_key_file") else None
        if not key and path and path.exists():
            key = path.read_text().strip()
        if not key:
            raise RuntimeError(f"no Prime Intellect API key: set ${cfg['api_key_env']} or write {cfg['api_key_file']}")
        self.key = key

    def req(self, method: str, path: str, body: dict | None = None) -> dict | list:
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(API + path, data=data, method=method, headers={
            "authorization": f"Bearer {self.key}", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Prime Intellect {method} {path}: HTTP {e.code} {e.read()[:300]!r}") from None
        return json.loads(raw) if raw else {}

    def offers(self, gpu: str) -> list[dict]:
        q = urllib.parse.urlencode({"gpu_type": gpu, "gpu_count": 1})
        d = self.req("GET", f"/availability/gpus?{q}")
        rows = d if isinstance(d, list) else d.get("items", d.get("data", []))
        return [o for o in rows if isinstance(o, dict)]

    def pods(self) -> list[dict]:
        d = self.req("GET", "/pods/?limit=100")
        return d.get("data", []) if isinstance(d, dict) else d

    def ssh_key_id(self, pub: str) -> str:
        d = self.req("GET", "/ssh_keys/")
        body = pub.split()[1]
        for k in d.get("data", []) if isinstance(d, dict) else d:
            if (k.get("publicKey") or "").split()[1:2] == [body]:
                return k["id"]
        raise RuntimeError("our SSH public key is not registered with Prime Intellect (add it under SSH keys)")


def choose(api: PrimeIntellect, cfg: dict) -> list[dict]:
    """Candidate offers, best first: spot in the job's GPU order, then on-demand (if allowed)."""
    spot, ondemand = [], []
    for rank, gpu in enumerate(cfg["gpus"]):
        for o in api.offers(gpu):
            price = (o.get("prices") or {}).get("onDemand")
            if o.get("stockStatus") != "Available" or price is None or price > cfg["max_price_hr"]:
                continue
            if cfg["image"] not in (o.get("images") or []):
                continue
            (spot if o.get("isSpot") else ondemand).append((price, rank, o))
    # the job's GPU order first (it knows what its model needs), the cheapest offer of that GPU next
    out = [o for *_, o in sorted(spot, key=lambda x: (x[1], x[0]))] if cfg["spot"] else []
    if cfg["on_demand_fallback"] or not cfg["spot"]:
        out += [o for *_, o in sorted(ondemand, key=lambda x: (x[1], x[0]))]
    return out


def _ledger(job: Job, entry: dict) -> None:
    job.workdir.mkdir(parents=True, exist_ok=True)
    with open(job.workdir / "cloud-pods.jsonl", "a") as fh:
        fh.write(json.dumps({"at": dt.datetime.now().isoformat(timespec="seconds"), **entry}) + "\n")


def create(api: PrimeIntellect, job: Job, cfg: dict, offer: dict, key_id: str, name: str) -> str:
    pod = {
        "name": name, "cloudId": offer["cloudId"], "gpuType": offer["gpuType"], "socket": offer.get("socket"),
        "gpuCount": 1, "image": cfg["image"], "dataCenterId": offer.get("dataCenter"),
        "country": offer.get("country"), "security": offer.get("security"), "diskSize": int(cfg["disk_gb"]),
        "sshKeyId": key_id,
    }
    d = api.req("POST", "/pods/", {"pod": pod, "provider": {"type": offer["provider"]}})
    pid = d["id"]
    _ledger(job, {"action": "create", "pod": pid, "name": name, "cloudId": offer["cloudId"],
                  "price_hr": d.get("priceHr"), "spot": bool(offer.get("isSpot"))})
    return pid


def delete(api: PrimeIntellect, job: Job | None, pid: str, why: str) -> None:
    for attempt in range(5):
        try:
            api.req("DELETE", f"/pods/{pid}")
            break
        except RuntimeError as e:
            if "404" in str(e):
                break
            print(f"cloud: delete {pid} failed ({e}); retrying", flush=True)
            time.sleep(10 * (attempt + 1))
    if job is not None:
        _ledger(job, {"action": "delete", "pod": pid, "why": why})


def wait_active(api: PrimeIntellect, pid: str, timeout_s: int = 900) -> tuple[str, str, str]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        d = api.req("GET", f"/pods/{pid}")
        status, conn = d.get("status"), d.get("sshConnection")
        if status in ("ERROR", "TERMINATED", "FAILED"):
            raise Preempted(f"pod {pid} is {status} before it started")
        if status == "ACTIVE" and conn and d.get("installationStatus") in (None, "FINISHED"):
            conn = conn[0] if isinstance(conn, list) else conn
            parts = conn.split()  # "root@1.2.3.4 -p 22"
            user, host = parts[0].split("@")
            port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
            return user, host, port
        time.sleep(10)
    raise RuntimeError(f"pod {pid} not ready after {timeout_s}s")


def reap(cfg: dict | None = None, older_than_h: float = 0.0) -> int:
    """Delete our pods (name starts with `lt-`) older than `older_than_h` hours. Returns how many."""
    api = PrimeIntellect(cfg or DEFAULTS)
    n = 0
    now = dt.datetime.now(dt.timezone.utc)
    for p in api.pods():
        if not (p.get("name") or "").startswith(POD_PREFIX) or p.get("status") == "TERMINATED":
            continue
        created = p.get("createdAt") or ""
        try:
            age_h = (now - dt.datetime.fromisoformat(created.replace("Z", "+00:00"))).total_seconds() / 3600
        except ValueError:
            age_h = 1e9
        if age_h >= older_than_h:
            print(f"cloud-reap: deleting {p.get('name')} ({p['id']}, {age_h:.1f} h old)", flush=True)
            delete(api, None, p["id"], "reap")
            n += 1
    return n


# --- ssh -----------------------------------------------------------------------------------------------------


class Box:
    def __init__(self, job: Job, cfg: dict, user: str, host: str, port: str):
        self.dest = f"{user}@{host}"
        kh = job.workdir / ".cloud_known_hosts"
        self.opts = ["-i", str(expand(cfg["ssh_key"])), "-p", port, "-o", "StrictHostKeyChecking=accept-new",
                     "-o", f"UserKnownHostsFile={kh}", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
                     "-o", "ServerAliveCountMax=4", "-o", "BatchMode=yes"]

    def run(self, script: str, check: bool = True, timeout: float | None = 120) -> subprocess.CompletedProcess:
        p = subprocess.run(["ssh", *self.opts, self.dest, "bash -lc " + shlex.quote(script)],
                           capture_output=True, text=True, timeout=timeout)
        if check and p.returncode != 0:
            raise RuntimeError(f"remote failed rc={p.returncode}: {p.stderr[-500:] or p.stdout[-500:]}")
        return p

    def _rsync(self, args: list[str]) -> None:
        ssh = "ssh " + " ".join(shlex.quote(o) for o in self.opts)
        p = subprocess.run(["rsync", "-a", "--partial", "-e", ssh, *args], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"rsync failed rc={p.returncode}: {p.stderr[-500:]}")

    def push(self, src: Path, dst: Path, exclude: tuple[str, ...] = ()) -> None:
        ex = [a for e in exclude for a in ("--exclude", e)]
        self._rsync([*ex, f"{src}/" if src.is_dir() else str(src), f"{self.dest}:{dst}{'/' if src.is_dir() else ''}"])

    def pull(self, src: Path, dst: Path, is_dir: bool = False) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._rsync([f"{self.dest}:{src}{'/' if is_dir else ''}", f"{dst}{'/' if is_dir else ''}"])


REMOTE_SCRIPT = r"""
set -uo pipefail
exec > {run}/remote.log 2>&1
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$HOME/.cache/uv PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# uv's own Python ships its headers; the image's system Python may not, and Triton compiles against them
export UV_PYTHON_PREFERENCE=only-managed
export MAMBA_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=FALSE
step() {{ echo "=== $(date -u +%T) $*"; echo "$*" > {run}/STEP; }}
fail() {{ echo "FAILED: $*"; echo "$*" > {run}/FAILED; exit 1; }}
step setup; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; df -h / | tail -1
# Some images ship a CUDA toolkit and profilers we do not need (torch brings its own CUDA); drop them if space is short.
[ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -lt 60 ] && rm -rf /usr/local/cuda-1* /opt/nvidia/nsight-* && df -h / | tail -1
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || fail "install uv"
cd {rl}
uv sync --extra gpu --extra flash-attn --no-install-package flashinfer-jit-cache --no-install-package flashinfer-cubin \
  > {run}/sync.log 2>&1 || fail "uv sync"
step sft
uv run --no-sync sft @ {run}/sft.toml --no-monitors.prime > {run}/sft.log 2>&1 || fail "sft"
ck=$(ls -d {run}/rl/sft/checkpoints/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
[ -n "$ck" ] || fail "no checkpoint"
step export
uv run --no-sync python tools/convert_dcp_to_bf16.py "$ck" {run}/weights > {run}/export.log 2>&1 || fail "export"
step eval
export PYTHONPATH={pkg}
uv run --no-sync python -m lumbridge_train.evaluate {run}/job.toml {baseline} {run}/eval.jsonl {run}/eval-baseline.json \
  > {run}/eval-baseline.log 2>&1 || fail "eval baseline"
uv run --no-sync python -m lumbridge_train.evaluate {run}/job.toml {run}/weights {run}/eval.jsonl {run}/eval-trained.json \
  > {run}/eval-trained.log 2>&1 || fail "eval trained"
step done; touch {run}/DONE
"""


def _remote_work(job: Job, cfg: dict, box: Box, run_dir: Path, baseline: str) -> tuple[Path, dict, dict]:
    from .train import rl_dir, sft_toml

    box.run(f"mkdir -p {RUN} {PKG}")
    box.push(rl_dir(job), RL, exclude=(".venv", "outputs", ".git", "__pycache__", "*.log"))
    box.push(Path(__file__).resolve().parent, PKG / "lumbridge_train", exclude=("__pycache__",))
    box.push(run_dir / "sft", RUN / "sft")
    box.push(run_dir / "eval.jsonl", RUN / "eval.jsonl")
    box.push(job.path, RUN / "job.toml")
    box.push(sft_toml(job, run_dir, remote_dir=RUN), RUN / "sft.toml")
    remote_baseline = baseline
    if Path(baseline).is_dir():
        box.push(Path(baseline), RUN / "baseline")
        remote_baseline = str(RUN / "baseline")

    script = REMOTE_SCRIPT.format(run=RUN, rl=RL, pkg=PKG, baseline=shlex.quote(remote_baseline))
    box.push(_tmpfile(run_dir / "remote.sh", script), RUN / "remote.sh")
    box.run(f"nohup bash {RUN}/remote.sh >/dev/null 2>&1 &")

    deadline = time.time() + float(cfg["max_hours"]) * 3600
    last, misses = "", 0
    while True:
        if time.time() > deadline:
            raise RuntimeError(f"remote work passed max_hours={cfg['max_hours']}")
        time.sleep(30)
        try:
            p = box.run(f"cat {RUN}/STEP 2>/dev/null; ls {RUN}/DONE {RUN}/FAILED 2>/dev/null; cat {RUN}/FAILED 2>/dev/null",
                        check=False, timeout=60)
        except subprocess.TimeoutExpired:
            p = None
        if p is None or p.returncode == 255:  # ssh itself failed
            misses += 1
            if misses >= 6:
                raise Preempted("lost the pod (ssh unreachable for 3 min)")
            continue
        misses = 0
        out = p.stdout.split()
        step = out[0] if out else ""
        if step != last:
            print(f"cloud: {step}", flush=True)
            last = step
        if f"{RUN}/FAILED" in p.stdout:
            box.pull(RUN / "remote.log", run_dir / "remote.log")
            for name in ("sft.log", "export.log", "eval-baseline.log", "eval-trained.log", "sync.log"):
                try:
                    box.pull(RUN / name, run_dir / name)
                except RuntimeError:
                    pass
            raise RuntimeError(f"remote step failed: {' '.join(out[1:])}; logs in {run_dir}")
        if f"{RUN}/DONE" in p.stdout:
            break

    for name in ("remote.log", "sft.log", "export.log", "eval-baseline.json", "eval-trained.json",
                 "eval-baseline.log", "eval-trained.log"):
        box.pull(RUN / name, run_dir / name)
    box.pull(RUN / "weights", run_dir / "weights", is_dir=True)
    return (run_dir / "weights", json.loads((run_dir / "eval-baseline.json").read_text()),
            json.loads((run_dir / "eval-trained.json").read_text()))


def _tmpfile(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def train_and_eval(job: Job, run_dir: Path, baseline: str) -> tuple[Path, dict, dict]:
    """Rent, train, export, evaluate baseline and new weights, copy back, delete. Returns (weights, base, new)."""
    cfg = settings(job)
    if cfg["data_leaves_fleet"] is not True:
        raise RuntimeError(
            f"job {job.name}: [train.cloud] needs data_leaves_fleet = true. The dataset would be copied to a rented "
            "machine; only set this for data you are allowed to send there.")
    api = PrimeIntellect(cfg)
    pub = Path(str(expand(cfg["ssh_key"])) + ".pub").read_text()
    key_id = cfg.get("ssh_key_id") or api.ssh_key_id(pub)
    offers = choose(api, cfg)
    if not offers:
        raise RuntimeError(f"no available offer for {cfg['gpus']} under ${cfg['max_price_hr']}/h")
    errors = []
    for attempt, offer in enumerate(offers[: 1 + int(cfg["retries"])]):
        name = f"{POD_PREFIX}{job.name}-{run_dir.name}"[:60]
        price = offer["prices"]["onDemand"]
        print(f"cloud: renting {offer['gpuType']} {'spot' if offer.get('isSpot') else 'on-demand'} "
              f"{offer.get('dataCenter')} ${price}/h (attempt {attempt + 1})", flush=True)
        pid = create(api, job, cfg, offer, key_id, name)
        t0 = time.time()
        why = "done"
        try:
            user, host, port = wait_active(api, pid)
            print(f"cloud: pod {pid} up in {time.time() - t0:.0f}s", flush=True)
            return _remote_work(job, cfg, Box(job, cfg, user, host, port), run_dir, baseline)
        except Preempted as e:
            why = f"preempted: {e}"
            errors.append(why)
            print(f"cloud: {why}; trying the next offer", flush=True)
        except BaseException as e:
            why = f"error: {type(e).__name__}: {e}"
            raise
        finally:
            delete(api, job, pid, why)
            mins = (time.time() - t0) / 60
            print(f"cloud: pod {pid} deleted after {mins:.1f} min (~${price * mins / 60:.2f})", flush=True)
    raise RuntimeError(f"cloud run failed on every offer: {errors}")
