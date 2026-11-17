"""The labeling page: a teacher pre-labels, a person confirms or fixes.

    python -m lumbridge_train label <job.toml> [<job.toml> ...] [--host 127.0.0.1] [--port 8861]

Rows come from the job's sources (also rows with no label yet). A background worker asks the job's label
teacher (`[label] teacher`, default the first teacher) for its answer on the newest rows and caches it in
`<workdir>/prelabels.jsonl`. The page shows one row at a time with the teacher's answer already selected
(or the source's, with `[label] suggest = "source"` when the source is a person), so most rows take one key press. Rows where the teacher and the source disagree come first, then rows the
source has no label for, then rare answers, newest first.

What the person decides goes to `<workdir>/labels.jsonl` (append-only, last write wins). `sources.collect`
applies it to every build, and `split.eval_from = "human"` makes those rows the frozen eval set.

Only the standard library. The page has no login: bind it to localhost or a private network.
"""

from __future__ import annotations

import collections
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import sources
from .config import Job
from .synth import label as teacher_label


def _append(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class Board:
    """One job's rows, the teacher's pre-labels and the person's labels."""

    def __init__(self, job: Job):
        self.job = job
        self.lock = threading.Lock()
        cfg = job.label
        self.teacher = cfg.get("teacher") or next(iter(job.teachers), None)
        self.goal = int(cfg.get("goal", job.split.get("min_human_eval", 200)))
        self.labeler = cfg.get("labeler", "human")  # who answers on this page, recorded with each label
        self.prelabel_max = int(cfg.get("prelabel_max", 600))
        # Which answer starts selected: the teacher's (default) or the source's, when the source is a person
        self.prefer_source = cfg.get("suggest", "teacher") == "source"
        self.max_chars = int(cfg.get("max_input_chars", job.train.get("max_input_chars", 2000)))
        self.pre_path = job.workdir / "prelabels.jsonl"
        self.labels_path = job.workdir / "labels.jsonl"
        self.reload()

    def reload(self) -> None:
        rows = sources.collect(self.job, keep_unlabeled=True, human=False)
        by_key: dict[str, sources.Example] = {}
        for e in rows:
            by_key.setdefault(sources.example_key(e.input), e)
        pre: dict[str, dict] = {}
        if self.pre_path.exists():
            for line in self.pre_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                    pre[d["key"]] = d["labels"]
                except (ValueError, KeyError):
                    pass
        with self.lock:
            self.rows = by_key
            self.pre = pre
            self.human = sources.human_labels(self.job)
            self.history: list[str] = []

    # --- ordering -------------------------------------------------------------------------------------

    def _primary(self, labels: dict | None):
        return (labels or {}).get(self.job.primary.name)

    def suggestion(self, key: str) -> dict:
        first, second = self.rows[key].labels, self.pre.get(key)
        if not self.prefer_source:
            first, second = second, first
        return first or second or {}

    def queue(self, choice: str | None = None, exclude: set[str] = frozenset()) -> list[str]:
        counts = collections.Counter(self._primary(self.suggestion(k)) for k in self.rows)
        out = []
        for k, e in self.rows.items():
            if k in self.human or k in exclude:
                continue
            sug = self.suggestion(k)
            if choice and self._primary(sug) != choice:
                continue
            pre = self.pre.get(k)
            if pre and e.labels and pre != e.labels:
                tier = 0  # teacher and source disagree: most worth a look
            elif pre and not e.labels:
                tier = 1  # only the teacher has an answer
            elif pre:
                tier = 2
            else:
                tier = 3  # not pre-labeled yet
            out.append((tier, counts[self._primary(sug)], -e.ts, k))
        out.sort()
        return [k for *_, k in out]

    # --- the API --------------------------------------------------------------------------------------

    def item(self, choice: str | None, exclude: set[str]) -> dict | None:
        with self.lock:
            q = self.queue(choice, exclude)
            if not q:
                return None
            k = q[0]
            e = self.rows[k]
            return {
                "key": k,
                "input": e.input[: self.max_chars],
                "truncated": len(e.input) > self.max_chars,
                "ts": e.ts,
                "origin": e.origin,
                "source": e.labels or None,
                "teacher": self.pre.get(k),
                "suggested": self.suggestion(k),
                "left": len(q),
            }

    def save(self, body: dict) -> dict:
        key = body["key"]
        with self.lock:
            if key not in self.rows and not body.get("undo"):
                raise KeyError("unknown example")
            rec = {"key": key, "by": body.get("by", self.labeler), "at": int(time.time())}
            if body.get("undo"):
                key = key or (self.history[-1] if self.history else "")
                if not key:
                    return self.stats()  # nothing to undo
                rec["key"] = key
                rec["undo"] = True
                self.human.pop(key, None)
                if key in self.history:
                    self.history.remove(key)
            elif body.get("drop"):
                rec["drop"] = True
                self.human[key] = {"drop": True}
                self.history.append(key)
            else:
                labels = sources._coerce(self.job, body.get("labels") or {})
                if labels is None:
                    raise ValueError("labels do not match the job's fields")
                e = self.rows[key]
                rec.update(labels=labels, teacher=self.pre.get(key), source=e.labels or None, origin=e.origin)
                self.human[key] = {"labels": labels}
                self.history.append(key)
            _append(self.labels_path, rec)
        return self.stats()

    def stats(self) -> dict:
        f = self.job.primary
        labeled = {k: v for k, v in self.human.items() if not v.get("drop")}
        agree = total = 0
        for k, v in labeled.items():
            t = self.pre.get(k) or (self.rows[k].labels if k in self.rows else None)
            if t:
                total += 1
                agree += t == v["labels"]
        return {
            "job": self.job.name,
            "rows": len(self.rows),
            "labeled": len(labeled),
            "dropped": sum(1 for v in self.human.values() if v.get("drop")),
            "goal": self.goal,
            "prelabeled": len(self.pre),
            "teacher": self.teacher,
            "teacher_agreement": round(agree / total, 3) if total else None,
            "labeled_by_choice": dict(collections.Counter(v["labels"][f.name] for v in labeled.values())),
            "suggested_by_choice": dict(collections.Counter(self._primary(self.suggestion(k)) for k in self.rows)),
        }

    def schema(self) -> dict:
        return {
            "job": self.job.name,
            "instructions": self.job.instructions,
            "fields": [
                {"name": f.name, "type": f.type, "choices": f.choices, "descriptions": f.descriptions}
                for f in self.job.fields
            ],
        }

    # --- background pre-labeling ------------------------------------------------------------------------

    def prelabel_forever(self) -> None:
        if not self.teacher:
            return
        conc = self.job.teachers[self.teacher].concurrency
        while True:
            with self.lock:
                newest = sorted(self.rows, key=lambda k: -self.rows[k].ts)[: self.prelabel_max]
                todo = [k for k in newest if k not in self.pre and k not in self.human]
            if not todo:
                time.sleep(60)
                continue
            with ThreadPoolExecutor(max_workers=conc) as pool:
                for k, labels in zip(todo, pool.map(self._ask, todo)):
                    if labels:
                        with self.lock:
                            self.pre[k] = labels
                        _append(self.pre_path, {"key": k, "labels": labels, "teacher": self.teacher, "at": int(time.time())})

    def _ask(self, key: str) -> dict | None:
        try:
            return teacher_label(self.job, self.teacher, self.rows[key].input[: self.max_chars])
        except Exception as e:  # a teacher hiccup skips one row; it is retried on the next pass
            print(f"label: {self.job.name}: teacher: {type(e).__name__}: {e}", flush=True)
            time.sleep(5)
            return None


def make_handler(boards: dict[str, Board]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet: inputs are private, keep them out of logs
            pass

        def _send(self, code: int, body, ctype="application/json") -> None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("cache-control", "no-store")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _board(self, q) -> Board | None:
            name = (q.get("job") or [next(iter(boards))])[0]
            return boards.get(name)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/jobs":
                return self._send(200, [b.schema() | {"stats": b.stats()} for b in boards.values()])
            b = self._board(q)
            if b is None:
                return self._send(404, {"error": "unknown job"})
            if u.path == "/api/next":
                exclude = set(filter(None, (q.get("skip") or [""])[0].split(",")))
                choice = (q.get("choice") or [None])[0] or None
                return self._send(200, {"item": b.item(choice, exclude), "stats": b.stats()})
            if u.path == "/api/stats":
                return self._send(200, b.stats())
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            b = self._board(parse_qs(u.query))
            if u.path != "/api/label" or b is None:
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
                return self._send(200, b.save(body))
            except (ValueError, KeyError) as e:
                return self._send(400, {"error": str(e)})

    return Handler


def serve(jobs: list[Job], host: str = "127.0.0.1", port: int = 8861, prelabel: bool = True) -> None:
    boards = {j.name: Board(j) for j in jobs}
    for b in boards.values():
        print(f"label: {b.job.name}: {len(b.rows)} rows, {len(b.pre)} pre-labeled, {len(b.human)} done", flush=True)
        if prelabel:
            threading.Thread(target=b.prelabel_forever, daemon=True, name=f"prelabel-{b.job.name}").start()
    print(f"label: http://{host}:{port}/", flush=True)
    ThreadingHTTPServer((host, port), make_handler(boards)).serve_forever()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label</title>
<style>
:root{--bg:#fff;--fg:#0a0a0a;--mute:#6b6b6b;--line:#e5e5e5;--soft:#f5f5f5;--inv-bg:#0a0a0a;--inv-fg:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#0a0a0a;--fg:#f2f2f2;--mute:#9a9a9a;--line:#262626;--soft:#161616;--inv-bg:#f2f2f2;--inv-fg:#0a0a0a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 Geist,ui-sans-serif,system-ui,-apple-system,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}
header{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:12px}
h1{font-size:18px;margin:0;font-weight:600}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
button,select{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer}
button:hover{border-color:var(--fg)}
button.on{background:var(--inv-bg);color:var(--inv-fg);border-color:var(--inv-bg)}
.meta{color:var(--mute);font-size:13px}
.bar{height:4px;background:var(--soft);border-radius:2px;overflow:hidden;margin:8px 0 16px}
.bar>i{display:block;height:100%;background:var(--fg)}
.card{border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
pre{white-space:pre-wrap;word-break:break-word;margin:0;font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:46vh;overflow:auto}
.note{font-size:13px;margin-top:10px;padding:8px 10px;border-radius:8px;background:var(--soft)}
.field{margin:14px 0}
.field h2{font-size:13px;font-weight:600;margin:0 0 6px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute)}
.choices{display:flex;flex-wrap:wrap;gap:6px}
.choices button{display:flex;gap:6px;align-items:center}
kbd{font:12px ui-monospace,monospace;color:var(--mute)}
button.on kbd{color:var(--inv-fg);opacity:.7}
.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
.actions .save{flex:1;min-width:160px;font-weight:600}
.empty{padding:40px 0;text-align:center;color:var(--mute)}
.desc{color:var(--mute);font-size:13px;min-height:20px}
@media (max-width:600px){button{padding:12px 14px}.actions .save{min-width:100%}}
</style></head>
<body><main>
<header><h1>Label</h1><div class="tabs" id="tabs"></div><span class="meta" id="stat"></span></header>
<div class="bar"><i id="prog" style="width:0"></i></div>
<div class="meta" style="margin-bottom:8px">Show <select id="filter"><option value="">everything, hardest first</option></select></div>
<div id="view"></div>
<p class="meta">Keys: <b>1–9</b> pick an answer, <b>R T Y</b> flip yes/no fields, <b>Enter</b> save, <b>S</b> skip, <b>X</b> not a real example, <b>U</b> undo.</p>
</main>
<script>
const $=s=>document.querySelector(s);
let jobs=[],job=null,item=null,sel={},skipped=new Set();
const BOOL_KEYS="rty";
async function api(path,opts){const r=await fetch(path,opts);if(!r.ok)throw new Error((await r.json()).error||r.status);return r.json()}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function fmt(l){return l?Object.entries(l).map(([k,v])=>k+"="+v).join(", "):"none"}
async function init(){
  jobs=await api("/api/jobs");
  const saved=(()=>{try{return localStorage.getItem("label.job")}catch(e){return null}})();
  job=jobs.find(j=>j.job===saved)||jobs[0];
  renderTabs();await next();
}
function renderTabs(){
  $("#tabs").innerHTML=jobs.map(j=>`<button class="${j.job===job.job?"on":""}" data-j="${esc(j.job)}">${esc(j.job)}</button>`).join("");
  $("#tabs").querySelectorAll("button").forEach(b=>b.onclick=async()=>{job=jobs.find(j=>j.job===b.dataset.j);try{localStorage.setItem("label.job",job.job)}catch(e){}skipped.clear();renderTabs();fillFilter();await next()});
  fillFilter();
}
function fillFilter(){
  const p=job.fields[0];
  $("#filter").innerHTML=`<option value="">everything, hardest first</option>`+p.choices.map(c=>`<option value="${esc(c)}">suggested: ${esc(c)}</option>`).join("");
  $("#filter").onchange=()=>{skipped.clear();next()};
}
function stats(s){
  $("#stat").textContent=`${s.labeled} of ${s.goal} labeled · ${s.rows} rows · ${s.prelabeled} pre-labeled by ${s.teacher||"—"}`+(s.teacher_agreement!=null?` · teacher right ${Math.round(s.teacher_agreement*100)}%`:"");
  $("#prog").style.width=Math.min(100,100*s.labeled/Math.max(1,s.goal))+"%";
}
async function next(){
  const q=new URLSearchParams({job:job.job,choice:$("#filter").value,skip:[...skipped].join(",")});
  const r=await api("/api/next?"+q);stats(r.stats);item=r.item;
  if(!item){$("#view").innerHTML=`<div class="empty">Nothing left here. ${$("#filter").value?"Try another filter.":"All rows are labeled."}</div>`;return}
  sel=JSON.parse(JSON.stringify(item.suggested||{}));
  job.fields.forEach(f=>{if(f.type==="bool"&&sel[f.name]===undefined)sel[f.name]=false});
  render();
}
function render(){
  const when=item.ts?new Date(item.ts*1000).toLocaleString():"";
  let note="";
  if(item.teacher&&item.source&&fmt(item.teacher)!==fmt(item.source))note=`<div class="note">Disagree: teacher says <b>${esc(fmt(item.teacher))}</b>, source (${esc(item.origin)}) says <b>${esc(fmt(item.source))}</b></div>`;
  else if(!item.teacher)note=`<div class="note">Not pre-labeled yet; showing the source's answer (${esc(item.origin)}).</div>`;
  let fields=job.fields.map((f,fi)=>{
    if(f.type==="bool"){const k=BOOL_KEYS[job.fields.filter(x=>x.type==="bool").indexOf(f)]||"";
      return `<div class="field"><h2>${esc(f.name)}</h2><div class="choices">${[true,false].map(v=>`<button class="${sel[f.name]===v?"on":""}" data-f="${esc(f.name)}" data-v="${v}">${v?"yes":"no"}${v&&k?` <kbd>${k.toUpperCase()}</kbd>`:""}</button>`).join("")}</div></div>`}
    return `<div class="field"><h2>${esc(f.name)}</h2><div class="choices">${f.choices.map((c,i)=>`<button class="${sel[f.name]===c?"on":""}" data-f="${esc(f.name)}" data-c="${esc(c)}">${fi===0&&i<9?`<kbd>${i+1}</kbd> `:""}${esc(c)}</button>`).join("")}</div><div class="desc">${esc((f.descriptions||{})[sel[f.name]]||"")}</div></div>`;
  }).join("");
  $("#view").innerHTML=`<div class="card"><div class="meta" style="margin-bottom:8px">${esc(when)} · ${item.left} left${item.truncated?" · cut to fit":""}</div><pre>${esc(item.input)}</pre>${note}</div>${fields}
  <div class="actions"><button class="save on" id="save">Save <kbd>Enter</kbd></button><button id="skip">Skip <kbd>S</kbd></button><button id="drop">Not a real example <kbd>X</kbd></button><button id="undo">Undo <kbd>U</kbd></button></div>`;
  $("#view").querySelectorAll("[data-f]").forEach(b=>b.onclick=()=>{sel[b.dataset.f]=b.dataset.c!==undefined?b.dataset.c:b.dataset.v==="true";render()});
  $("#save").onclick=save;$("#skip").onclick=skip;$("#drop").onclick=drop;$("#undo").onclick=undo;
}
async function post(body){const s=await api("/api/label?job="+encodeURIComponent(job.job),{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});stats(s)}
async function save(){if(!item)return;await post({key:item.key,labels:sel});await next()}
async function drop(){if(!item)return;await post({key:item.key,drop:true});await next()}
async function skip(){if(!item)return;skipped.add(item.key);await next()}
async function undo(){await post({key:"",undo:true});skipped.clear();await next()}
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="SELECT"||e.metaKey||e.ctrlKey||!item&&e.key.toLowerCase()!=="u")return;
  const k=e.key.toLowerCase(),p=job.fields[0];
  if(/^[1-9]$/.test(k)&&p.type!=="bool"&&p.choices[+k-1]){sel[p.name]=p.choices[+k-1];render()}
  else if(BOOL_KEYS.includes(k)){const f=job.fields.filter(x=>x.type==="bool")[BOOL_KEYS.indexOf(k)];if(f){sel[f.name]=!sel[f.name];render()}}
  else if(k==="enter"){e.preventDefault();save()}
  else if(k==="s")skip();else if(k==="x")drop();else if(k==="u")undo();
});
init().catch(e=>$("#view").innerHTML=`<div class="empty">${esc(e.message)}</div>`);
</script></body></html>
"""
