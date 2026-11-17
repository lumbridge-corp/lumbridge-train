import datetime as dt
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from lumbridge_train import config, dataset, runner, sources
from lumbridge_train.teacher import parse_json

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "jobs" / "example-ticket-routing.toml"


def job(tmp: Path, extra: str = "") -> config.Job:
    text = EXAMPLE.read_text().replace(
        'workdir = "~/lumbridge-train-runs/example-ticket-routing"', f'workdir = "{tmp}"'
    )
    p = tmp / "job.toml"
    p.write_text(text + extra)
    return config.load(p)


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_system_prompt_lists_choices(self):
        j = job(self.tmp)
        sp = j.system_prompt()
        self.assertIn("billing, bug, account, feature, other", sp)
        self.assertIn('"urgent": true or false', sp)

    def test_training_day_validated(self):
        p = self.tmp / "bad.toml"
        p.write_text(EXAMPLE.read_text().replace('training_day = "sunday"', 'training_day = "someday"'))
        with self.assertRaises(ValueError):
            config.load(p)

    def test_due(self):
        j = job(self.tmp)
        self.assertTrue(runner.due(j, dt.date(2026, 11, 15)))  # a Sunday
        self.assertFalse(runner.due(j, dt.date(2026, 11, 14)))

    def test_sqlite_source_coerces_and_drops_invalid(self):
        db = self.tmp / "t.db"
        con = sqlite3.connect(db)
        con.execute("create table t (body text, cat text, urg int, ts int)")
        con.executemany("insert into t values (?,?,?,?)", [
            ("charged twice", "billing", 1, 3), ("app crashes", "bug", 0, 2), ("hello", "nonsense", 0, 1),
        ])
        con.commit()
        con.close()
        j = job(self.tmp, f'''
[[source]]
type = "sqlite"
path = "{db}"
query = "select body as input, cat as category, urg as urgent, ts from t"
''')
        rows = sources.collect(j)
        self.assertEqual([r.labels for r in rows], [
            {"category": "billing", "urgent": True}, {"category": "bug", "urgent": False}])

    def test_split_is_by_time_and_eval_only_goes_to_eval(self):
        j = job(self.tmp)
        rows = [sources.Example(f"t{i}", {"category": "bug", "urgent": False}, ts=i) for i in range(10)]
        rows.append(sources.Example("claude said", {"category": "bug", "urgent": False}, ts=0, eval_only=True))
        tr, ev = dataset.split(j, rows)
        self.assertEqual(max(e.ts for e in tr) < min(e.ts for e in ev if not e.eval_only), True)
        self.assertIn("claude said", [e.input for e in ev])
        self.assertNotIn("claude said", [e.input for e in tr])

    def test_langfuse_rows_from_restricted_models_are_eval_only(self):
        self.assertTrue(sources._match("claude-opus-5-5", sources.DEFAULT_EVAL_ONLY))
        self.assertFalse(sources._match("brain", sources.DEFAULT_EVAL_ONLY))

    def test_write_produces_prompt_completion_rows(self):
        j = job(self.tmp)
        tr = [sources.Example("refund please", {"category": "billing", "urgent": True})]
        stats = dataset.write(j, self.tmp / "run", tr, tr)
        row = json.loads((self.tmp / "run" / "sft" / "train.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["completion"][0]["content"], '{"category":"billing","urgent":true}')
        self.assertEqual(row["prompt"][1]["content"], "refund please")
        self.assertEqual(stats["train"], 1)

    def test_gate(self):
        j = job(self.tmp)
        ok, _ = runner.gate(j, {"category_acc": 0.5, "valid_json": 1}, {"category_acc": 0.7, "valid_json": 1})
        self.assertTrue(ok)
        ok, _ = runner.gate(j, {"category_acc": 0.7, "valid_json": 1}, {"category_acc": 0.7, "valid_json": 1})
        self.assertFalse(ok)
        ok, _ = runner.gate(j, {"category_acc": 0.5, "valid_json": 1}, {"category_acc": 0.9, "valid_json": 0.5})
        self.assertFalse(ok)

    def test_parse_json_strips_thinking_and_prose(self):
        self.assertEqual(parse_json('<think>hmm</think>\n{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json('Sure: {"a": 2} done'), {"a": 2})
        self.assertIsNone(parse_json("no json"))


if __name__ == "__main__":
    unittest.main()


class LabelTest(unittest.TestCase):
    """Human labels from the labeling page override sources; the page orders and records them."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rows = [
            {"input": "I was charged twice", "category": "billing", "urgent": True, "ts": 1},
            {"input": "The app crashes on start", "category": "bug", "urgent": False, "ts": 2},
            {"input": "Please add dark mode", "ts": 3},  # no label yet
        ]
        (self.tmp / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.extra = f'\n[[source]]\ntype = "jsonl"\nname = "rows"\npath = "{self.tmp}/rows.jsonl"\n'

    def write_labels(self, *records):
        with open(self.tmp / "labels.jsonl", "a") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    def test_unlabeled_rows_only_on_request(self):
        j = job(self.tmp, self.extra)
        self.assertEqual(len(sources.collect(j)), 2)
        rows = sources.collect(j, keep_unlabeled=True)
        self.assertEqual(len(rows), 3)
        self.assertEqual([e.labels for e in rows if e.input.startswith("Please")], [{}])

    def test_human_labels_override_drop_and_undo(self):
        j = job(self.tmp, self.extra)
        k = sources.example_key
        self.write_labels(
            {"key": k("i was  CHARGED twice"), "labels": {"category": "account", "urgent": False}},
            {"key": k("The app crashes on start"), "drop": True},
            {"key": k("Please add dark mode"), "labels": {"category": "feature", "urgent": False}},
            {"key": k("Please add dark mode"), "undo": True},
        )
        rows = {e.input: e for e in sources.collect(j)}
        self.assertEqual(set(rows), {"I was charged twice"})
        e = rows["I was charged twice"]
        self.assertEqual(e.labels, {"category": "account", "urgent": False})
        self.assertTrue(e.origin.startswith("human:"))

    def test_eval_from_human(self):
        j = job(self.tmp, self.extra)
        j.split.update(eval_from="human", min_human_eval=1)
        self.write_labels({"key": sources.example_key("I was charged twice"), "labels": {"category": "billing", "urgent": True}})
        train, ev = dataset.split(j, sources.collect(j))
        self.assertEqual([e.input for e in ev], ["I was charged twice"])
        self.assertEqual([e.input for e in train], ["The app crashes on start"])

    def test_board_orders_and_records(self):
        from lumbridge_train import label

        j = job(self.tmp, self.extra)
        b = label.Board(j)
        k = sources.example_key
        # teacher disagrees with the source on the bug row: it comes first
        b.pre[k("The app crashes on start")] = {"category": "feature", "urgent": False}
        b.pre[k("I was charged twice")] = {"category": "billing", "urgent": True}
        first = b.item(None, set())
        self.assertEqual(first["input"], "The app crashes on start")
        self.assertEqual(first["suggested"]["category"], "feature")
        s = b.save({"key": first["key"], "labels": {"category": "bug", "urgent": False}})
        self.assertEqual(s["labeled"], 1)
        self.assertEqual(s["teacher_agreement"], 0.0)
        self.assertNotEqual(b.item(None, set())["key"], first["key"])
        b.prefer_source = True
        self.assertEqual(b.suggestion(first["key"])["category"], "bug")
        b.prefer_source = False
        b.save({"key": "", "undo": True})
        self.assertEqual(b.stats()["labeled"], 0)
        rec = json.loads((self.tmp / "labels.jsonl").read_text().splitlines()[0])
        self.assertEqual(rec["labels"]["category"], "bug")
        self.assertEqual(oct((self.tmp / "labels.jsonl").stat().st_mode & 0o777), "0o600")


class CloudTest(unittest.TestCase):
    """The cloud runner without the network: offer choice, the data guard, cleanup on failure."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def offer(self, gpu, price, spot, dc="FIN-01", status="Available"):
        return {"cloudId": f"{gpu}{'_SPOT' if spot else ''}", "gpuType": gpu, "socket": "PCIe", "provider": "dc",
                "dataCenter": dc, "country": "FI", "security": "secure_cloud", "stockStatus": status,
                "prices": {"onDemand": price}, "images": ["ubuntu_22_cuda_12"], "isSpot": spot}

    def fake_api(self, offers):
        from lumbridge_train import cloud

        api = cloud.PrimeIntellect.__new__(cloud.PrimeIntellect)
        api.offers = lambda gpu: [o for o in offers if o["gpuType"] == gpu]
        return api

    def test_choose_spot_first_then_on_demand_within_price(self):
        from lumbridge_train import cloud

        api = self.fake_api([
            self.offer("A100_80GB", 2.45, False), self.offer("A100_80GB", 1.25, True),
            self.offer("RTX_PRO_6000B_96GB", 1.43, True), self.offer("H100_80GB", 5.29, False),
            self.offer("RTX_PRO_6000B_96GB", 1.10, True, status="Unavailable"),
        ])
        cfg = {**cloud.DEFAULTS}
        got = [(o["gpuType"], o["isSpot"]) for o in cloud.choose(api, cfg)]
        self.assertEqual(got, [("RTX_PRO_6000B_96GB", True), ("A100_80GB", True), ("A100_80GB", False)])
        cfg["on_demand_fallback"] = False
        self.assertEqual(len(cloud.choose(api, cfg)), 2)

    def test_refuses_without_data_leaves_fleet(self):
        from lumbridge_train import cloud

        j = job(self.tmp, '\n[train.cloud]\ngpus = ["A100_80GB"]\n')
        self.assertTrue(cloud.wanted(j))
        with self.assertRaisesRegex(RuntimeError, "data_leaves_fleet"):
            cloud.train_and_eval(j, self.tmp, "base")

    def test_pod_deleted_on_failure_and_retried_on_preemption(self):
        from unittest import mock

        from lumbridge_train import cloud

        j = job(self.tmp, '\n[train.cloud]\ndata_leaves_fleet = true\nretries = 1\n')
        (self.tmp / "key.pub").write_text("ssh-ed25519 AAAAkey me")
        j.train["cloud"]["ssh_key"] = str(self.tmp / "key")
        offers = [self.offer("RTX_PRO_6000B_96GB", 1.43, True), self.offer("A100_80GB", 2.45, False)]
        deleted, created = [], []
        api = self.fake_api(offers)
        api.ssh_key_id = lambda pub: "k1"
        with mock.patch.object(cloud, "PrimeIntellect", return_value=api), \
             mock.patch.object(cloud, "create", side_effect=lambda *a: created.append(a[3]["cloudId"]) or f"p{len(created)}"), \
             mock.patch.object(cloud, "delete", side_effect=lambda api, job, pid, why: deleted.append((pid, why.split(":")[0]))), \
             mock.patch.object(cloud, "wait_active", return_value=("root", "1.2.3.4", "22")), \
             mock.patch.object(cloud, "_remote_work", side_effect=[cloud.Preempted("gone"), RuntimeError("sft broke")]):
            with self.assertRaisesRegex(RuntimeError, "sft broke"):
                cloud.train_and_eval(j, self.tmp, "base")
        self.assertEqual(created, ["RTX_PRO_6000B_96GB_SPOT", "A100_80GB"])
        self.assertEqual(deleted, [("p1", "preempted"), ("p2", "error")])

    def test_sft_toml_remote_paths(self):
        from lumbridge_train import train

        j = job(self.tmp)
        p = train.sft_toml(j, self.tmp, remote_dir=Path("/root/run"))
        text = p.read_text()
        self.assertIn('output_dir = "/root/run/rl"', text)
        self.assertIn('name = "/root/run/sft"', text)
        self.assertEqual(p.name, "sft-remote.toml")
