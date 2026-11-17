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
