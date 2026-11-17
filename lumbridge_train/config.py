"""Job config: one TOML file describes a decision job end to end.

A job says what the model must decide (fields and their choices), where real examples come
from (sources), how to top them up with synthetic ones (a teacher model), which small model
to train, and how the new version must beat the old one before it is promoted.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


@dataclass
class Field:
    name: str
    type: str = "choice"  # choice | bool
    choices: list[str] = field(default_factory=list)
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class Teacher:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    concurrency: int = 2
    timeout_s: float = 120.0
    extra_body: dict = field(default_factory=dict)


@dataclass
class Job:
    name: str
    instructions: str
    fields: list[Field]
    sources: list[dict]
    teachers: dict[str, Teacher]
    synth: dict
    split: dict
    train: dict
    gate: dict
    schedule: dict
    workdir: Path
    path: Path

    @property
    def primary(self) -> Field:
        return self.fields[0]

    def system_prompt(self) -> str:
        lines = [self.instructions.strip(), "", "Reply with one JSON object and nothing else:"]
        for f in self.fields:
            if f.type == "bool":
                lines.append(f'- "{f.name}": true or false')
            else:
                lines.append(f'- "{f.name}": one of {", ".join(f.choices)}')
                for c, d in f.descriptions.items():
                    lines.append(f"    {c}: {d}")
        return "\n".join(lines)


def load(path: str | os.PathLike) -> Job:
    path = expand(path)
    raw = tomllib.loads(path.read_text())
    j = raw["job"]
    fields = [Field(**f) for f in j["fields"]]
    teachers = {
        name: Teacher(name=name, **t) for name, t in raw.get("teachers", {}).items()
    }
    schedule = raw.get("schedule", {})
    day = schedule.get("training_day", "saturday").lower()
    if day not in DAYS:
        raise ValueError(f"schedule.training_day must be one of {DAYS}, got {day!r}")
    schedule["training_day"] = day
    return Job(
        name=j["name"],
        instructions=j["instructions"],
        fields=fields,
        sources=raw.get("source", []),
        teachers=teachers,
        synth=raw.get("synth", {}),
        split=raw.get("split", {}),
        train=raw.get("train", {}),
        gate=raw.get("gate", {}),
        schedule=schedule,
        workdir=expand(raw.get("workdir", f"~/lumbridge-train-runs/{j['name']}")),
        path=path,
    )
