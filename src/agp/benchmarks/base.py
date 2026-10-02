from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..errors import BenchmarkError, MissingResourceError


@dataclass(frozen=True)
class BenchmarkCase:
    identifier: str
    payload: Mapping[str, Any]
    violates: bool
    source: str


@dataclass(frozen=True)
class BenchmarkSet:
    name: str
    clause: str
    cases: tuple[BenchmarkCase, ...]
    external: bool

    def __len__(self) -> int:
        return len(self.cases)

    def violating(self) -> tuple[BenchmarkCase, ...]:
        return tuple(case for case in self.cases if case.violates)

    def compliant(self) -> tuple[BenchmarkCase, ...]:
        return tuple(case for case in self.cases if not case.violates)

    def require_both_classes(self) -> None:
        if not self.violating():
            raise BenchmarkError(
                f"benchmark '{self.name}' contains no violating case; recall cannot be measured"
            )
        if not self.compliant():
            raise BenchmarkError(
                f"benchmark '{self.name}' contains no compliant case; the false-block rate "
                "cannot be measured"
            )


def read_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    if not path.is_file():
        raise MissingResourceError(
            f"benchmark file not found at {path}. Download the published benchmark from its "
            "original source and set its path in configuration; the framework does not "
            "generate benchmark cases."
        )
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"line {line_number} of {path} is not valid JSON") from exc


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise MissingResourceError(
            f"benchmark file not found at {path}. Download the published benchmark from its "
            "original source and set its path in configuration."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_fields(row: Mapping[str, Any], fields: tuple[str, ...], source: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise BenchmarkError(
            f"{source}: record lacks required fields: " + ", ".join(missing)
        )
