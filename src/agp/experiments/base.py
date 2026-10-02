from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import Config
from ..errors import ConfigurationError


@dataclass
class ExperimentResult:
    name: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, **row: Any) -> None:
        self.rows.append(row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.name,
            "rows": self.rows,
            "comparisons": self.comparisons,
            "metadata": self.metadata,
        }

    def write(self, root: Path) -> Path:
        path = root / "results" / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, default=str)
        return path


def seeds_from(config: Config) -> tuple[int, ...]:
    values = config.list_value("experiments.seeds")
    seeds = tuple(int(v) for v in values)
    if len(set(seeds)) != len(seeds):
        raise ConfigurationError("experiments.seeds must not repeat a seed")
    return seeds


def systems_from(config: Config) -> tuple[str, ...]:
    return tuple(str(name) for name in config.list_value("experiments.systems"))


def artifact_root(config: Config) -> Path:
    return Path(config.str_value("paths.artifact_root")).expanduser()


def bootstrap_settings(config: Config) -> tuple[int, float]:
    return (
        config.int_value("experiments.bootstrap.resamples"),
        config.require_probability("experiments.bootstrap.confidence"),
    )


def family_wise_alpha(config: Config) -> float:
    return config.require_probability("experiments.significance.family_wise_alpha")


def modality_of(config: Config, system_name: str) -> str:
    return config.str_value(f"systems.{system_name}.modality")


def pooled(values: Mapping[str, Sequence[float]]) -> dict[str, float]:
    return {key: float(np.mean(list(sample))) for key, sample in values.items()}


def require_stream_length(count: int, config: Config, label: str) -> None:
    window = config.int_value("attestation.window")
    minimum_windows = config.int_value("experiments.minimum_detector_windows")
    required = window + minimum_windows - 1
    if count < required:
        raise ConfigurationError(
            f"{label} holds {count} executions, but the attestation detector spans a window of "
            f"{window} and experiments.minimum_detector_windows requires {minimum_windows} "
            f"sliding windows, so at least {required} executions are needed. Enlarge the split, "
            "shorten attestation.window, or lower experiments.minimum_detector_windows."
        )
