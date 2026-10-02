from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .errors import ConfigurationError, MissingResourceError

_UNSET = object()


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    source: tuple[Path, ...]

    @classmethod
    def load(cls, *paths: str | Path) -> "Config":
        if not paths:
            raise ConfigurationError("at least one configuration file must be supplied")
        merged: dict[str, Any] = {}
        resolved: list[Path] = []
        for path in paths:
            file_path = Path(path).expanduser().resolve()
            if not file_path.is_file():
                raise MissingResourceError(f"configuration file not found: {file_path}")
            with file_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                raise ConfigurationError(f"configuration root must be a mapping: {file_path}")
            merged = _deep_merge(merged, loaded)
            resolved.append(file_path)
        return cls(data=merged, source=tuple(resolved))

    def get(self, dotted_key: str, default: Any = _UNSET) -> Any:
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _UNSET:
                    raise ConfigurationError(
                        f"required configuration key '{dotted_key}' is absent; "
                        f"set it in one of {[str(p) for p in self.source]}"
                    )
                return default
            node = node[part]
        if node is None and default is _UNSET:
            raise ConfigurationError(
                f"configuration key '{dotted_key}' is null; it must be set explicitly "
                f"in one of {[str(p) for p in self.source]}"
            )
        return node

    def section(self, dotted_key: str) -> "Config":
        node = self.get(dotted_key)
        if not isinstance(node, Mapping):
            raise ConfigurationError(f"configuration key '{dotted_key}' is not a mapping")
        return Config(data=dict(node), source=self.source)

    def has(self, dotted_key: str) -> bool:
        return self.get(dotted_key, default=None) is not None

    def path(self, dotted_key: str, must_exist: bool = True) -> Path:
        raw = self.get(dotted_key)
        if not isinstance(raw, str):
            raise ConfigurationError(f"configuration key '{dotted_key}' must be a filesystem path")
        resolved = Path(raw).expanduser()
        if must_exist and not resolved.exists():
            raise MissingResourceError(
                f"path configured at '{dotted_key}' does not exist: {resolved}. "
                "The framework operates only on real datasets and artefacts; "
                "no substitute data is generated."
            )
        return resolved

    def float_value(self, dotted_key: str) -> float:
        value = self.get(dotted_key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(f"configuration key '{dotted_key}' must be a number")
        return float(value)

    def int_value(self, dotted_key: str) -> int:
        value = self.get(dotted_key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(f"configuration key '{dotted_key}' must be an integer")
        return value

    def bool_value(self, dotted_key: str) -> bool:
        value = self.get(dotted_key)
        if not isinstance(value, bool):
            raise ConfigurationError(f"configuration key '{dotted_key}' must be a boolean")
        return value

    def str_value(self, dotted_key: str) -> str:
        value = self.get(dotted_key)
        if not isinstance(value, str):
            raise ConfigurationError(f"configuration key '{dotted_key}' must be a string")
        return value

    def list_value(self, dotted_key: str) -> list[Any]:
        value = self.get(dotted_key)
        if not isinstance(value, list):
            raise ConfigurationError(f"configuration key '{dotted_key}' must be a list")
        if not value:
            raise ConfigurationError(f"configuration key '{dotted_key}' must not be empty")
        return list(value)

    def keys(self, dotted_key: str) -> list[str]:
        node = self.get(dotted_key)
        if not isinstance(node, Mapping):
            raise ConfigurationError(f"configuration key '{dotted_key}' is not a mapping")
        return list(node.keys())

    def require_probability(self, dotted_key: str) -> float:
        value = self.float_value(dotted_key)
        if not 0.0 < value < 1.0:
            raise ConfigurationError(
                f"configuration key '{dotted_key}' must lie strictly within (0, 1); got {value}"
            )
        return value

    def require_unit_interval(self, dotted_key: str) -> float:
        value = self.float_value(dotted_key)
        if not 0.0 <= value <= 1.0:
            raise ConfigurationError(
                f"configuration key '{dotted_key}' must lie within [0, 1]; got {value}"
            )
        return value


def require_sum_to_one(weights: Mapping[str, float], tolerance: float, label: str) -> None:
    total = float(sum(weights.values()))
    if abs(total - 1.0) > tolerance:
        raise ConfigurationError(
            f"{label} must sum to 1.0 within {tolerance}; got {total:.6f} "
            f"over {sorted(weights)}"
        )


def require_disjoint(groups: Mapping[str, Iterable[Any]], label: str) -> None:
    seen: dict[Any, str] = {}
    for name, members in groups.items():
        for member in members:
            if member in seen:
                raise ConfigurationError(
                    f"{label}: element {member!r} appears in both "
                    f"'{seen[member]}' and '{name}'; splits must be disjoint"
                )
            seen[member] = name
