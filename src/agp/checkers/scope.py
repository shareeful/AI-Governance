from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..errors import ConfigurationError


@dataclass(frozen=True)
class ActionRegisterEntry:
    action: str
    high_risk: bool
    required_checks: frozenset[str]


@dataclass(frozen=True)
class ActionRegister:
    entries: Mapping[str, ActionRegisterEntry]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Mapping[str, object]]) -> "ActionRegister":
        if not raw:
            raise ConfigurationError("the permitted-action register must not be empty")
        entries: dict[str, ActionRegisterEntry] = {}
        for action, spec in raw.items():
            if not isinstance(spec, Mapping):
                raise ConfigurationError(f"action register entry '{action}' must be a mapping")
            if "high_risk" not in spec:
                raise ConfigurationError(
                    f"action register entry '{action}' must declare 'high_risk'"
                )
            checks = spec.get("required_checks")
            if not isinstance(checks, list) or not checks:
                raise ConfigurationError(
                    f"action register entry '{action}' must declare a non-empty "
                    "'required_checks' list"
                )
            entries[action] = ActionRegisterEntry(
                action=action,
                high_risk=bool(spec["high_risk"]),
                required_checks=frozenset(str(c) for c in checks),
            )
        return cls(entries=entries)

    def permits(self, action: str | None) -> bool:
        return action is not None and action in self.entries

    def is_high_risk(self, action: str) -> bool:
        if action not in self.entries:
            raise ConfigurationError(f"action '{action}' is absent from the action register")
        return self.entries[action].high_risk

    def required_checks(self, action: str) -> frozenset[str]:
        if action not in self.entries:
            raise ConfigurationError(f"action '{action}' is absent from the action register")
        return self.entries[action].required_checks

    def __contains__(self, action: object) -> bool:
        return action in self.entries

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


class ScopeChecker:
    def __init__(self, register: ActionRegister) -> None:
        self.register = register

    def in_scope(self, requested_action: str | None) -> bool:
        return self.register.permits(requested_action)
