from __future__ import annotations

from typing import Any, Mapping

from ..config import Config
from ..errors import ConfigurationError
from .clause import (
    EXACT_CLAUSES,
    LEARNED_CLAUSES,
    ClauseId,
    ClausePolicy,
    ClauseRecord,
    DecisionKind,
    ExecutionComponent,
    FailureAction,
)

_MEASURE_SYMBOL: dict[ClauseId, str] = {
    ClauseId.NON_DISCRIMINATION: "mu_nd",
    ClauseId.DATA_MINIMISATION: "mu_dm",
    ClauseId.REASONING_FAITHFULNESS: "mu_rf",
    ClauseId.PRECONDITION_COMPLIANCE: "mu_pc",
    ClauseId.FACTUAL_GROUNDEDNESS: "mu_fg",
}

_REEXECUTING_CLAUSES: frozenset[ClauseId] = frozenset(
    {ClauseId.NON_DISCRIMINATION, ClauseId.REASONING_FAITHFULNESS}
)


def _parse_components(raw: Any, clause: ClauseId) -> tuple[ExecutionComponent, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigurationError(
            f"clause '{clause.value}': read_access must be a non-empty list of execution components"
        )
    components: list[ExecutionComponent] = []
    for item in raw:
        try:
            components.append(ExecutionComponent(item))
        except ValueError as exc:
            valid = ", ".join(c.value for c in ExecutionComponent)
            raise ConfigurationError(
                f"clause '{clause.value}': unknown execution component {item!r}; valid: {valid}"
            ) from exc
    return tuple(components)


def _parse_failure_actions(raw: Any, clause: ClauseId) -> tuple[FailureAction, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigurationError(
            f"clause '{clause.value}': failure_action must be a non-empty list"
        )
    actions: list[FailureAction] = []
    for item in raw:
        try:
            actions.append(FailureAction(item))
        except ValueError as exc:
            valid = ", ".join(a.value for a in FailureAction)
            raise ConfigurationError(
                f"clause '{clause.value}': unknown failure action {item!r}; valid: {valid}"
            ) from exc
    return tuple(actions)


def compile_clause(clause: ClauseId, spec: Mapping[str, Any]) -> ClauseRecord:
    if clause in EXACT_CLAUSES:
        decision_kind = DecisionKind.EXACT
        threshold: float | None = float(spec["threshold"]) if "threshold" in spec else 1.0
    elif clause in LEARNED_CLAUSES:
        decision_kind = DecisionKind.LEARNED
        threshold = None
        if spec.get("threshold") is not None:
            raise ConfigurationError(
                f"clause '{clause.value}' is decided by a learned measure; its threshold is set "
                "by Learn-then-Test in Phase 2 and must not be fixed in configuration"
            )
    else:
        raise ConfigurationError(f"clause '{clause.value}' belongs to no decision family")

    for field_name in ("regulatory_basis", "checker", "read_access", "failure_action"):
        if field_name not in spec:
            raise ConfigurationError(
                f"clause '{clause.value}': required field '{field_name}' is absent from the "
                "compiled record"
            )

    return ClauseRecord(
        clause=clause,
        regulatory_basis=str(spec["regulatory_basis"]),
        measure=_MEASURE_SYMBOL[clause],
        decision_kind=decision_kind,
        threshold=threshold,
        checker=str(spec["checker"]),
        read_access=_parse_components(spec["read_access"], clause),
        failure_action=_parse_failure_actions(spec["failure_action"], clause),
        requires_reexecution=clause in _REEXECUTING_CLAUSES,
        parameters=dict(spec.get("parameters") or {}),
    )


def compile_policy(config: Config) -> ClausePolicy:
    section = config.get("policy.clauses")
    if not isinstance(section, Mapping):
        raise ConfigurationError("policy.clauses must be a mapping of clause identifiers to records")
    records: dict[ClauseId, ClauseRecord] = {}
    for raw_name, spec in section.items():
        try:
            clause = ClauseId(raw_name)
        except ValueError as exc:
            valid = ", ".join(c.value for c in ClauseId)
            raise ConfigurationError(
                f"unknown clause identifier {raw_name!r} in policy.clauses; valid: {valid}"
            ) from exc
        if not isinstance(spec, Mapping):
            raise ConfigurationError(f"policy.clauses.{raw_name} must be a mapping")
        records[clause] = compile_clause(clause, spec)
    return ClausePolicy(records=records)
