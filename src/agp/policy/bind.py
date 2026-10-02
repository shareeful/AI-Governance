from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..errors import BindingError
from ..execution import GovernedSystem
from .clause import ClauseId, ClausePolicy, ClauseRecord, ExecutionComponent


@dataclass(frozen=True)
class ClauseBinding:
    record: ClauseRecord
    system: GovernedSystem
    protected_attributes: tuple[str, ...]
    required_checks: Mapping[str, frozenset[str]]
    minimum_necessary: frozenset[str]


@dataclass(frozen=True)
class BoundPolicy:
    policy: ClausePolicy
    system: GovernedSystem
    bindings: Mapping[ClauseId, ClauseBinding]

    def __getitem__(self, clause: ClauseId) -> ClauseBinding:
        return self.bindings[clause]

    def record(self, clause: ClauseId) -> ClauseRecord:
        return self.bindings[clause].record


def _verify_reexecution_capability(system: GovernedSystem) -> None:
    for capability in ("action_for_input", "action_for_trace", "substitute_attribute"):
        attribute = getattr(type(system), capability, None)
        if attribute is None or getattr(attribute, "__isabstractmethod__", False):
            raise BindingError(
                f"system '{system.name}' does not implement '{capability}'; the counterfactual "
                "and ablation runners require controlled re-execution"
            )


def bind_policy(
    policy: ClausePolicy,
    system: GovernedSystem,
    protected_attributes: tuple[str, ...],
    action_register: Mapping[str, object],
) -> BoundPolicy:
    if any(policy[c].requires_reexecution for c in ClauseId):
        _verify_reexecution_capability(system)

    if not protected_attributes:
        raise BindingError(
            f"system '{system.name}' declares no protected attributes; the non-discrimination "
            "clause cannot be bound without at least one documented attribute"
        )
    for attribute in protected_attributes:
        domain = system.protected_attribute_domain(attribute)
        if len(domain) == 0:
            raise BindingError(
                f"system '{system.name}': protected attribute '{attribute}' has an empty domain"
            )

    if not action_register:
        raise BindingError(
            f"system '{system.name}' declares an empty action register; the precondition clause "
            "requires a required-check list per permitted action"
        )

    required_checks: dict[str, frozenset[str]] = {}
    for action in action_register:
        checks = system.required_checks(action)
        if not checks:
            raise BindingError(
                f"system '{system.name}': action '{action}' declares no required checks; "
                "the precondition clause would certify vacuously"
            )
        required_checks[action] = checks

    minimum_necessary = system.minimum_necessary_personal_data()

    bindings: dict[ClauseId, ClauseBinding] = {}
    for clause in ClauseId:
        record = policy[clause]
        bindings[clause] = ClauseBinding(
            record=record,
            system=system,
            protected_attributes=protected_attributes,
            required_checks=required_checks,
            minimum_necessary=minimum_necessary,
        )

    _verify_read_access(policy)
    return BoundPolicy(policy=policy, system=system, bindings=bindings)


def _verify_read_access(policy: ClausePolicy) -> None:
    expected: dict[ClauseId, frozenset[ExecutionComponent]] = {
        ClauseId.NON_DISCRIMINATION: frozenset({ExecutionComponent.INPUT, ExecutionComponent.ACTION}),
        ClauseId.DATA_MINIMISATION: frozenset(
            {ExecutionComponent.INPUT, ExecutionComponent.TRACE, ExecutionComponent.RESPONSE}
        ),
        ClauseId.REASONING_FAITHFULNESS: frozenset(
            {ExecutionComponent.TRACE, ExecutionComponent.ACTION}
        ),
        ClauseId.PRECONDITION_COMPLIANCE: frozenset(
            {ExecutionComponent.TRACE, ExecutionComponent.ACTION}
        ),
        ClauseId.FACTUAL_GROUNDEDNESS: frozenset(
            {ExecutionComponent.INPUT, ExecutionComponent.RESPONSE}
        ),
    }
    for clause, allowed in expected.items():
        declared = frozenset(policy[clause].read_access)
        if not declared <= allowed:
            excess = ", ".join(sorted(c.value for c in declared - allowed))
            raise BindingError(
                f"clause '{clause.value}' declares read access to components it may not read: "
                f"{excess}"
            )
