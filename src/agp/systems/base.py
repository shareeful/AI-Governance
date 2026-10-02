from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import Config
from ..errors import ConfigurationError, InterfaceError
from ..execution import (
    Claim,
    EvidenceItem,
    Execution,
    GovernedSystem,
    Input,
    Reason,
    ReasoningTrace,
    Response,
)
from ..checkers.scope import ActionRegister


@dataclass(frozen=True)
class RationaleTemplates:
    reason: str
    claim: str
    evidence: str
    check: str

    @classmethod
    def from_config(cls, config: Config, prefix: str) -> "RationaleTemplates":
        templates = cls(
            reason=config.str_value(f"{prefix}.reason"),
            claim=config.str_value(f"{prefix}.claim"),
            evidence=config.str_value(f"{prefix}.evidence"),
            check=config.str_value(f"{prefix}.check"),
        )
        for name, template, placeholders in (
            ("reason", templates.reason, ("{group}", "{mass}")),
            ("claim", templates.claim, ("{group}", "{action}")),
            ("evidence", templates.evidence, ("{group}", "{mass}")),
            ("check", templates.check, ("{check}",)),
        ):
            for placeholder in placeholders:
                if placeholder not in template:
                    raise ConfigurationError(
                        f"{prefix}.{name} must contain the {placeholder} placeholder"
                    )
        return templates


class InstrumentedSystem(GovernedSystem):
    def __init__(
        self,
        name: str,
        modality: str,
        action_register: ActionRegister,
        feature_groups: tuple[str, ...],
        protected_attributes: tuple[str, ...],
        attribute_domains: Mapping[str, Sequence[Any]],
        minimum_necessary: frozenset[str],
        templates: RationaleTemplates,
        reason_count: int,
        recorded_check_policy: str,
    ) -> None:
        if not feature_groups:
            raise ConfigurationError(f"system '{name}' declares no feature groups")
        if reason_count < 1:
            raise ConfigurationError(f"system '{name}': reason_count must be at least 1")
        if reason_count > len(feature_groups):
            raise ConfigurationError(
                f"system '{name}': reason_count ({reason_count}) exceeds the number of "
                f"feature groups ({len(feature_groups)})"
            )
        if recorded_check_policy not in {"complete", "as_evaluated"}:
            raise ConfigurationError(
                f"system '{name}': recorded_check_policy must be 'complete' or 'as_evaluated'"
            )
        self.name = name
        self.modality = modality
        self.action_register = action_register
        self.feature_groups = feature_groups
        self.protected_attributes = protected_attributes
        self.attribute_domains = {k: tuple(v) for k, v in attribute_domains.items()}
        self.minimum_necessary = minimum_necessary
        self.templates = templates
        self.reason_count = reason_count
        self.recorded_check_policy = recorded_check_policy

    @abstractmethod
    def predict_action(self, request: Input) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict_action_with_groups_removed(
        self, request: Input, removed_groups: frozenset[str]
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def recorded_checks(self, request: Input, action: str) -> tuple[str, ...]:
        raise NotImplementedError

    def protected_attribute_domain(self, attribute: str) -> Sequence[Any]:
        if attribute not in self.attribute_domains:
            raise InterfaceError(
                f"system '{self.name}' declares no domain for protected attribute '{attribute}'"
            )
        return self.attribute_domains[attribute]

    def required_checks(self, action: str) -> frozenset[str]:
        return self.action_register.required_checks(action)

    def minimum_necessary_personal_data(self) -> frozenset[str]:
        return self.minimum_necessary

    def action_for_input(self, request: Input) -> str:
        return self.predict_action(request)

    def action_for_trace(self, request: Input, trace: ReasoningTrace) -> str:
        retained = {reason.support["group"] for reason in trace.reasons}
        stated = self._stated_groups(request)
        removed = frozenset(stated) - retained
        return self.predict_action_with_groups_removed(request, removed)

    def _stated_groups(self, request: Input) -> tuple[str, ...]:
        attribution = self.attribution(request)
        ranked = sorted(self.feature_groups, key=lambda g: abs(attribution[g]), reverse=True)
        return tuple(ranked[: self.reason_count])

    def build_trace(self, request: Input, action: str) -> ReasoningTrace:
        attribution = self.attribution(request)
        groups = self._stated_groups(request)
        reasons = tuple(
            Reason(
                identifier=f"reason:{group}",
                statement=self.templates.reason.format(
                    group=group, mass=f"{abs(attribution[group]):.6f}", action=action
                ),
                support={"group": group, "mass": float(abs(attribution[group]))},
            )
            for group in groups
        )
        checks = self.recorded_checks(request, action)
        check_statements = [self.templates.check.format(check=check) for check in checks]
        text = "\n".join([*(r.statement for r in reasons), *check_statements])
        return ReasoningTrace(
            reasons=reasons,
            recorded_checks=checks,
            commit_index=len(checks),
            text=text,
        )

    def build_evidence(self, request: Input) -> tuple[EvidenceItem, ...]:
        if request.evidence:
            return request.evidence
        attribution = self.attribution(request)
        return tuple(
            EvidenceItem(
                identifier=f"evidence:{group}",
                content=self.templates.evidence.format(
                    group=group, mass=f"{abs(attribution[group]):.6f}"
                ),
                provenance=str(request.metadata.get("provenance", "")) or None,
                modality=self.modality,
            )
            for group in self.feature_groups
        )

    def build_response(self, request: Input, action: str, trace: ReasoningTrace) -> Response:
        claims = tuple(
            Claim(
                identifier=f"claim:{reason.support['group']}",
                statement=self.templates.claim.format(
                    group=reason.support["group"],
                    action=action,
                    mass=f"{reason.support['mass']:.6f}",
                ),
                evidence_ids=(f"evidence:{reason.support['group']}",),
            )
            for reason in trace.reasons
        )
        return Response(claims=claims, text="\n".join(c.statement for c in claims))

    def execute(self, request: Input) -> Execution:
        evidence = self.build_evidence(request)
        grounded_request = replace(request, evidence=evidence)
        action = self.predict_action(grounded_request)
        trace = self.build_trace(grounded_request, action)
        response = self.build_response(grounded_request, action, trace)
        return Execution(
            identifier=Execution.new_identifier(),
            inputs=grounded_request,
            trace=trace,
            action=action,
            response=response,
            protected_attributes={
                attribute: request.metadata[attribute]
                for attribute in self.protected_attributes
                if attribute in request.metadata
            },
        )

    def attribution_vector(self, request: Input) -> np.ndarray:
        attribution = self.attribution(request)
        return np.array([abs(float(attribution[g])) for g in self.feature_groups], dtype=float)
