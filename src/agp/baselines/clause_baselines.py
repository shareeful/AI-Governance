from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from ..benchmarks.embedding import ProbeCase
from ..checkers.deid import DeIdentifier
from ..checkers.injection import InjectionJailbreakChecker
from ..checkers.nli import EntailmentChecker
from ..checkers.safety import SafetyChecker
from ..errors import ConfigurationError
from ..execution import Execution, Verdict
from ..policy.bind import BoundPolicy
from ..policy.clause import ClauseId
from ..policy.guards import decide
from ..policy.measures import precondition_compliance
from ..runtime.pipeline import GovernancePipeline

CLAUSE_BASELINES = (
    "ungoverned",
    "input_only",
    "output_only",
    "trace_aware",
    "action_level",
    "reference_based",
    "proposed",
)


class ClauseBaseline(ABC):
    name: str

    @abstractmethod
    def blocks(self, probe: ProbeCase) -> bool:
        raise NotImplementedError


class UngovernedBaseline(ClauseBaseline):
    name = "ungoverned"

    def blocks(self, probe: ProbeCase) -> bool:
        return False


@dataclass
class InputOnlyBaseline(ClauseBaseline):
    injection: InjectionJailbreakChecker
    deidentifier: DeIdentifier
    minimum_necessary: frozenset[str]
    name: str = "input_only"

    def blocks(self, probe: ProbeCase) -> bool:
        verdict = self.injection.check(probe.execution.inputs.request)
        if verdict.is_attack:
            return True
        observed = self.deidentifier.extract(probe.execution.inputs.request)
        return bool(observed - self.minimum_necessary)


@dataclass
class OutputOnlyBaseline(ClauseBaseline):
    safety: SafetyChecker
    entailment: EntailmentChecker
    deidentifier: DeIdentifier
    minimum_necessary: frozenset[str]
    name: str = "output_only"

    def blocks(self, probe: ProbeCase) -> bool:
        execution = probe.execution
        if self.safety.check(execution.response.text).unsafe:
            return True
        observed = self.deidentifier.extract(execution.response.text)
        if observed - self.minimum_necessary:
            return True
        return not self._grounded(execution)

    def _grounded(self, execution: Execution) -> bool:
        evidence = "\n".join(str(item.content) for item in execution.inputs.evidence)
        if not evidence.strip():
            return True
        return all(
            self.entailment.entails(evidence, claim.statement)
            for claim in execution.response.claims
        )


@dataclass
class TraceAwareBaseline(ClauseBaseline):
    safety: SafetyChecker
    entailment: EntailmentChecker
    deidentifier: DeIdentifier
    minimum_necessary: frozenset[str]
    name: str = "trace_aware"

    def blocks(self, probe: ProbeCase) -> bool:
        execution = probe.execution
        surfaces = (
            execution.inputs.request,
            execution.trace.text,
            execution.response.text,
        )
        for surface in surfaces:
            if self.safety.check(surface).unsafe:
                return True
            if self.deidentifier.extract(surface) - self.minimum_necessary:
                return True
        evidence = "\n".join(str(item.content) for item in execution.inputs.evidence)
        if evidence.strip():
            for claim in execution.response.claims:
                if not self.entailment.entails(evidence, claim.statement):
                    return True
            for reason in execution.trace.reasons:
                if not self.entailment.entails(evidence, reason.statement):
                    return True
        return False


@dataclass
class ActionLevelBaseline(ClauseBaseline):
    bound_policy: BoundPolicy
    name: str = "action_level"

    def blocks(self, probe: ProbeCase) -> bool:
        execution = probe.execution
        if not self.bound_policy.system.action_register.permits(execution.action):
            return True
        record = self.bound_policy.record(ClauseId.PRECONDITION_COMPLIANCE)
        required = self.bound_policy[ClauseId.PRECONDITION_COMPLIANCE].required_checks[
            execution.action
        ]
        measure = precondition_compliance(execution, required)
        if decide(record, measure.value) is Verdict.REJECT:
            return True
        return self.bound_policy.system.action_register.is_high_risk(execution.action) and (
            measure.value < 1.0
        )


@dataclass
class ReferenceBasedBaseline(ClauseBaseline):
    pipeline: GovernancePipeline
    name: str = "reference_based"

    def blocks(self, probe: ProbeCase) -> bool:
        reasoning = self.pipeline.reasoning_guard(probe.execution)
        if reasoning.verdict is Verdict.REJECT:
            return True
        output = self.pipeline.output_guard(
            probe.execution,
            probe.execution.inputs,
            _store_for(probe.execution),
        )
        return output.verdict is Verdict.REJECT


@dataclass
class ProposedPlane(ClauseBaseline):
    pipeline: GovernancePipeline
    name: str = "proposed"

    def blocks(self, probe: ProbeCase) -> bool:
        from ..execution import Decision

        result = self.pipeline(probe.execution.inputs)
        return result.decision is Decision.WITHHOLD


def _store_for(execution: Execution):
    from ..guards.input_guard import SegregatedAttributeStore

    return SegregatedAttributeStore(values=dict(execution.protected_attributes))


def build_clause_baselines(
    names: Sequence[str],
    injection: InjectionJailbreakChecker,
    safety: SafetyChecker,
    entailment: EntailmentChecker,
    deidentifier: DeIdentifier,
    bound_policy: BoundPolicy,
    pipeline: GovernancePipeline,
    minimum_necessary: frozenset[str],
) -> list[ClauseBaseline]:
    built: list[ClauseBaseline] = []
    for name in names:
        if name == "ungoverned":
            built.append(UngovernedBaseline())
        elif name == "input_only":
            built.append(InputOnlyBaseline(injection, deidentifier, minimum_necessary))
        elif name == "output_only":
            built.append(
                OutputOnlyBaseline(safety, entailment, deidentifier, minimum_necessary)
            )
        elif name == "trace_aware":
            built.append(
                TraceAwareBaseline(safety, entailment, deidentifier, minimum_necessary)
            )
        elif name == "action_level":
            built.append(ActionLevelBaseline(bound_policy))
        elif name == "reference_based":
            built.append(ReferenceBasedBaseline(pipeline))
        elif name == "proposed":
            built.append(ProposedPlane(pipeline))
        else:
            raise ConfigurationError(
                f"unknown clause baseline {name!r}; valid baselines are "
                + ", ".join(CLAUSE_BASELINES)
            )
    return built
