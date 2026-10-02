from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..checkers.deid import DeIdentifier
from ..checkers.nli import EntailmentChecker
from ..checkers.safety import SafetyChecker
from ..execution import Execution, GovernedSystem, Input, Verdict
from ..guards.input_guard import SegregatedAttributeStore
from ..policy.bind import BoundPolicy
from ..policy.clause import ClauseId
from ..policy.guards import GuardEntry, decide
from ..policy.measures import counterfactual_invariance, factual_groundedness


@dataclass(frozen=True)
class OutputGuardOutcome:
    verdict: Verdict
    measures: dict[str, float]
    entry: GuardEntry


class OutputGuard:
    name = "output"

    def __init__(
        self,
        system: GovernedSystem,
        bound_policy: BoundPolicy,
        entailment: EntailmentChecker,
        safety: SafetyChecker,
        deidentifier: DeIdentifier,
    ) -> None:
        self.system = system
        self.bound_policy = bound_policy
        self.entailment = entailment
        self.safety = safety
        self.deidentifier = deidentifier

    def __call__(
        self,
        execution: Execution,
        original_input: Input,
        attribute_store: SegregatedAttributeStore,
    ) -> OutputGuardOutcome:
        nd_record = self.bound_policy.record(ClauseId.NON_DISCRIMINATION)
        fg_record = self.bound_policy.record(ClauseId.FACTUAL_GROUNDEDNESS)
        dm_record = self.bound_policy.record(ClauseId.DATA_MINIMISATION)
        binding = self.bound_policy[ClauseId.NON_DISCRIMINATION]

        attributes = tuple(
            attribute
            for attribute in binding.protected_attributes
            if attribute in attribute_store.attributes()
        )
        invariance = counterfactual_invariance(
            system=self.system,
            request=original_input,
            realised_action=execution.action,
            attributes=attributes,
        )
        groundedness = factual_groundedness(execution, self.entailment)

        response_personal_data = self.deidentifier.extract(execution.response.text)
        excess = response_personal_data - binding.minimum_necessary
        minimisation_value = float(len(excess) == 0)

        safety_verdict = self.safety.check(execution.response.text)

        nd_verdict = decide(nd_record, invariance.value)
        fg_verdict = decide(fg_record, groundedness.value)
        dm_verdict = decide(dm_record, minimisation_value)
        safe = Verdict.REJECT if safety_verdict.unsafe else Verdict.CERTIFY

        verdict = (
            Verdict.CERTIFY
            if Verdict.REJECT not in {nd_verdict, fg_verdict, dm_verdict, safe}
            else Verdict.REJECT
        )

        measures = {
            "mu_nd": invariance.value,
            "mu_fg": groundedness.value,
            "mu_dm_response": minimisation_value,
            "safety_score": safety_verdict.score,
        }
        thresholds = {
            "theta_nd": float(nd_record.threshold),
            "theta_fg": float(fg_record.threshold),
            "theta_dm": float(dm_record.threshold),
            "theta_safety": self.safety.threshold,
        }
        detail: dict[str, Any] = {
            "non_discrimination": dict(invariance.detail),
            "groundedness": dict(groundedness.detail),
            "response_excess_personal_data": sorted(excess),
            "clause_verdicts": {
                ClauseId.NON_DISCRIMINATION.value: nd_verdict.value,
                ClauseId.FACTUAL_GROUNDEDNESS.value: fg_verdict.value,
                ClauseId.DATA_MINIMISATION.value: dm_verdict.value,
                "content_safety": safe.value,
            },
        }
        entry = GuardEntry(
            guard=self.name,
            clauses=(ClauseId.NON_DISCRIMINATION, ClauseId.FACTUAL_GROUNDEDNESS),
            measures=measures,
            thresholds=thresholds,
            verdict=verdict,
            evidence_digest=execution.response_digest(),
            detail=detail,
        )
        return OutputGuardOutcome(verdict=verdict, measures=measures, entry=entry)
