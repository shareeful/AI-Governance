from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..checkers.deid import DeIdentifier
from ..execution import Execution, GovernedSystem, Verdict
from ..policy.bind import BoundPolicy
from ..policy.clause import ClauseId
from ..policy.guards import GuardEntry, decide
from ..policy.measures import precondition_compliance, reasoning_faithfulness


@dataclass(frozen=True)
class ReasoningGuardOutcome:
    verdict: Verdict
    measures: dict[str, float]
    entry: GuardEntry


class ReasoningGuard:
    name = "reasoning"

    def __init__(
        self,
        system: GovernedSystem,
        bound_policy: BoundPolicy,
        deidentifier: DeIdentifier,
    ) -> None:
        self.system = system
        self.bound_policy = bound_policy
        self.deidentifier = deidentifier

    def __call__(self, execution: Execution) -> ReasoningGuardOutcome:
        faithfulness_record = self.bound_policy.record(ClauseId.REASONING_FAITHFULNESS)
        precondition_record = self.bound_policy.record(ClauseId.PRECONDITION_COMPLIANCE)
        minimisation_record = self.bound_policy.record(ClauseId.DATA_MINIMISATION)
        binding = self.bound_policy[ClauseId.PRECONDITION_COMPLIANCE]

        faithfulness = reasoning_faithfulness(self.system, execution)
        required = binding.required_checks[execution.action]
        precondition = precondition_compliance(execution, required)

        trace_personal_data = self.deidentifier.extract(execution.trace.text)
        excess = trace_personal_data - binding.minimum_necessary
        minimisation_value = float(len(excess) == 0)

        faithfulness_verdict = decide(faithfulness_record, faithfulness.value)
        precondition_verdict = decide(precondition_record, precondition.value)
        minimisation_verdict = decide(minimisation_record, minimisation_value)

        verdict = (
            Verdict.CERTIFY
            if Verdict.REJECT
            not in {faithfulness_verdict, precondition_verdict, minimisation_verdict}
            else Verdict.REJECT
        )

        measures = {
            "mu_rf": faithfulness.value,
            "mu_pc": precondition.value,
            "mu_dm_trace": minimisation_value,
        }
        thresholds = {
            "theta_rf": float(faithfulness_record.threshold),
            "theta_pc": float(precondition_record.threshold),
            "theta_dm": float(minimisation_record.threshold),
        }
        detail: dict[str, Any] = {
            "faithfulness": dict(faithfulness.detail),
            "precondition": dict(precondition.detail),
            "trace_excess_personal_data": sorted(excess),
            "clause_verdicts": {
                ClauseId.REASONING_FAITHFULNESS.value: faithfulness_verdict.value,
                ClauseId.PRECONDITION_COMPLIANCE.value: precondition_verdict.value,
                ClauseId.DATA_MINIMISATION.value: minimisation_verdict.value,
            },
        }
        entry = GuardEntry(
            guard=self.name,
            clauses=(ClauseId.REASONING_FAITHFULNESS, ClauseId.PRECONDITION_COMPLIANCE),
            measures=measures,
            thresholds=thresholds,
            verdict=verdict,
            evidence_digest=execution.trace_digest(),
            detail=detail,
        )
        return ReasoningGuardOutcome(verdict=verdict, measures=measures, entry=entry)
