from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..attestation.detector import AttestationDetector, AttestationVerdict
from ..attestation.signals import SignalConfiguration, compute_signals
from ..checkers.scope import ActionRegister
from ..execution import Decision, Execution, GovernedSystem, Input, RiskBand, Verdict
from ..guards.input_guard import InputGuard
from ..guards.output_guard import OutputGuard
from ..guards.oversight import HumanOversight, OversightDecision
from ..guards.reasoning_guard import ReasoningGuard
from ..policy.bind import BoundPolicy
from ..policy.clause import ClauseId, FailureAction
from ..policy.guards import GuardEntry
from ..trust.ledger import AppendOnlyLedger, LedgerRecord
from ..trust.monitor import DriftSignal, PolicyMonitor
from ..trust.risk import RiskAssessment, RiskScorer


@dataclass(frozen=True)
class EnforcementResult:
    decision: Decision
    execution: Execution | None
    guard_entries: tuple[GuardEntry, ...]
    attestation: AttestationVerdict | None
    risk: RiskAssessment
    oversight: OversightDecision | None
    record: LedgerRecord | None
    drift: DriftSignal | None
    cause: str | None
    measures: Mapping[ClauseId, float]
    held_for_approval: bool

    @property
    def admitted(self) -> bool:
        return self.decision is Decision.ADMIT

    @property
    def released(self) -> bool:
        return self.admitted and not self.held_for_approval

    @property
    def referred(self) -> bool:
        return self.oversight is not None and self.oversight.referred


class GovernancePipeline:
    def __init__(
        self,
        system: GovernedSystem,
        bound_policy: BoundPolicy,
        input_guard: InputGuard,
        reasoning_guard: ReasoningGuard,
        output_guard: OutputGuard,
        detector: AttestationDetector,
        signal_configuration: SignalConfiguration,
        risk_scorer: RiskScorer,
        oversight: HumanOversight,
        action_register: ActionRegister,
        ledger: AppendOnlyLedger | None,
        monitor: PolicyMonitor | None,
        rng: np.random.Generator,
    ) -> None:
        self.system = system
        self.bound_policy = bound_policy
        self.input_guard = input_guard
        self.reasoning_guard = reasoning_guard
        self.output_guard = output_guard
        self.detector = detector
        self.signal_configuration = signal_configuration
        self.risk_scorer = risk_scorer
        self.oversight = oversight
        self.action_register = action_register
        self.ledger = ledger
        self.monitor = monitor
        self.rng = rng

    def assess_risk(
        self,
        measures: Mapping[ClauseId, float],
        not_attested: bool,
        clause_rejected: bool,
        action: str | None,
    ) -> RiskAssessment:
        completed = {clause: measures.get(clause, 1.0) for clause in ClauseId}
        high_risk_action = (
            self.action_register.is_high_risk(action)
            if action is not None and self.action_register.permits(action)
            else False
        )
        return self.risk_scorer.assess(
            measures=completed,
            not_attested=not_attested,
            clause_rejected=clause_rejected,
            action_is_high_risk=high_risk_action,
        )

    def __call__(self, request: Input) -> EnforcementResult:
        entries: list[GuardEntry] = []

        admission = self.input_guard(request)
        entries.append(admission.entry)
        if admission.verdict is Verdict.REJECT or admission.admitted_input is None:
            admission.attribute_store.discard()
            return self._withhold(
                entries=entries,
                execution=None,
                cause=str(admission.entry.detail.get("cause", "input_guard_reject")),
                clause=ClauseId.DATA_MINIMISATION,
                measures={ClauseId.DATA_MINIMISATION: 0.0},
                action=request.requested_action,
            )

        execution = self.system.execute(admission.admitted_input)
        if not self.action_register.permits(execution.action):
            admission.attribute_store.discard()
            return self._withhold(
                entries=entries,
                execution=execution,
                cause="realised_action_out_of_scope",
                clause=ClauseId.PRECONDITION_COMPLIANCE,
                measures={ClauseId.PRECONDITION_COMPLIANCE: 0.0},
                action=None,
            )

        reasoning = self.reasoning_guard(execution)
        entries.append(reasoning.entry)
        measures: dict[ClauseId, float] = {
            ClauseId.REASONING_FAITHFULNESS: reasoning.measures["mu_rf"],
            ClauseId.PRECONDITION_COMPLIANCE: reasoning.measures["mu_pc"],
            ClauseId.DATA_MINIMISATION: reasoning.measures["mu_dm_trace"],
        }
        if reasoning.verdict is Verdict.REJECT:
            admission.attribute_store.discard()
            return self._withhold(
                entries=entries,
                execution=execution,
                cause="reasoning_guard_reject",
                clause=ClauseId.REASONING_FAITHFULNESS,
                measures=measures,
                action=execution.action,
            )

        try:
            output = self.output_guard(execution, request, admission.attribute_store)
        finally:
            admission.attribute_store.discard()
        entries.append(output.entry)
        measures[ClauseId.NON_DISCRIMINATION] = output.measures["mu_nd"]
        measures[ClauseId.FACTUAL_GROUNDEDNESS] = output.measures["mu_fg"]
        measures[ClauseId.DATA_MINIMISATION] = min(
            reasoning.measures["mu_dm_trace"], output.measures["mu_dm_response"]
        )
        if output.verdict is Verdict.REJECT:
            return self._withhold(
                entries=entries,
                execution=execution,
                cause="output_guard_reject",
                clause=ClauseId.NON_DISCRIMINATION,
                measures=measures,
                action=execution.action,
            )

        signals = compute_signals(
            system=self.system,
            execution=execution,
            configuration=self.signal_configuration,
            rng=self.rng,
            faithfulness_value=reasoning.measures["mu_rf"],
        )
        attestation = self.detector.observe(signals)
        risk = self.assess_risk(
            measures=measures,
            not_attested=not attestation.attested,
            clause_rejected=False,
            action=execution.action,
        )

        if not attestation.attested:
            oversight = self.oversight.route(
                execution=execution,
                band=RiskBand.HIGH,
                evidence={
                    "attestation": attestation.statistic,
                    "threshold": attestation.threshold,
                },
            )
            return self._finalise(
                decision=Decision.WITHHOLD,
                execution=execution,
                entries=entries,
                attestation=attestation,
                risk=risk,
                oversight=oversight,
                cause="not_attested",
                measures=measures,
                held_for_approval=False,
            )

        oversight = self.oversight.route(
            execution=execution,
            band=risk.band,
            evidence={"risk_score": risk.score, "overrides": list(risk.overrides)},
        )
        if oversight.rejected:
            return self._finalise(
                decision=Decision.WITHHOLD,
                execution=execution,
                entries=entries,
                attestation=attestation,
                risk=risk,
                oversight=oversight,
                cause="oversight_rejected",
                measures=measures,
                held_for_approval=False,
            )

        return self._finalise(
            decision=Decision.ADMIT,
            execution=execution,
            entries=entries,
            attestation=attestation,
            risk=risk,
            oversight=oversight,
            cause=None,
            measures=measures,
            held_for_approval=oversight.held,
        )

    def _withhold(
        self,
        entries: list[GuardEntry],
        execution: Execution | None,
        cause: str,
        clause: ClauseId,
        measures: Mapping[ClauseId, float],
        action: str | None,
    ) -> EnforcementResult:
        risk = self.assess_risk(
            measures=measures, not_attested=False, clause_rejected=True, action=action
        )
        actions = self.bound_policy.record(clause).failure_action
        oversight: OversightDecision | None = None
        if execution is not None and FailureAction.REFER in actions:
            oversight = self.oversight.route(
                execution=execution,
                band=RiskBand.HIGH,
                evidence={"cause": cause, "clause": clause.value},
            )
        return self._finalise(
            decision=Decision.WITHHOLD,
            execution=execution,
            entries=entries,
            attestation=None,
            risk=risk,
            oversight=oversight,
            cause=cause,
            measures=measures,
            held_for_approval=False,
        )

    def _finalise(
        self,
        decision: Decision,
        execution: Execution | None,
        entries: list[GuardEntry],
        attestation: AttestationVerdict | None,
        risk: RiskAssessment,
        oversight: OversightDecision | None,
        cause: str | None,
        measures: Mapping[ClauseId, float],
        held_for_approval: bool,
    ) -> EnforcementResult:
        record: LedgerRecord | None = None
        if self.ledger is not None:
            record = self.ledger.append(
                self._record_body(
                    decision, execution, entries, attestation, risk, oversight, cause,
                    held_for_approval,
                )
            )
        drift: DriftSignal | None = None
        if self.monitor is not None and execution is not None:
            drift = self.monitor.observe(
                decision=decision,
                attested=attestation.attested if attestation is not None else False,
                embedding=float(np.mean(self.system.decision_scores(execution.inputs))),
            )
        return EnforcementResult(
            decision=decision,
            execution=execution,
            guard_entries=tuple(entries),
            attestation=attestation,
            risk=risk,
            oversight=oversight,
            record=record,
            drift=drift,
            cause=cause,
            measures=dict(measures),
            held_for_approval=held_for_approval,
        )

    def _record_body(
        self,
        decision: Decision,
        execution: Execution | None,
        entries: list[GuardEntry],
        attestation: AttestationVerdict | None,
        risk: RiskAssessment,
        oversight: OversightDecision | None,
        cause: str | None,
        held_for_approval: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "exec_id": execution.identifier if execution else Execution.new_identifier(),
            "ts": execution.timestamp if execution else None,
            "input": execution.input_digest() if execution else None,
            "guards": [entry.to_dict() for entry in entries],
            "verdict": decision.value,
            "held_for_approval": held_for_approval,
            "cause": cause,
            "system": self.system.name,
            "policy": self.bound_policy.policy.to_dict(),
            "risk": risk.to_dict(),
        }
        body["attestation"] = (
            {
                "D_t": attestation.statistic,
                "tau": attestation.threshold,
                "verdict": "attested" if attestation.attested else "not_attested",
                "window_size": attestation.window_size,
                "p_values": dict(attestation.p_values),
                "adjusted_p_values": dict(attestation.adjusted_p_values),
            }
            if attestation is not None
            else None
        )
        body["oversight"] = oversight.to_dict() if oversight is not None else None
        return body
