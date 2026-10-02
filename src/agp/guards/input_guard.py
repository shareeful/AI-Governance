from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from ..checkers.deid import DeIdentifier
from ..checkers.injection import InjectionJailbreakChecker
from ..checkers.provenance import ProvenanceChecker
from ..checkers.scope import ScopeChecker
from ..errors import InterfaceError
from ..execution import Input, Verdict, digest
from ..policy.clause import ClauseId
from ..policy.guards import GuardEntry


class SegregatedAttributeStore:
    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values: dict[str, Any] | None = dict(values)

    def read(self, attribute: str) -> Any:
        if self._values is None:
            raise InterfaceError(
                "the segregated protected-attribute store was discarded after the output guard "
                "returned its verdict; it may not be read again"
            )
        if attribute not in self._values:
            raise KeyError(
                f"protected attribute '{attribute}' is not held in the segregated store"
            )
        return self._values[attribute]

    def attributes(self) -> tuple[str, ...]:
        if self._values is None:
            return ()
        return tuple(self._values)

    def discard(self) -> None:
        if self._values is not None:
            for key in list(self._values):
                self._values[key] = None
            self._values = None

    @property
    def discarded(self) -> bool:
        return self._values is None


@dataclass(frozen=True)
class InputGuardOutcome:
    verdict: Verdict
    admitted_input: Input | None
    attribute_store: SegregatedAttributeStore
    entry: GuardEntry


class InputGuard:
    name = "input"

    def __init__(
        self,
        injection: InjectionJailbreakChecker,
        deidentifier: DeIdentifier,
        scope: ScopeChecker,
        provenance: ProvenanceChecker,
        protected_attributes: tuple[str, ...],
        minimum_necessary: frozenset[str],
    ) -> None:
        self.injection = injection
        self.deidentifier = deidentifier
        self.scope = scope
        self.provenance = provenance
        self.protected_attributes = protected_attributes
        self.minimum_necessary = minimum_necessary

    def __call__(self, request: Input) -> InputGuardOutcome:
        detail: dict[str, Any] = {}

        segregated = {
            attribute: request.metadata[attribute]
            for attribute in self.protected_attributes
            if attribute in request.metadata
        }
        store = SegregatedAttributeStore(values=segregated)
        detail["segregated_attributes"] = sorted(segregated)

        injection_verdict = self.injection.check(request.request)
        detail["injection_score"] = injection_verdict.score
        detail["injection_threshold"] = self.injection.threshold
        if injection_verdict.is_attack or injection_verdict.sanitised is None:
            return self._reject(request, store, detail, "injection_or_jailbreak")

        if not self.scope.in_scope(request.requested_action):
            detail["requested_action"] = request.requested_action
            return self._reject(request, store, detail, "action_out_of_scope")

        rejected_evidence = self.provenance.rejected(request.evidence)
        if rejected_evidence:
            detail["evidence_without_provenance"] = [i.identifier for i in rejected_evidence]
            return self._reject(request, store, detail, "evidence_provenance_absent")

        masked = self.deidentifier.mask(injection_verdict.sanitised, self.minimum_necessary)
        detail["personal_data_removed"] = sorted(masked.removed)
        if not masked.interpretable:
            return self._reject(request, store, detail, "masking_leaves_request_uninterpretable")

        admitted = replace(
            request,
            request=masked.text,
            metadata={
                key: value
                for key, value in request.metadata.items()
                if key not in self.protected_attributes
            },
        )
        entry = GuardEntry(
            guard=self.name,
            clauses=(ClauseId.DATA_MINIMISATION,),
            measures={"injection_score": injection_verdict.score},
            thresholds={"injection_threshold": self.injection.threshold},
            verdict=Verdict.CERTIFY,
            evidence_digest=digest(masked.text),
            detail=detail,
        )
        return InputGuardOutcome(
            verdict=Verdict.CERTIFY,
            admitted_input=admitted,
            attribute_store=store,
            entry=entry,
        )

    def _reject(
        self,
        request: Input,
        store: SegregatedAttributeStore,
        detail: dict[str, Any],
        cause: str,
    ) -> InputGuardOutcome:
        detail["cause"] = cause
        entry = GuardEntry(
            guard=self.name,
            clauses=(ClauseId.DATA_MINIMISATION,),
            measures={},
            thresholds={},
            verdict=Verdict.REJECT,
            evidence_digest=digest(request.request),
            detail=detail,
        )
        return InputGuardOutcome(
            verdict=Verdict.REJECT,
            admitted_input=None,
            attribute_store=store,
            entry=entry,
        )
