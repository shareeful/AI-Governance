from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from agp.errors import InterfaceError
from agp.execution import (
    Claim,
    EvidenceItem,
    Execution,
    GovernedSystem,
    Input,
    Reason,
    ReasoningTrace,
    Response,
)
from agp.policy.measures import (
    counterfactual_invariance,
    data_minimisation,
    factual_groundedness,
    precondition_compliance,
    reasoning_faithfulness,
)


class AttributeSensitiveSystem(GovernedSystem):
    name = "attribute_sensitive"
    modality = "tabular"

    def __init__(self, sensitive: bool, load_bearing: frozenset[str]) -> None:
        self.sensitive = sensitive
        self.load_bearing = load_bearing

    def execute(self, request: Input) -> Execution:
        raise NotImplementedError

    def action_for_input(self, request: Input) -> str:
        if self.sensitive and request.metadata.get("sex") == "F":
            return "decline"
        return "approve"

    def action_for_trace(self, request: Input, trace: ReasoningTrace) -> str:
        present = {reason.identifier for reason in trace.reasons}
        return "approve" if self.load_bearing <= present else "decline"

    def substitute_attribute(self, request: Input, attribute: str, value: Any) -> Input:
        from dataclasses import replace

        metadata = dict(request.metadata)
        metadata[attribute] = value
        return replace(request, metadata=metadata)

    def protected_attribute_domain(self, attribute: str) -> Sequence[Any]:
        return ("F", "M")

    def attribution(self, request: Input) -> Mapping[str, float]:
        return {"g": 1.0}

    def decision_scores(self, request: Input) -> np.ndarray:
        return np.array([0.5, 0.5])

    def perturbations(self, request, count, rng):
        return [request] * count

    def paraphrases(self, request, count, rng):
        return [request] * count

    def required_checks(self, action: str) -> frozenset[str]:
        return frozenset({"check_a", "check_b"})

    def minimum_necessary_personal_data(self) -> frozenset[str]:
        return frozenset()


class KeywordExtractor:
    def __init__(self, items: Mapping[str, frozenset[str]]) -> None:
        self.items = items

    def extract(self, text: str) -> frozenset[str]:
        found: set[str] = set()
        for token, labels in self.items.items():
            if token in text:
                found |= labels
        return frozenset(found)


class SubstringEntailment:
    def entails(self, premise: str, hypothesis: str) -> bool:
        return hypothesis in premise


def _request() -> Input:
    return Input(
        identifier="case-1",
        request="applicant record",
        features=np.zeros(3, dtype=np.float32),
        evidence=(),
        metadata={"sex": "M"},
        requested_action="decide",
    )


def _execution(reasons: tuple[Reason, ...], checks: tuple[str, ...], commit: int) -> Execution:
    trace = ReasoningTrace(
        reasons=reasons,
        recorded_checks=checks,
        commit_index=commit,
        text="\n".join(r.statement for r in reasons),
    )
    return Execution(
        identifier="exec-1",
        inputs=_request(),
        trace=trace,
        action="approve",
        response=Response(claims=(), text=""),
    )


def test_counterfactual_invariance_is_one_when_the_action_is_invariant():
    system = AttributeSensitiveSystem(sensitive=False, load_bearing=frozenset())
    outcome = counterfactual_invariance(system, _request(), "approve", ("sex",))
    assert outcome.value == 1.0


def test_counterfactual_invariance_detects_attribute_dependence():
    system = AttributeSensitiveSystem(sensitive=True, load_bearing=frozenset())
    outcome = counterfactual_invariance(system, _request(), "approve", ("sex",))
    assert outcome.value == 0.5


def test_counterfactual_invariance_requires_a_documented_attribute():
    system = AttributeSensitiveSystem(sensitive=False, load_bearing=frozenset())
    with pytest.raises(InterfaceError):
        counterfactual_invariance(system, _request(), "approve", ())


def test_faithfulness_is_one_when_every_reason_is_load_bearing():
    reasons = (
        Reason(identifier="r1", statement="first"),
        Reason(identifier="r2", statement="second"),
    )
    system = AttributeSensitiveSystem(sensitive=False, load_bearing=frozenset({"r1", "r2"}))
    outcome = reasoning_faithfulness(system, _execution(reasons, ("check_a",), 1))
    assert outcome.value == 1.0


def test_faithfulness_falls_when_a_reason_is_post_hoc():
    reasons = (
        Reason(identifier="r1", statement="first"),
        Reason(identifier="r2", statement="second"),
    )
    system = AttributeSensitiveSystem(sensitive=False, load_bearing=frozenset({"r1"}))
    outcome = reasoning_faithfulness(system, _execution(reasons, ("check_a",), 1))
    assert outcome.value == 0.5


def test_precondition_measure_counts_only_checks_before_the_commit_point():
    execution = _execution((Reason(identifier="r1", statement="first"),), ("check_a", "check_b"), 1)
    outcome = precondition_compliance(execution, frozenset({"check_a", "check_b"}))
    assert outcome.value == 0.5
    assert outcome.detail["missing_checks"] == ["check_b"]


def test_precondition_measure_is_one_when_every_check_precedes_the_commit():
    execution = _execution((Reason(identifier="r1", statement="first"),), ("check_a", "check_b"), 2)
    assert precondition_compliance(execution, frozenset({"check_a", "check_b"})).value == 1.0


def test_data_minimisation_certifies_only_the_minimum_necessary_set():
    from dataclasses import replace

    execution = _execution((Reason(identifier="r1", statement="name Jane"),), ("check_a",), 1)
    execution = replace(execution, response=Response(claims=(), text="approved"))
    extractor = KeywordExtractor({"Jane": frozenset({"NAME:Jane"})})
    permitted = data_minimisation(execution, extractor, frozenset({"NAME:Jane"}))
    excess = data_minimisation(execution, extractor, frozenset())
    assert permitted.value == 1.0
    assert excess.value == 0.0


def test_groundedness_is_the_proportion_of_entailed_claims():
    from dataclasses import replace

    evidence = (
        EvidenceItem(identifier="e1", content="alpha beta", provenance="dataset:x", modality="text"),
    )
    claims = (
        Claim(identifier="c1", statement="alpha", evidence_ids=("e1",)),
        Claim(identifier="c2", statement="gamma", evidence_ids=("e1",)),
    )
    execution = _execution((Reason(identifier="r1", statement="first"),), ("check_a",), 1)
    execution = replace(
        execution,
        inputs=replace(execution.inputs, evidence=evidence),
        response=Response(claims=claims, text="alpha gamma"),
    )
    assert factual_groundedness(execution, SubstringEntailment()).value == 0.5


def test_trace_removal_rejects_an_unknown_reason():
    execution = _execution((Reason(identifier="r1", statement="first"),), ("check_a",), 1)
    with pytest.raises(InterfaceError):
        execution.trace.without("missing")
