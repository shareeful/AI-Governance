from __future__ import annotations

import pytest

from agp.errors import ConfigurationError
from agp.execution import Verdict
from agp.policy.clause import (
    EXACT_CLAUSES,
    LEARNED_CLAUSES,
    ClauseId,
    ClausePolicy,
    ClauseRecord,
    DecisionKind,
    ExecutionComponent,
    FailureAction,
)
from agp.policy.compile import compile_clause
from agp.policy.guards import decide

EXACT_SPEC = {
    "regulatory_basis": "EU AI Act Art. 10",
    "threshold": 1.0,
    "checker": "counterfactual_invariance",
    "read_access": ["x", "a"],
    "failure_action": ["withhold", "log"],
}

LEARNED_SPEC = {
    "regulatory_basis": "EU AI Act Art. 13",
    "threshold": None,
    "checker": "causal_ablation",
    "read_access": ["r", "a"],
    "failure_action": ["withhold", "refer", "log"],
}


def test_every_clause_belongs_to_exactly_one_decision_family():
    assert EXACT_CLAUSES | LEARNED_CLAUSES == set(ClauseId)
    assert not EXACT_CLAUSES & LEARNED_CLAUSES


def test_exact_clause_compiles_with_a_unit_threshold():
    record = compile_clause(ClauseId.NON_DISCRIMINATION, EXACT_SPEC)
    assert record.decision_kind is DecisionKind.EXACT
    assert record.threshold == 1.0
    assert record.requires_reexecution


def test_learned_clause_refuses_a_configured_threshold():
    spec = dict(LEARNED_SPEC, threshold=0.7)
    with pytest.raises(ConfigurationError):
        compile_clause(ClauseId.REASONING_FAITHFULNESS, spec)


def test_exact_clause_refuses_a_threshold_below_one():
    spec = dict(EXACT_SPEC, threshold=0.9)
    with pytest.raises(ConfigurationError):
        compile_clause(ClauseId.NON_DISCRIMINATION, spec)


def test_guard_certifies_exactly_at_the_threshold():
    record = compile_clause(ClauseId.NON_DISCRIMINATION, EXACT_SPEC)
    assert decide(record, 1.0) is Verdict.CERTIFY
    assert decide(record, 0.999) is Verdict.REJECT


def test_learned_guard_without_calibration_is_refused():
    record = compile_clause(ClauseId.REASONING_FAITHFULNESS, LEARNED_SPEC)
    with pytest.raises(ValueError):
        decide(record, 0.9)


def test_policy_requires_all_five_clauses():
    partial = {
        ClauseId.NON_DISCRIMINATION: compile_clause(ClauseId.NON_DISCRIMINATION, EXACT_SPEC)
    }
    with pytest.raises(ConfigurationError):
        ClausePolicy(records=partial)


def test_recalibration_is_refused_for_exact_clauses():
    record = compile_clause(ClauseId.DATA_MINIMISATION, dict(EXACT_SPEC, read_access=["x", "r", "y"]))
    with pytest.raises(ConfigurationError):
        record.with_threshold(0.8)


def test_record_round_trips_through_its_dictionary_form():
    record = compile_clause(ClauseId.PRECONDITION_COMPLIANCE, dict(EXACT_SPEC, read_access=["r", "a"]))
    payload = record.to_dict()
    assert payload["clause"] == ClauseId.PRECONDITION_COMPLIANCE.value
    assert payload["read_access"] == [ExecutionComponent.TRACE.value, ExecutionComponent.ACTION.value]
    assert FailureAction.WITHHOLD.value in payload["failure_action"]


def test_clause_record_rejects_an_empty_failure_action():
    with pytest.raises(ConfigurationError):
        ClauseRecord(
            clause=ClauseId.NON_DISCRIMINATION,
            regulatory_basis="Art. 10",
            measure="mu_nd",
            decision_kind=DecisionKind.EXACT,
            threshold=1.0,
            checker="counterfactual_invariance",
            read_access=(ExecutionComponent.INPUT,),
            failure_action=(),
            requires_reexecution=True,
            parameters={},
        )
