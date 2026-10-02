from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
import yaml

from agp.attestation.detector import AttestationDetector
from agp.attestation.fingerprint import BehaviouralFingerprint
from agp.attestation.signals import SignalConfiguration, compute_signals
from agp.checkers.scope import ActionRegister, ScopeChecker
from agp.config import Config
from agp.execution import Decision, Input, Verdict
from agp.guards.input_guard import InputGuard
from agp.guards.output_guard import OutputGuard
from agp.guards.oversight import HumanOversight, QueueingReviewer, ReviewOutcome, Reviewer
from agp.guards.reasoning_guard import ReasoningGuard
from agp.policy.bind import bind_policy
from agp.policy.clause import ClauseId
from agp.policy.compile import compile_policy
from agp.runtime.pipeline import GovernancePipeline
from agp.systems.base import InstrumentedSystem, RationaleTemplates
from agp.trust.risk import RiskScorer

FEATURE_GROUPS = ("g0", "g1", "g2", "g3")
REQUIRED_CHECKS = ("quality_verification", "history_review")

BASE_CONFIG = {
    "policy": {
        "clauses": {
            "non_discrimination": {
                "regulatory_basis": "Art. 10",
                "threshold": 1.0,
                "checker": "counterfactual_invariance",
                "read_access": ["x", "a"],
                "failure_action": ["withhold", "log"],
            },
            "data_minimisation": {
                "regulatory_basis": "Art. 10(5)",
                "threshold": 1.0,
                "checker": "personal_data_absence",
                "read_access": ["x", "r", "y"],
                "failure_action": ["withhold", "log"],
            },
            "reasoning_faithfulness": {
                "regulatory_basis": "Art. 13",
                "threshold": None,
                "checker": "causal_ablation",
                "read_access": ["r", "a"],
                "failure_action": ["withhold", "refer", "log"],
            },
            "precondition_compliance": {
                "regulatory_basis": "Art. 12",
                "threshold": 1.0,
                "checker": "required_check_presence",
                "read_access": ["r", "a"],
                "failure_action": ["withhold", "refer", "log"],
            },
            "factual_groundedness": {
                "regulatory_basis": "Art. 15",
                "threshold": None,
                "checker": "natural_language_inference",
                "read_access": ["x", "y"],
                "failure_action": ["withhold", "log"],
            },
        }
    },
    "trust": {
        "risk": {
            "rho": 0.5,
            "weight_sum_tolerance": 1.0e-9,
            "weights": {
                "attestation": 0.30,
                "non_discrimination": 0.16,
                "data_minimisation": 0.14,
                "reasoning_faithfulness": 0.14,
                "precondition_compliance": 0.13,
                "factual_groundedness": 0.13,
            },
        }
    },
    "oversight": {
        "high_band_regime": "human_in_the_loop",
        "low_band_regime": "human_on_the_loop",
        "reviewer_identifier": "reviewer-1",
    },
}


class DeterministicSystem(InstrumentedSystem):
    def __init__(self, register: ActionRegister, attribute_sensitive: bool = False) -> None:
        self.attribute_sensitive = attribute_sensitive
        super().__init__(
            name="deterministic",
            modality="tabular",
            action_register=register,
            feature_groups=FEATURE_GROUPS,
            protected_attributes=("sex",),
            attribute_domains={"sex": ("F", "M")},
            minimum_necessary=frozenset(),
            templates=RationaleTemplates(
                reason="Group {group} carries mass {mass}.",
                claim="The {action} decision rests on group {group}.",
                evidence="Group {group} carries mass {mass}.",
                check="Recorded check: {check}.",
            ),
            reason_count=3,
            recorded_check_policy="complete",
        )

    def decision_scores(self, request: Input) -> np.ndarray:
        weights = np.asarray(request.features, dtype=float)
        if self.attribute_sensitive and request.metadata.get("sex") == "F":
            weights = weights * np.linspace(1.0, 0.1, weights.size)
        positive = float(np.clip(weights.sum() / (weights.size or 1), 0.01, 0.99))
        return np.array([1.0 - positive, positive], dtype=float)

    def predict_action(self, request: Input) -> str:
        return "approve" if self.decision_scores(request)[1] >= 0.5 else "decline"

    def predict_action_with_groups_removed(
        self, request: Input, removed_groups: frozenset[str]
    ) -> str:
        features = np.array(request.features, dtype=float, copy=True)
        for group in removed_groups:
            features[FEATURE_GROUPS.index(group)] = 0.0
        return self.predict_action(replace(request, features=features))

    def attribution(self, request: Input) -> Mapping[str, float]:
        features = np.abs(np.asarray(request.features, dtype=float))
        return {name: float(v) for name, v in zip(FEATURE_GROUPS, features)}

    def substitute_attribute(self, request: Input, attribute: str, value: Any) -> Input:
        metadata = dict(request.metadata)
        metadata[attribute] = value
        return replace(request, metadata=metadata)

    def perturbations(self, request: Input, count: int, rng: np.random.Generator) -> list[Input]:
        features = np.asarray(request.features, dtype=float)
        return [
            replace(request, features=(features + rng.normal(0.0, 0.01, features.shape)))
            for _ in range(count)
        ]

    def paraphrases(self, request: Input, count: int, rng: np.random.Generator) -> list[Input]:
        features = np.asarray(request.features, dtype=float)
        return [replace(request, features=features * (1.0 + 1e-6 * index)) for index in range(count)]

    def recorded_checks(self, request: Input, action: str) -> tuple[str, ...]:
        recorded = request.metadata.get("recorded_checks")
        if recorded is None:
            return tuple(sorted(self.action_register.required_checks(action)))
        return tuple(recorded)


class AlwaysCleanInjection:
    threshold = 0.5

    def check(self, text: str):
        from agp.checkers.injection import InjectionVerdict

        return InjectionVerdict(is_attack=False, score=0.0, sanitised=text)


class BlockingInjection:
    threshold = 0.5

    def check(self, text: str):
        from agp.checkers.injection import InjectionVerdict

        return InjectionVerdict(is_attack=True, score=0.99, sanitised=None)


class NoPersonalData:
    def extract(self, text: str) -> frozenset[str]:
        return frozenset()

    def mask(self, text: str, retain: frozenset[str]):
        from agp.checkers.deid import DeidentificationOutcome

        return DeidentificationOutcome(text=text, removed=frozenset(), interpretable=True)


class AlwaysEntails:
    threshold = 0.5

    def entails(self, premise: str, hypothesis: str) -> bool:
        return True


class NeverEntails:
    threshold = 0.5

    def entails(self, premise: str, hypothesis: str) -> bool:
        return False


class AlwaysSafe:
    threshold = 0.5

    def check(self, text: str):
        from agp.checkers.safety import SafetyVerdict

        return SafetyVerdict(unsafe=False, score=0.0)


class AcceptEverything:
    def admissible(self, item) -> bool:
        return True

    def rejected(self, evidence: Sequence[Any]) -> tuple[Any, ...]:
        return ()


class ApprovingReviewer(Reviewer):
    identifier = "reviewer-1"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def review(self, execution, band, evidence) -> ReviewOutcome:
        self.seen.append(execution.identifier)
        return ReviewOutcome.APPROVED


def _config(tmp_path, **overrides) -> Config:
    payload = dict(BASE_CONFIG)
    for key, value in overrides.items():
        payload[key] = value
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return Config.load(path)


def _register(high_risk: bool = False) -> ActionRegister:
    return ActionRegister.from_mapping(
        {
            "approve": {"high_risk": high_risk, "required_checks": list(REQUIRED_CHECKS)},
            "decline": {"high_risk": high_risk, "required_checks": list(REQUIRED_CHECKS)},
        }
    )


def _request(identifier: str = "case-1", sex: str = "M", features=None) -> Input:
    return Input(
        identifier=identifier,
        request=f"case={identifier}",
        features=np.array([0.65, 0.6, 0.6, 0.2] if features is None else features, dtype=float),
        evidence=(),
        metadata={"sex": sex, "provenance": "dataset:test"},
        requested_action="approve",
    )


def _build(
    tmp_path,
    system: DeterministicSystem,
    register: ActionRegister,
    entailment=None,
    injection=None,
    reviewer=None,
) -> GovernancePipeline:
    config = _config(tmp_path)
    policy = compile_policy(config).with_thresholds(
        {
            ClauseId.REASONING_FAITHFULNESS: 0.3,
            ClauseId.FACTUAL_GROUNDEDNESS: 0.9,
        }
    )
    bound = bind_policy(policy, system, ("sex",), register.entries)
    deidentifier = NoPersonalData()
    rng = np.random.default_rng(0)
    signal_configuration = SignalConfiguration(
        perturbation_count=4,
        blend_weight=0.5,
        paraphrase_count=4,
        feature_groups=FEATURE_GROUPS,
    )
    workload = []
    for index in range(40):
        request = _request(f"cal-{index}", features=[0.65, 0.6, 0.6, 0.2])
        workload.append(
            compute_signals(system, system.execute(request), signal_configuration, rng)
        )
    fingerprint = BehaviouralFingerprint.build(system.name, workload)
    detector = AttestationDetector(fingerprint, window=4, enabled_signals=("faithfulness",))
    detector.build_null_distribution(workload, draws=50, rng=rng)
    detector._threshold = 1e9

    return GovernancePipeline(
        system=system,
        bound_policy=bound,
        input_guard=InputGuard(
            injection=injection or AlwaysCleanInjection(),
            deidentifier=deidentifier,
            scope=ScopeChecker(register),
            provenance=AcceptEverything(),
            protected_attributes=("sex",),
            minimum_necessary=frozenset(),
        ),
        reasoning_guard=ReasoningGuard(system, bound, deidentifier),
        output_guard=OutputGuard(
            system, bound, entailment or AlwaysEntails(), AlwaysSafe(), deidentifier
        ),
        detector=detector,
        signal_configuration=signal_configuration,
        risk_scorer=RiskScorer(config),
        oversight=HumanOversight(
            config, reviewer or QueueingReviewer("reviewer-1", tmp_path / "queue.jsonl")
        ),
        action_register=register,
        ledger=None,
        monitor=None,
        rng=rng,
    )


def test_compliant_execution_is_admitted(tmp_path):
    register = _register()
    pipeline = _build(tmp_path, DeterministicSystem(register), register)
    result = pipeline(_request())
    assert result.decision is Decision.ADMIT
    assert result.cause is None
    assert set(result.measures) == set(ClauseId)


def test_input_guard_rejection_withholds_before_inference(tmp_path):
    register = _register()
    pipeline = _build(
        tmp_path, DeterministicSystem(register), register, injection=BlockingInjection()
    )
    result = pipeline(_request())
    assert result.decision is Decision.WITHHOLD
    assert result.execution is None
    assert result.cause == "injection_or_jailbreak"


def test_missing_precondition_withholds_and_refers(tmp_path):
    register = _register()
    system = DeterministicSystem(register)
    pipeline = _build(tmp_path, system, register)
    request = _request()
    request = replace(
        request, metadata={**request.metadata, "recorded_checks": (REQUIRED_CHECKS[0],)}
    )
    result = pipeline(request)
    assert result.decision is Decision.WITHHOLD
    assert result.cause == "reasoning_guard_reject"
    assert result.measures[ClauseId.PRECONDITION_COMPLIANCE] == pytest.approx(0.5)
    assert result.referred


def test_attribute_dependent_action_is_withheld_by_the_output_guard(tmp_path):
    register = _register()
    system = DeterministicSystem(register, attribute_sensitive=True)
    pipeline = _build(tmp_path, system, register)
    result = pipeline(_request())
    assert result.decision is Decision.WITHHOLD
    assert result.cause == "output_guard_reject"
    assert result.measures[ClauseId.NON_DISCRIMINATION] < 1.0


def test_ungrounded_response_is_withheld(tmp_path):
    register = _register()
    pipeline = _build(
        tmp_path, DeterministicSystem(register), register, entailment=NeverEntails()
    )
    result = pipeline(_request())
    assert result.decision is Decision.WITHHOLD
    assert result.cause == "output_guard_reject"
    assert result.measures[ClauseId.FACTUAL_GROUNDEDNESS] == pytest.approx(0.0)


def test_not_attested_execution_is_withheld_and_referred(tmp_path):
    register = _register()
    pipeline = _build(tmp_path, DeterministicSystem(register), register)
    pipeline.detector._threshold = -1.0
    for _ in range(pipeline.detector.window):
        result = pipeline(_request())
    assert result.decision is Decision.WITHHOLD
    assert result.cause == "not_attested"
    assert result.referred
    assert result.risk.band.value == "high"


def test_high_risk_action_is_held_for_approval_not_blocked(tmp_path):
    register = _register(high_risk=True)
    pipeline = _build(tmp_path, DeterministicSystem(register), register)
    result = pipeline(_request())
    assert result.decision is Decision.ADMIT
    assert result.held_for_approval
    assert not result.released
    assert result.referred


def test_reviewer_approval_releases_a_high_risk_action(tmp_path):
    register = _register(high_risk=True)
    reviewer = ApprovingReviewer()
    pipeline = _build(tmp_path, DeterministicSystem(register), register, reviewer=reviewer)
    result = pipeline(_request())
    assert result.decision is Decision.ADMIT
    assert result.released
    assert reviewer.seen == [result.execution.identifier]


def test_out_of_scope_action_is_withheld(tmp_path):
    register = _register()
    system = DeterministicSystem(register)
    pipeline = _build(tmp_path, system, register)
    pipeline.action_register = ActionRegister.from_mapping(
        {"decline": {"high_risk": False, "required_checks": list(REQUIRED_CHECKS)}}
    )
    pipeline.input_guard.scope = ScopeChecker(pipeline.action_register)
    result = pipeline(_request(features=[0.9, 0.9, 0.9, 0.9]))
    assert result.decision is Decision.WITHHOLD
    assert result.cause in {"action_out_of_scope", "realised_action_out_of_scope"}


def test_guard_entries_are_recorded_in_order(tmp_path):
    register = _register()
    pipeline = _build(tmp_path, DeterministicSystem(register), register)
    result = pipeline(_request())
    assert [entry.guard for entry in result.guard_entries] == ["input", "reasoning", "output"]
    assert all(entry.verdict is Verdict.CERTIFY for entry in result.guard_entries)


def test_protected_attribute_store_is_discarded_after_the_output_guard(tmp_path):
    register = _register()
    pipeline = _build(tmp_path, DeterministicSystem(register), register)
    captured = {}
    original = pipeline.input_guard.__call__

    def capturing(request):
        outcome = original(request)
        captured["store"] = outcome.attribute_store
        return outcome

    pipeline.input_guard = capturing
    result = pipeline(_request())
    assert result.decision is Decision.ADMIT
    assert captured["store"].discarded
    assert captured["store"].attributes() == ()
    with pytest.raises(Exception):
        captured["store"].read("sex")


def test_attribute_store_is_discarded_even_when_a_guard_rejects(tmp_path):
    register = _register()
    pipeline = _build(
        tmp_path, DeterministicSystem(register), register, entailment=NeverEntails()
    )
    captured = {}
    original = pipeline.input_guard.__call__

    def capturing(request):
        outcome = original(request)
        captured["store"] = outcome.attribute_store
        return outcome

    pipeline.input_guard = capturing
    result = pipeline(_request())
    assert result.decision is Decision.WITHHOLD
    assert captured["store"].discarded
