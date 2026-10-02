from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .attestation.detector import AttestationDetector
from .attestation.fingerprint import BehaviouralFingerprint, sign_fingerprint
from .attestation.signals import SignalConfiguration, SignalVector, compute_signals
from .benchmarks.base import BenchmarkSet
from .benchmarks.loaders import load_all
from .calibration.ltt import CalibrationResult, CalibrationTargets, candidate_thresholds, learn_then_test
from .checkers.deid import DeIdentifier
from .checkers.injection import InjectionJailbreakChecker
from .checkers.nli import EntailmentChecker
from .checkers.provenance import ProvenanceChecker
from .checkers.safety import SafetyChecker
from .checkers.scope import ScopeChecker
from .config import Config
from .errors import CalibrationError, ConfigurationError
from .execution import Input
from .guards.input_guard import InputGuard
from .guards.output_guard import OutputGuard
from .guards.oversight import HumanOversight, build_reviewer
from .guards.reasoning_guard import ReasoningGuard
from .policy.bind import BoundPolicy, bind_policy
from .policy.clause import ClauseId, ClausePolicy
from .policy.compile import compile_policy
from .policy.measures import factual_groundedness, reasoning_faithfulness
from .runtime.pipeline import GovernancePipeline
from .seeding import derive_seed
from .systems.registry import SystemBundle, build_system
from .trust.ledger import AppendOnlyLedger
from .trust.monitor import PolicyMonitor
from .trust.risk import RiskScorer


@dataclass
class CheckerSuite:
    injection: InjectionJailbreakChecker
    deidentifier: DeIdentifier
    entailment: EntailmentChecker
    safety: SafetyChecker
    provenance: ProvenanceChecker


@dataclass
class CertifiedDeployment:
    bundle: SystemBundle
    bound_policy: BoundPolicy
    checkers: CheckerSuite
    detector: AttestationDetector
    fingerprint: BehaviouralFingerprint
    signal_configuration: SignalConfiguration
    pipeline: GovernancePipeline
    clause_calibration: Mapping[ClauseId, CalibrationResult]
    checker_calibration: Mapping[str, CalibrationResult]
    detector_calibration: CalibrationResult
    certification_vectors: tuple[SignalVector, ...]
    seed: int

    def calibration_report(self) -> dict[str, object]:
        return {
            "system": self.bundle.name,
            "seed": self.seed,
            "clauses": {
                clause.value: result.to_dict()
                for clause, result in self.clause_calibration.items()
            },
            "checkers": {
                name: result.to_dict() for name, result in self.checker_calibration.items()
            },
            "attestation": self.detector_calibration.to_dict(),
            "fingerprint": {
                "workload_size": self.fingerprint.workload_size,
                "summaries": {
                    name: summary.to_dict()
                    for name, summary in self.fingerprint.summaries.items()
                },
            },
        }


def build_checkers(config: Config, benchmarks: Mapping[str, BenchmarkSet]) -> tuple[
    CheckerSuite, dict[str, CalibrationResult]
]:
    injection = InjectionJailbreakChecker(config)
    deidentifier = DeIdentifier(config)
    entailment = EntailmentChecker(config)
    safety = SafetyChecker(config)
    provenance = ProvenanceChecker(
        frozenset(str(a) for a in config.list_value("checkers.provenance.accepted_authorities"))
    )

    injection_set = benchmarks["injection"]
    injection_result = injection.calibrate(
        texts=[str(case.payload["prompt"]) for case in injection_set.cases],
        is_attack=[case.violates for case in injection_set.cases],
        config=config,
    )

    groundedness_set = benchmarks["factual_groundedness"]
    entailment_result = entailment.calibrate(
        premises=[str(case.payload["evidence"]) for case in groundedness_set.cases],
        hypotheses=[str(case.payload["claim"]) for case in groundedness_set.cases],
        is_unsupported=[case.violates for case in groundedness_set.cases],
        config=config,
    )

    safety_set = benchmarks["safety"]
    safety_result = safety.calibrate(
        texts=[str(case.payload["text"]) for case in safety_set.cases],
        is_unsafe=[case.violates for case in safety_set.cases],
        config=config,
    )

    return (
        CheckerSuite(
            injection=injection,
            deidentifier=deidentifier,
            entailment=entailment,
            safety=safety,
            provenance=provenance,
        ),
        {
            "injection": injection_result,
            "entailment": entailment_result,
            "safety": safety_result,
        },
    )


def calibrate_clause_thresholds(
    config: Config,
    bundle: SystemBundle,
    checkers: CheckerSuite,
    calibration_requests: Sequence[Input],
    benchmarks: Mapping[str, BenchmarkSet],
    rng: np.random.Generator,
) -> dict[ClauseId, CalibrationResult]:
    from .benchmarks.embedding import embed_faithfulness, embed_groundedness

    targets = CalibrationTargets.from_config(config, "calibration")
    thresholds = candidate_thresholds(config, "calibration")
    ratio = config.require_probability("experiments.violation_ratio")

    faithfulness_probes = embed_faithfulness(
        bundle, benchmarks["reasoning_faithfulness"], calibration_requests, ratio, rng
    )
    faithfulness_result = learn_then_test(
        measures=[
            reasoning_faithfulness(bundle.system, probe.execution).value
            for probe in faithfulness_probes
        ],
        violates=[probe.violates for probe in faithfulness_probes],
        thresholds=thresholds.tolist(),
        targets=targets,
    )

    groundedness_probes = embed_groundedness(
        bundle, benchmarks["factual_groundedness"], calibration_requests, ratio, rng
    )
    groundedness_result = learn_then_test(
        measures=[
            factual_groundedness(probe.execution, checkers.entailment).value
            for probe in groundedness_probes
        ],
        violates=[probe.violates for probe in groundedness_probes],
        thresholds=thresholds.tolist(),
        targets=targets,
    )

    return {
        ClauseId.REASONING_FAITHFULNESS: faithfulness_result,
        ClauseId.FACTUAL_GROUNDEDNESS: groundedness_result,
    }


def build_fingerprint(
    config: Config,
    bundle: SystemBundle,
    calibration_requests: Sequence[Input],
    rng: np.random.Generator,
) -> tuple[BehaviouralFingerprint, SignalConfiguration, tuple[SignalVector, ...]]:
    signal_configuration = SignalConfiguration.from_config(
        config, bundle.system.feature_groups
    )
    workload_size = config.int_value("attestation.workload_size")
    if workload_size > len(calibration_requests):
        raise ConfigurationError(
            f"attestation.workload_size ({workload_size}) exceeds the calibration split of "
            f"system '{bundle.name}' ({len(calibration_requests)} executions)"
        )
    indices = rng.choice(len(calibration_requests), workload_size, replace=False)
    vectors: list[SignalVector] = []
    for index in indices:
        request = calibration_requests[int(index)]
        execution = bundle.system.execute(request)
        vectors.append(compute_signals(bundle.system, execution, signal_configuration, rng))
    fingerprint = BehaviouralFingerprint.build(bundle.name, vectors)
    return fingerprint, signal_configuration, tuple(vectors)


def certify(
    config: Config,
    system_name: str,
    seed: int,
    benchmarks: Mapping[str, BenchmarkSet] | None = None,
    enable_ledger: bool = True,
    enable_monitor: bool = True,
) -> CertifiedDeployment:
    rng = np.random.default_rng(derive_seed(seed, system_name, "certify"))
    bundle = build_system(config, system_name, seed)
    resolved_benchmarks = benchmarks if benchmarks is not None else load_all(config)

    checkers, checker_calibration = build_checkers(config, resolved_benchmarks)

    policy: ClausePolicy = compile_policy(config)
    calibration_requests = bundle.split("calibration")
    clause_calibration = calibrate_clause_thresholds(
        config, bundle, checkers, calibration_requests, resolved_benchmarks, rng
    )
    policy = policy.with_thresholds(
        {clause: result.threshold for clause, result in clause_calibration.items()}
    )

    bound_policy = bind_policy(
        policy=policy,
        system=bundle.system,
        protected_attributes=bundle.protected_attributes,
        action_register=bundle.action_register.entries,
    )

    fingerprint, signal_configuration, vectors = build_fingerprint(
        config, bundle, calibration_requests, rng
    )
    detector = AttestationDetector(
        fingerprint=fingerprint,
        window=config.int_value("attestation.window"),
        enabled_signals=[str(s) for s in config.list_value("attestation.signals")],
    )
    detector_calibration = detector.calibrate(vectors, config, rng)

    artifact_root = Path(config.str_value("paths.artifact_root")).expanduser()
    fingerprint_path = artifact_root / "fingerprints" / system_name / f"seed_{seed}.json"
    sign_fingerprint(
        fingerprint,
        Path(config.str_value("trust.ledger.private_key")).expanduser(),
        fingerprint_path,
    )

    input_guard = InputGuard(
        injection=checkers.injection,
        deidentifier=checkers.deidentifier,
        scope=ScopeChecker(bundle.action_register),
        provenance=checkers.provenance,
        protected_attributes=bundle.protected_attributes,
        minimum_necessary=bundle.system.minimum_necessary_personal_data(),
    )
    reasoning_guard = ReasoningGuard(
        system=bundle.system,
        bound_policy=bound_policy,
        deidentifier=checkers.deidentifier,
    )
    output_guard = OutputGuard(
        system=bundle.system,
        bound_policy=bound_policy,
        entailment=checkers.entailment,
        safety=checkers.safety,
        deidentifier=checkers.deidentifier,
    )

    ledger = None
    if enable_ledger:
        ledger = AppendOnlyLedger(
            path=artifact_root / "ledger" / system_name / f"seed_{seed}.jsonl",
            private_key_path=Path(config.str_value("trust.ledger.private_key")).expanduser(),
            public_key_path=Path(config.str_value("trust.ledger.public_key")).expanduser(),
        )

    monitor = None
    if enable_monitor:
        embedding = np.array(
            [float(np.mean(bundle.system.decision_scores(r))) for r in calibration_requests],
            dtype=float,
        )
        monitor = PolicyMonitor(config, embedding)

    oversight = HumanOversight(
        config,
        build_reviewer(
            config,
            artifact_root / "oversight" / system_name / f"seed_{seed}.jsonl",
        ),
    )

    pipeline = GovernancePipeline(
        system=bundle.system,
        bound_policy=bound_policy,
        input_guard=input_guard,
        reasoning_guard=reasoning_guard,
        output_guard=output_guard,
        detector=detector,
        signal_configuration=signal_configuration,
        risk_scorer=RiskScorer(config),
        oversight=oversight,
        action_register=bundle.action_register,
        ledger=ledger,
        monitor=monitor,
        rng=rng,
    )

    deployment = CertifiedDeployment(
        bundle=bundle,
        bound_policy=bound_policy,
        checkers=checkers,
        detector=detector,
        fingerprint=fingerprint,
        signal_configuration=signal_configuration,
        pipeline=pipeline,
        clause_calibration=clause_calibration,
        checker_calibration=checker_calibration,
        detector_calibration=detector_calibration,
        certification_vectors=vectors,
        seed=seed,
    )

    report_path = artifact_root / "certification" / system_name / f"seed_{seed}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(deployment.calibration_report(), handle, indent=2, sort_keys=True)
    return deployment


def recertify(
    config: Config,
    deployment: CertifiedDeployment,
    approved_requests: Sequence[Input],
    benchmarks: Mapping[str, BenchmarkSet],
) -> CertifiedDeployment:
    if not approved_requests:
        raise CalibrationError(
            "re-certification requires approved runtime evidence; the screening step returned "
            "no usable record"
        )
    rng = np.random.default_rng(
        derive_seed(deployment.seed, deployment.bundle.name, "recertify")
    )
    clause_calibration = calibrate_clause_thresholds(
        config, deployment.bundle, deployment.checkers, approved_requests, benchmarks, rng
    )
    policy = deployment.bound_policy.policy.with_thresholds(
        {clause: result.threshold for clause, result in clause_calibration.items()}
    )
    bound_policy = bind_policy(
        policy=policy,
        system=deployment.bundle.system,
        protected_attributes=deployment.bundle.protected_attributes,
        action_register=deployment.bundle.action_register.entries,
    )
    fingerprint, signal_configuration, vectors = build_fingerprint(
        config, deployment.bundle, approved_requests, rng
    )
    detector = AttestationDetector(
        fingerprint=fingerprint,
        window=config.int_value("attestation.window"),
        enabled_signals=[str(s) for s in config.list_value("attestation.signals")],
    )
    detector_calibration = detector.calibrate(vectors, config, rng)

    deployment.pipeline.bound_policy = bound_policy
    deployment.pipeline.reasoning_guard.bound_policy = bound_policy
    deployment.pipeline.output_guard.bound_policy = bound_policy
    deployment.pipeline.detector = detector
    deployment.pipeline.signal_configuration = signal_configuration

    return CertifiedDeployment(
        bundle=deployment.bundle,
        bound_policy=bound_policy,
        checkers=deployment.checkers,
        detector=detector,
        fingerprint=fingerprint,
        signal_configuration=signal_configuration,
        pipeline=deployment.pipeline,
        clause_calibration=clause_calibration,
        checker_calibration=deployment.checker_calibration,
        detector_calibration=detector_calibration,
        certification_vectors=vectors,
        seed=deployment.seed,
    )
