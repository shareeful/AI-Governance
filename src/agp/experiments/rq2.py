from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..attestation.signals import SignalVector, compute_signals
from ..attacks import build_attack
from ..attacks.base import CompromisedModel
from ..baselines.integrity import build_integrity_baseline
from ..benchmarks.base import BenchmarkSet
from ..certify import CertifiedDeployment, certify
from ..config import Config
from ..errors import ConfigurationError
from ..execution import Input, Verdict
from ..policy.measures import factual_groundedness
from ..seeding import derive_seed, seed_everything
from ..stats.bootstrap import mean_interval
from ..stats.metrics import detection_at_false_alarm, detection_auc, roc_curve
from ..stats.tests import compare_paired
from .base import (
    ExperimentResult,
    artifact_root,
    bootstrap_settings,
    family_wise_alpha,
    require_stream_length,
    seeds_from,
    systems_from,
)


def _signal_stream(
    system,
    requests: Sequence[Input],
    deployment: CertifiedDeployment,
    rng: np.random.Generator,
) -> list[SignalVector]:
    vectors: list[SignalVector] = []
    for request in requests:
        execution = system.execute(request)
        vectors.append(
            compute_signals(system, execution, deployment.signal_configuration, rng)
        )
    return vectors


def _reference_based_scores(
    system,
    requests: Sequence[Input],
    deployment: CertifiedDeployment,
) -> np.ndarray:
    scores: list[float] = []
    for request in requests:
        execution = system.execute(request)
        groundedness = factual_groundedness(execution, deployment.checkers.entailment)
        reasoning = deployment.pipeline.reasoning_guard(execution)
        rejected = float(reasoning.verdict is Verdict.REJECT)
        scores.append(rejected + (1.0 - groundedness.value))
    return np.asarray(scores, dtype=float)


def _attestation_scores(
    detector,
    vectors: Sequence[SignalVector],
) -> np.ndarray:
    return detector.score_stream(vectors)


def run(config: Config, benchmarks: Mapping[str, BenchmarkSet]) -> ExperimentResult:
    result = ExperimentResult(name="rq2_integrity_detection")
    seeds = seeds_from(config)
    systems = systems_from(config)
    families = [str(f) for f in config.list_value("experiments.attack_families")]
    baseline_names = [str(n) for n in config.list_value("experiments.integrity_baselines")]
    strengths = {
        family: float(config.float_value(f"experiments.attack_reference_strength.{family}"))
        for family in families
    }
    resamples, confidence = bootstrap_settings(config)
    budget = config.require_probability("attestation.calibration.alpha")

    per_detector: dict[tuple[str, str], list[float]] = {}

    for seed in seeds:
        seed_everything(seed)
        for system_name in systems:
            deployment = certify(config, system_name, seed, benchmarks, enable_ledger=False)
            clean_requests = deployment.bundle.split("test")
            require_stream_length(
                len(clean_requests), config, f"the test split of system '{system_name}'"
            )
            rng = np.random.default_rng(derive_seed(seed, system_name, "rq2"))

            clean_vectors = _signal_stream(
                deployment.bundle.system, clean_requests, deployment, rng
            )
            clean_attestation = _attestation_scores(deployment.detector, clean_vectors)
            dropped_signal = config.str_value("experiments.leave_one_signal_out")
            reduced_detector = deployment.detector.without_signal(dropped_signal)
            reduced_detector.calibrate(
                deployment.certification_vectors,
                config,
                np.random.default_rng(derive_seed(seed, system_name, "rq2", "reduced")),
            )
            clean_reduced = _attestation_scores(reduced_detector, clean_vectors)
            clean_reference = _reference_based_scores(
                deployment.bundle.system, clean_requests, deployment
            )

            fitted_baselines = {}
            for name in baseline_names:
                detector = build_integrity_baseline(name, config)
                detector.fit(
                    deployment.bundle.system,
                    deployment.bundle.split("calibration"),
                    np.random.default_rng(derive_seed(seed, system_name, "rq2", name)),
                )
                fitted_baselines[name] = detector

            clean_baseline_scores = {
                name: detector.score(
                    deployment.bundle.system,
                    clean_requests,
                    np.random.default_rng(derive_seed(seed, system_name, "rq2", name, "clean")),
                )
                for name, detector in fitted_baselines.items()
            }

            for family in families:
                compromised = _compromise(
                    config, family, deployment, seed, strengths[family], clean_requests
                )
                compromised_vectors = _signal_stream(
                    compromised.system, clean_requests, deployment, rng
                )

                entries: dict[str, tuple[np.ndarray, np.ndarray]] = {
                    "proposed_attestation": (
                        clean_attestation,
                        _attestation_scores(deployment.detector, compromised_vectors),
                    ),
                    f"proposed_without_{dropped_signal}": (
                        clean_reduced,
                        _attestation_scores(reduced_detector, compromised_vectors),
                    ),
                    "reference_based_guards": (
                        clean_reference,
                        _reference_based_scores(compromised.system, clean_requests, deployment),
                    ),
                }
                for name, detector in fitted_baselines.items():
                    entries[name] = (
                        clean_baseline_scores[name],
                        detector.score(
                            compromised.system,
                            clean_requests,
                            np.random.default_rng(
                                derive_seed(seed, system_name, "rq2", name, family)
                            ),
                        ),
                    )

                for name, (clean_scores, compromised_scores) in entries.items():
                    scores = np.concatenate([clean_scores, compromised_scores])
                    labels = np.concatenate(
                        [
                            np.zeros(clean_scores.size, dtype=bool),
                            np.ones(compromised_scores.size, dtype=bool),
                        ]
                    )
                    auc = detection_auc(scores, labels)
                    detection = detection_at_false_alarm(scores, labels, budget)
                    per_detector.setdefault((name, family), []).append(auc)
                    result.add(
                        seed=seed,
                        system=system_name,
                        modality=deployment.bundle.modality,
                        detector=name,
                        attack_family=family,
                        attack_strength=strengths[family],
                        auc=auc,
                        detection_at_budget=detection,
                        access=_access_for(name, fitted_baselines),
                        runtime=_runtime_for(name, fitted_baselines),
                    )
                    if name in {"proposed_attestation", "reference_based_guards"}:
                        false_positive, true_positive = roc_curve(scores, labels)
                        result.metadata.setdefault("roc", {}).setdefault(family, {})[
                            f"{name}:{system_name}:{seed}"
                        ] = {
                            "false_positive_rate": false_positive.tolist(),
                            "true_positive_rate": true_positive.tolist(),
                        }

    rng = np.random.default_rng(derive_seed(seeds[0], "rq2", "bootstrap"))
    for (name, family), values in per_detector.items():
        result.metadata.setdefault("intervals", {}).setdefault(family, {})[name] = (
            mean_interval(values, resamples, confidence, rng).to_dict()
        )

    comparisons = {}
    for family in families:
        proposed = per_detector.get(("proposed_attestation", family))
        if proposed is None:
            continue
        reduced_name = f"proposed_without_{config.str_value('experiments.leave_one_signal_out')}"
        for name in [*baseline_names, reduced_name, "reference_based_guards"]:
            other = per_detector.get((name, family))
            if other is None:
                continue
            comparisons[f"{family}:proposed_vs_{name}"] = (proposed, other)
    if comparisons:
        tested = compare_paired(comparisons, family_wise_alpha(config))
        result.comparisons = [c.to_dict() for c in tested.values()]

    result.write(artifact_root(config))
    return result


def _compromise(
    config: Config,
    family: str,
    deployment: CertifiedDeployment,
    seed: int,
    strength: float,
    probe_requests: Sequence[Input],
) -> CompromisedModel:
    attack = build_attack(family, config)
    if family == "A4":
        return attack.compromise(
            bundle=deployment.bundle,
            config=config,
            seed=seed,
            strength=strength,
            fingerprint=deployment.fingerprint,
            signal_configuration=deployment.signal_configuration,
            probe_requests=probe_requests,
        )
    return attack.compromise(deployment.bundle, config, seed, strength)


def _access_for(name: str, baselines: Mapping[str, object]) -> str:
    detector = baselines.get(name)
    if detector is None:
        return "BB"
    return ", ".join(detector.profile.access)


def _runtime_for(name: str, baselines: Mapping[str, object]) -> str:
    detector = baselines.get(name)
    if detector is None:
        return "per-execution"
    return detector.profile.runtime


def run_sensitivity(config: Config, benchmarks: Mapping[str, BenchmarkSet]) -> ExperimentResult:
    result = ExperimentResult(name="rq2_attack_strength_sensitivity")
    seeds = seeds_from(config)
    systems = systems_from(config)
    budget = config.require_probability("attestation.calibration.alpha")
    resamples, confidence = bootstrap_settings(config)

    sweeps = {
        family: [float(v) for v in config.list_value(f"experiments.attack_sweeps.{family}")]
        for family in config.keys("experiments.attack_sweeps")
    }
    if not sweeps:
        raise ConfigurationError("experiments.attack_sweeps must declare at least one family")

    collected: dict[tuple[str, float, str], list[float]] = {}
    for seed in seeds:
        seed_everything(seed)
        for system_name in systems:
            deployment = certify(config, system_name, seed, benchmarks, enable_ledger=False)
            clean_requests = deployment.bundle.split("test")
            require_stream_length(
                len(clean_requests), config, f"the test split of system '{system_name}'"
            )
            rng = np.random.default_rng(derive_seed(seed, system_name, "rq2", "sweep"))
            clean_vectors = _signal_stream(
                deployment.bundle.system, clean_requests, deployment, rng
            )
            clean_attestation = _attestation_scores(deployment.detector, clean_vectors)
            clean_reference = _reference_based_scores(
                deployment.bundle.system, clean_requests, deployment
            )
            for family, points in sweeps.items():
                for strength in points:
                    compromised = _compromise(
                        config, family, deployment, seed, strength, clean_requests
                    )
                    vectors = _signal_stream(compromised.system, clean_requests, deployment, rng)
                    pairs = (
                        (
                            "proposed_attestation",
                            clean_attestation,
                            _attestation_scores(deployment.detector, vectors),
                        ),
                        (
                            "reference_based_guards",
                            clean_reference,
                            _reference_based_scores(
                                compromised.system, clean_requests, deployment
                            ),
                        ),
                    )
                    for name, clean_scores, compromised_scores in pairs:
                        scores = np.concatenate([clean_scores, compromised_scores])
                        labels = np.concatenate(
                            [
                                np.zeros(clean_scores.size, dtype=bool),
                                np.ones(compromised_scores.size, dtype=bool),
                            ]
                        )
                        detection = detection_at_false_alarm(scores, labels, budget)
                        collected.setdefault((family, strength, name), []).append(detection)
                        result.add(
                            seed=seed,
                            system=system_name,
                            attack_family=family,
                            attack_strength=strength,
                            detector=name,
                            detection_at_budget=detection,
                        )

    rng = np.random.default_rng(derive_seed(seeds[0], "rq2", "sweep_bootstrap"))
    for (family, strength, name), values in collected.items():
        result.metadata.setdefault("intervals", {}).setdefault(family, {}).setdefault(
            f"{strength}", {}
        )[name] = mean_interval(values, resamples, confidence, rng).to_dict()

    result.write(artifact_root(config))
    return result
