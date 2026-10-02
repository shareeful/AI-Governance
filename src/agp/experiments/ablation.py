from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..attacks import build_attack
from ..attestation.signals import compute_signals
from ..benchmarks.base import BenchmarkSet
from ..benchmarks.embedding import ProbeCase
from ..certify import CertifiedDeployment, certify
from ..config import Config
from ..errors import ConfigurationError
from ..execution import RiskBand, Verdict
from ..guards.oversight import missed_referral_rate
from ..policy.clause import ClauseId
from ..seeding import derive_seed, seed_everything
from ..stats.bootstrap import mean_interval
from ..stats.metrics import clause_metrics, detection_at_false_alarm
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
from .rq1 import probes_for_clause

CONFIGURATIONS = (
    "full",
    "minus_input_guard",
    "minus_reasoning_guard",
    "minus_output_guard",
    "minus_attestation",
    "minus_human_oversight",
    "ungoverned",
)


def _blocks(
    deployment: CertifiedDeployment,
    configuration: str,
    probe: ProbeCase,
) -> bool:
    if configuration == "ungoverned":
        return False
    pipeline = deployment.pipeline
    if configuration != "minus_input_guard":
        admission = pipeline.input_guard(probe.execution.inputs)
        if admission.verdict is Verdict.REJECT:
            return True
    if configuration != "minus_reasoning_guard":
        if pipeline.reasoning_guard(probe.execution).verdict is Verdict.REJECT:
            return True
    if configuration != "minus_output_guard":
        from ..guards.input_guard import SegregatedAttributeStore

        store = SegregatedAttributeStore(values=dict(probe.execution.protected_attributes))
        if (
            pipeline.output_guard(probe.execution, probe.execution.inputs, store).verdict
            is Verdict.REJECT
        ):
            return True
    return False


def _integrity_detection(
    deployment: CertifiedDeployment,
    configuration: str,
    clean_requests: Sequence,
    config: Config,
    seed: int,
    rng: np.random.Generator,
) -> float:
    budget = config.require_probability("attestation.calibration.alpha")
    if configuration in {"minus_attestation", "ungoverned"}:
        return 0.0
    family = config.str_value("experiments.ablation.attack_family")
    strength = config.float_value(f"experiments.attack_reference_strength.{family}")
    attack = build_attack(family, config)
    if family == "A4":
        compromised = attack.compromise(
            bundle=deployment.bundle,
            config=config,
            seed=seed,
            strength=strength,
            fingerprint=deployment.fingerprint,
            signal_configuration=deployment.signal_configuration,
            probe_requests=clean_requests,
        )
    else:
        compromised = attack.compromise(deployment.bundle, config, seed, strength)

    clean_vectors = [
        compute_signals(
            deployment.bundle.system,
            deployment.bundle.system.execute(r),
            deployment.signal_configuration,
            rng,
        )
        for r in clean_requests
    ]
    compromised_vectors = [
        compute_signals(
            compromised.system,
            compromised.system.execute(r),
            deployment.signal_configuration,
            rng,
        )
        for r in clean_requests
    ]
    clean_scores = deployment.detector.score_stream(clean_vectors)
    compromised_scores = deployment.detector.score_stream(compromised_vectors)
    scores = np.concatenate([clean_scores, compromised_scores])
    labels = np.concatenate(
        [
            np.zeros(clean_scores.size, dtype=bool),
            np.ones(compromised_scores.size, dtype=bool),
        ]
    )
    return detection_at_false_alarm(scores, labels, budget)


def _missed_referral(
    deployment: CertifiedDeployment,
    configuration: str,
    requests: Sequence,
) -> float:
    referred: list[bool] = []
    true_high: list[bool] = []
    for request in requests:
        result = deployment.pipeline(request)
        if result.execution is None:
            continue
        true_high.append(result.risk.band is RiskBand.HIGH)
        if configuration in {"minus_human_oversight", "ungoverned"}:
            referred.append(False)
        else:
            referred.append(result.referred)
    if not any(true_high):
        raise ConfigurationError(
            "no execution in the ablation stream carries a high true risk band; the "
            "missed-referral rate is undefined"
        )
    return missed_referral_rate(referred, true_high)


def run(config: Config, benchmarks: Mapping[str, BenchmarkSet]) -> ExperimentResult:
    result = ExperimentResult(name="ablation")
    seeds = seeds_from(config)
    systems = systems_from(config)
    ratio = config.require_probability("experiments.violation_ratio")
    resamples, confidence = bootstrap_settings(config)
    oversight_sample = config.int_value("experiments.ablation.oversight_sample_size")
    integrity_sample = config.int_value("experiments.ablation.integrity_sample_size")

    collected: dict[tuple[str, str], list[float]] = {}

    for seed in seeds:
        seed_everything(seed)
        for system_name in systems:
            deployment = certify(config, system_name, seed, benchmarks, enable_ledger=False)
            rng = np.random.default_rng(derive_seed(seed, system_name, "ablation"))
            carriers = deployment.bundle.split("test")
            probes: list[ProbeCase] = []
            for clause in ClauseId:
                probes.extend(
                    probes_for_clause(deployment, clause, benchmarks, carriers, ratio, rng)
                )
            take = min(oversight_sample, len(carriers))
            oversight_requests = [
                carriers[int(i)] for i in rng.choice(len(carriers), take, replace=False)
            ]
            integrity_take = min(integrity_sample, len(carriers))
            require_stream_length(
                integrity_take, config, f"the integrity stream of system '{system_name}'"
            )
            integrity_requests = [
                carriers[int(i)]
                for i in rng.choice(len(carriers), integrity_take, replace=False)
            ]

            for configuration in CONFIGURATIONS:
                blocked = [_blocks(deployment, configuration, probe) for probe in probes]
                violates = [probe.violates for probe in probes]
                metrics = clause_metrics(blocked, violates)
                integrity = _integrity_detection(
                    deployment, configuration, integrity_requests, config, seed, rng
                )
                missed = _missed_referral(deployment, configuration, oversight_requests)

                for metric_name, value in (
                    ("detection_recall", metrics.recall),
                    ("false_block_rate", metrics.false_block_rate),
                    ("integrity_at_budget", integrity),
                    ("missed_referral_rate", missed),
                ):
                    collected.setdefault((configuration, metric_name), []).append(value)

                result.add(
                    seed=seed,
                    system=system_name,
                    modality=deployment.bundle.modality,
                    configuration=configuration,
                    detection_recall=metrics.recall,
                    false_block_rate=metrics.false_block_rate,
                    integrity_at_budget=integrity,
                    missed_referral_rate=missed,
                )

    rng = np.random.default_rng(derive_seed(seeds[0], "ablation", "bootstrap"))
    for (configuration, metric_name), values in collected.items():
        result.metadata.setdefault("intervals", {}).setdefault(configuration, {})[metric_name] = (
            mean_interval(values, resamples, confidence, rng).to_dict()
        )

    comparisons = {}
    for configuration in CONFIGURATIONS:
        if configuration == "full":
            continue
        for metric_name in (
            "detection_recall",
            "false_block_rate",
            "integrity_at_budget",
            "missed_referral_rate",
        ):
            full = collected.get(("full", metric_name))
            other = collected.get((configuration, metric_name))
            if full and other:
                comparisons[f"{configuration}:{metric_name}"] = (full, other)
    if comparisons:
        tested = compare_paired(comparisons, family_wise_alpha(config))
        result.comparisons = [c.to_dict() for c in tested.values()]

    result.write(artifact_root(config))
    return result
