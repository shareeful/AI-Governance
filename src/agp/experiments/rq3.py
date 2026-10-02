from __future__ import annotations

from typing import Mapping

import numpy as np

from ..benchmarks.base import BenchmarkSet
from ..benchmarks.embedding import embed_faithfulness, embed_groundedness
from ..calibration.ltt import CalibrationTargets, candidate_thresholds, learn_then_test
from ..certify import certify
from ..config import Config
from ..policy.clause import ClauseId
from ..policy.measures import factual_groundedness, reasoning_faithfulness
from ..seeding import derive_seed, seed_everything
from ..stats.bootstrap import mean_interval
from ..stats.metrics import certified_violation_rate
from .base import (
    ExperimentResult,
    artifact_root,
    bootstrap_settings,
    seeds_from,
    systems_from,
)


def run(config: Config, benchmarks: Mapping[str, BenchmarkSet]) -> ExperimentResult:
    result = ExperimentResult(name="rq3_threshold_validity")
    seeds = seeds_from(config)
    systems = systems_from(config)
    ratio = config.require_probability("experiments.violation_ratio")
    alphas = [float(a) for a in config.list_value("experiments.alpha_sweep")]
    delta = config.require_probability("calibration.delta")
    bound = config.str_value("calibration.bound")
    correction = config.str_value("calibration.correction")
    resamples, confidence = bootstrap_settings(config)

    collected: dict[tuple[str, float], list[float]] = {}

    for seed in seeds:
        seed_everything(seed)
        for system_name in systems:
            deployment = certify(config, system_name, seed, benchmarks, enable_ledger=False)
            rng = np.random.default_rng(derive_seed(seed, system_name, "rq3"))
            calibration_requests = deployment.bundle.split("calibration")
            test_requests = deployment.bundle.split("test")
            thresholds = candidate_thresholds(config, "calibration")

            for clause, embedder, measure_fn in (
                (
                    ClauseId.REASONING_FAITHFULNESS,
                    embed_faithfulness,
                    lambda probe: reasoning_faithfulness(
                        deployment.bundle.system, probe.execution
                    ).value,
                ),
                (
                    ClauseId.FACTUAL_GROUNDEDNESS,
                    embed_groundedness,
                    lambda probe: factual_groundedness(
                        probe.execution, deployment.checkers.entailment
                    ).value,
                ),
            ):
                benchmark = benchmarks[clause.value]
                calibration_probes = embedder(
                    deployment.bundle, benchmark, calibration_requests, ratio, rng
                )
                test_probes = embedder(
                    deployment.bundle, benchmark, test_requests, ratio, rng
                )
                calibration_measures = [measure_fn(p) for p in calibration_probes]
                calibration_labels = [p.violates for p in calibration_probes]
                test_measures = np.asarray([measure_fn(p) for p in test_probes], dtype=float)
                test_labels = np.asarray([p.violates for p in test_probes], dtype=bool)

                for alpha in alphas:
                    targets = CalibrationTargets(
                        alpha=alpha, delta=delta, bound=bound, correction=correction
                    )
                    calibrated = learn_then_test(
                        measures=calibration_measures,
                        violates=calibration_labels,
                        thresholds=thresholds.tolist(),
                        targets=targets,
                    )
                    certified = test_measures >= calibrated.threshold
                    empirical = certified_violation_rate(certified, test_labels)
                    collected.setdefault((clause.value, alpha), []).append(empirical)
                    result.add(
                        seed=seed,
                        system=system_name,
                        modality=deployment.bundle.modality,
                        clause=clause.value,
                        alpha=alpha,
                        delta=delta,
                        threshold=calibrated.threshold,
                        calibration_empirical_risk=calibrated.empirical_risk,
                        calibration_p_value=calibrated.p_value,
                        empirical_certified_violation_rate=empirical,
                        controlled=empirical <= alpha,
                    )

    rng = np.random.default_rng(derive_seed(seeds[0], "rq3", "bootstrap"))
    for (clause, alpha), values in collected.items():
        result.metadata.setdefault("intervals", {}).setdefault(clause, {})[f"{alpha}"] = (
            mean_interval(values, resamples, confidence, rng).to_dict()
        )
    result.write(artifact_root(config))
    return result
