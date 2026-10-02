from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..benchmarks.base import BenchmarkSet
from ..benchmarks.embedding import embed_faithfulness, embed_groundedness
from ..certify import certify, recertify
from ..config import Config
from ..errors import ConfigurationError
from ..execution import Input
from ..policy.clause import ClauseId
from ..policy.measures import factual_groundedness, reasoning_faithfulness
from ..seeding import derive_seed, seed_everything
from ..stats.bootstrap import mean_interval
from ..stats.metrics import certified_violation_rate
from ..trust.monitor import PolicyMonitor
from .base import (
    ExperimentResult,
    artifact_root,
    bootstrap_settings,
    seeds_from,
    systems_from,
)


def _drift_stream(
    config: Config,
    system_name: str,
    baseline: Sequence[Input],
    shifted: Sequence[Input],
    steps: int,
    rng: np.random.Generator,
) -> list[list[Input]]:
    if not shifted:
        raise ConfigurationError(
            f"system '{system_name}' declares no drift split; RQ4 requires a real held-out "
            "corpus drawn from a shifted acquisition site, device, or period, configured at "
            f"systems.{system_name}.dataset.splits.drift"
        )
    window = config.int_value("experiments.drift.window_size")
    windows: list[list[Input]] = []
    for step in range(steps):
        mix = step / max(steps - 1, 1)
        shifted_count = int(round(mix * window))
        baseline_count = window - shifted_count
        selected: list[Input] = []
        if baseline_count > 0:
            indices = rng.choice(len(baseline), baseline_count, replace=len(baseline) < baseline_count)
            selected.extend(baseline[int(i)] for i in indices)
        if shifted_count > 0:
            indices = rng.choice(len(shifted), shifted_count, replace=len(shifted) < shifted_count)
            selected.extend(shifted[int(i)] for i in indices)
        windows.append(selected)
    return windows


def run(config: Config, benchmarks: Mapping[str, BenchmarkSet]) -> ExperimentResult:
    result = ExperimentResult(name="rq4_durability_under_drift")
    seeds = seeds_from(config)
    systems = systems_from(config)
    steps = config.int_value("experiments.drift.steps")
    ratio = config.require_probability("experiments.violation_ratio")
    resamples, confidence = bootstrap_settings(config)

    collected: dict[tuple[str, str], list[float]] = {}

    for seed in seeds:
        seed_everything(seed)
        for system_name in systems:
            deployment = certify(config, system_name, seed, benchmarks, enable_ledger=False)
            static_policy = deployment.bound_policy
            rng = np.random.default_rng(derive_seed(seed, system_name, "rq4"))
            baseline_requests = deployment.bundle.split("test")
            shifted_requests = deployment.bundle.split("drift")
            windows = _drift_stream(
                config, system_name, baseline_requests, shifted_requests, steps, rng
            )

            embedding = np.array(
                [
                    float(np.mean(deployment.bundle.system.decision_scores(r)))
                    for r in deployment.bundle.split("calibration")
                ],
                dtype=float,
            )
            monitor = PolicyMonitor(config, embedding)
            adaptive = deployment
            recertifications = 0

            for step, window in enumerate(windows):
                for configuration, bound_policy in (
                    ("static", static_policy),
                    ("proposed", adaptive.bound_policy),
                ):
                    rate = _violation_rate(
                        adaptive, bound_policy, benchmarks, window, ratio, rng
                    )
                    collected.setdefault((system_name, configuration), []).append(rate)
                    result.add(
                        seed=seed,
                        system=system_name,
                        modality=deployment.bundle.modality,
                        step=step,
                        configuration=configuration,
                        certified_violation_rate=rate,
                        recertifications=recertifications,
                    )

                signal = _observe_window(adaptive, monitor, window)
                if signal.triggered:
                    adaptive = recertify(config, adaptive, window, benchmarks)
                    monitor.reset()
                    recertifications += 1
                    result.metadata.setdefault("recertification_steps", []).append(
                        {"seed": seed, "system": system_name, "step": step, "signal": signal.to_dict()}
                    )

    rng = np.random.default_rng(derive_seed(seeds[0], "rq4", "bootstrap"))
    for (system_name, configuration), values in collected.items():
        result.metadata.setdefault("intervals", {}).setdefault(system_name, {})[configuration] = (
            mean_interval(values, resamples, confidence, rng).to_dict()
        )
    result.write(artifact_root(config))
    return result


def _violation_rate(
    deployment,
    bound_policy,
    benchmarks: Mapping[str, BenchmarkSet],
    window: Sequence[Input],
    ratio: float,
    rng: np.random.Generator,
) -> float:
    certified: list[bool] = []
    violates: list[bool] = []
    for clause, embedder, measure_fn in (
        (
            ClauseId.REASONING_FAITHFULNESS,
            embed_faithfulness,
            lambda probe: reasoning_faithfulness(deployment.bundle.system, probe.execution).value,
        ),
        (
            ClauseId.FACTUAL_GROUNDEDNESS,
            embed_groundedness,
            lambda probe: factual_groundedness(probe.execution, deployment.checkers.entailment).value,
        ),
    ):
        probes = embedder(deployment.bundle, benchmarks[clause.value], window, ratio, rng)
        threshold = bound_policy.record(clause).threshold
        for probe in probes:
            certified.append(measure_fn(probe) >= threshold)
            violates.append(probe.violates)
    return certified_violation_rate(certified, violates)


def _observe_window(deployment, monitor: PolicyMonitor, window: Sequence[Input]):
    signal = None
    for request in window:
        result = deployment.pipeline(request)
        signal = monitor.observe(
            decision=result.decision,
            attested=result.attestation.attested if result.attestation else False,
            embedding=float(np.mean(deployment.bundle.system.decision_scores(request))),
        )
    if signal is None:
        raise ConfigurationError("the drift window held no execution")
    return signal
