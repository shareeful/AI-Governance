from __future__ import annotations

import time
from typing import Mapping

import numpy as np

from ..attestation.signals import compute_signals
from ..benchmarks.base import BenchmarkSet
from ..certify import certify
from ..config import Config
from ..seeding import derive_seed, seed_everything
from ..stats.bootstrap import mean_interval
from .base import (
    ExperimentResult,
    artifact_root,
    bootstrap_settings,
    seeds_from,
    systems_from,
)

COMPONENTS = ("input_guard", "reasoning_guard", "output_guard", "attestation")


def run(config: Config, benchmarks: Mapping[str, BenchmarkSet]) -> ExperimentResult:
    result = ExperimentResult(name="rq5_runtime_overhead")
    seeds = seeds_from(config)
    systems = systems_from(config)
    sample_size = config.int_value("experiments.overhead.sample_size")
    warmup = config.int_value("experiments.overhead.warmup")
    resamples, confidence = bootstrap_settings(config)

    collected: dict[str, list[float]] = {component: [] for component in COMPONENTS}
    ungoverned_samples: list[float] = []
    governed_samples: list[float] = []

    for seed in seeds:
        seed_everything(seed)
        for system_name in systems:
            deployment = certify(config, system_name, seed, benchmarks, enable_ledger=False)
            rng = np.random.default_rng(derive_seed(seed, system_name, "rq5"))
            requests = deployment.bundle.split("test")
            take = min(sample_size + warmup, len(requests))
            indices = rng.choice(len(requests), take, replace=False)
            selected = [requests[int(i)] for i in indices]

            for position, request in enumerate(selected):
                measured = position >= warmup

                start = time.perf_counter()
                execution = deployment.bundle.system.execute(request)
                ungoverned = time.perf_counter() - start

                start = time.perf_counter()
                admission = deployment.pipeline.input_guard(request)
                input_elapsed = time.perf_counter() - start

                start = time.perf_counter()
                deployment.pipeline.reasoning_guard(execution)
                reasoning_elapsed = time.perf_counter() - start

                start = time.perf_counter()
                deployment.pipeline.output_guard(
                    execution, request, admission.attribute_store
                )
                output_elapsed = time.perf_counter() - start

                start = time.perf_counter()
                signals = compute_signals(
                    deployment.bundle.system,
                    execution,
                    deployment.signal_configuration,
                    rng,
                )
                deployment.detector.observe(signals)
                attestation_elapsed = time.perf_counter() - start

                if not measured:
                    continue

                elapsed = {
                    "input_guard": input_elapsed,
                    "reasoning_guard": reasoning_elapsed,
                    "output_guard": output_elapsed,
                    "attestation": attestation_elapsed,
                }
                for component, value in elapsed.items():
                    collected[component].append(value * 1000.0)
                added = sum(elapsed.values())
                ungoverned_samples.append(ungoverned)
                governed_samples.append(ungoverned + added)
                result.add(
                    seed=seed,
                    system=system_name,
                    modality=deployment.bundle.modality,
                    ungoverned_seconds=ungoverned,
                    governed_seconds=ungoverned + added,
                    added_seconds=added,
                    **{f"{k}_ms": v * 1000.0 for k, v in elapsed.items()},
                )

    rng = np.random.default_rng(derive_seed(seeds[0], "rq5", "bootstrap"))
    total_added = float(np.sum([np.mean(collected[c]) for c in COMPONENTS]))
    for component in COMPONENTS:
        interval = mean_interval(collected[component], resamples, confidence, rng)
        result.metadata.setdefault("components", {})[component] = {
            "added_latency_ms": interval.to_dict(),
            "share": float(np.mean(collected[component]) / total_added)
            if total_added > 0.0
            else 0.0,
        }
    result.metadata["total_added_latency_ms"] = total_added
    result.metadata["total_added_latency_seconds"] = total_added / 1000.0
    ungoverned_throughput = 1.0 / float(np.mean(ungoverned_samples))
    governed_throughput = 1.0 / float(np.mean(governed_samples))
    result.metadata["throughput_retained"] = governed_throughput / ungoverned_throughput
    result.write(artifact_root(config))
    return result
