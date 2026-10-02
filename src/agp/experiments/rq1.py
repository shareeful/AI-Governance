from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..baselines.clause_baselines import build_clause_baselines
from ..benchmarks.base import BenchmarkSet
from ..benchmarks.embedding import EMBEDDERS, ProbeCase, build_precondition_set
from ..certify import CertifiedDeployment, certify
from ..config import Config
from ..execution import Verdict
from ..policy.clause import ClauseId
from ..policy.guards import decide
from ..policy.measures import (
    counterfactual_invariance,
    data_minimisation,
    factual_groundedness,
    precondition_compliance,
    reasoning_faithfulness,
)
from ..seeding import derive_seed, seed_everything
from ..stats.bootstrap import mean_interval
from ..stats.metrics import clause_metrics
from ..stats.tests import compare_paired
from .base import (
    ExperimentResult,
    artifact_root,
    bootstrap_settings,
    family_wise_alpha,
    seeds_from,
    systems_from,
)


def _blocked(
    deployment: CertifiedDeployment,
    clause: ClauseId,
    probe: ProbeCase,
) -> bool:
    record = deployment.bound_policy.record(clause)
    binding = deployment.bound_policy[clause]
    execution = probe.execution
    system = deployment.bundle.system

    if clause is ClauseId.NON_DISCRIMINATION:
        measure = counterfactual_invariance(
            system, execution.inputs, execution.action, binding.protected_attributes
        )
    elif clause is ClauseId.DATA_MINIMISATION:
        measure = data_minimisation(
            execution, deployment.checkers.deidentifier, binding.minimum_necessary
        )
    elif clause is ClauseId.REASONING_FAITHFULNESS:
        measure = reasoning_faithfulness(system, execution)
    elif clause is ClauseId.PRECONDITION_COMPLIANCE:
        measure = precondition_compliance(
            execution, binding.required_checks[execution.action]
        )
    elif clause is ClauseId.FACTUAL_GROUNDEDNESS:
        measure = factual_groundedness(execution, deployment.checkers.entailment)
    else:
        raise ValueError(f"unhandled clause {clause}")

    return decide(record, measure.value) is Verdict.REJECT


def probes_for_clause(
    deployment: CertifiedDeployment,
    clause: ClauseId,
    benchmarks: Mapping[str, BenchmarkSet],
    carriers: Sequence,
    ratio: float,
    rng: np.random.Generator,
) -> list[ProbeCase]:
    if clause is ClauseId.PRECONDITION_COMPLIANCE:
        return build_precondition_set(deployment.bundle, carriers, ratio, rng)
    embedder = EMBEDDERS[clause]
    return embedder(deployment.bundle, benchmarks[clause.value], carriers, ratio, rng)


def run(config: Config, benchmarks: Mapping[str, BenchmarkSet]) -> ExperimentResult:
    result = ExperimentResult(name="rq1_clause_enforcement")
    seeds = seeds_from(config)
    systems = systems_from(config)
    ratio = config.require_probability("experiments.violation_ratio")
    resamples, confidence = bootstrap_settings(config)
    baseline_names = [str(n) for n in config.list_value("experiments.clause_baselines")]

    per_seed_baseline: dict[str, list[float]] = {name: [] for name in baseline_names}
    per_seed_baseline_fb: dict[str, list[float]] = {name: [] for name in baseline_names}

    for seed in seeds:
        seed_everything(seed)
        for system_name in systems:
            deployment = certify(config, system_name, seed, benchmarks, enable_ledger=False)
            carriers = deployment.bundle.split("test")
            rng = np.random.default_rng(derive_seed(seed, system_name, "rq1"))

            for clause in ClauseId:
                probes = probes_for_clause(
                    deployment, clause, benchmarks, carriers, ratio, rng
                )
                blocked = [_blocked(deployment, clause, probe) for probe in probes]
                violates = [probe.violates for probe in probes]
                metrics = clause_metrics(blocked, violates)
                result.add(
                    seed=seed,
                    system=system_name,
                    modality=deployment.bundle.modality,
                    clause=clause.value,
                    benchmark=probes[0].source if probes else None,
                    external_benchmark=clause is not ClauseId.PRECONDITION_COMPLIANCE,
                    **metrics.to_dict(),
                )

            all_probes: list[ProbeCase] = []
            for clause in ClauseId:
                all_probes.extend(
                    probes_for_clause(deployment, clause, benchmarks, carriers, ratio, rng)
                )
            baselines = build_clause_baselines(
                names=baseline_names,
                injection=deployment.checkers.injection,
                safety=deployment.checkers.safety,
                entailment=deployment.checkers.entailment,
                deidentifier=deployment.checkers.deidentifier,
                bound_policy=deployment.bound_policy,
                pipeline=deployment.pipeline,
                minimum_necessary=deployment.bundle.system.minimum_necessary_personal_data(),
            )
            for baseline in baselines:
                blocked = [baseline.blocks(probe) for probe in all_probes]
                violates = [probe.violates for probe in all_probes]
                metrics = clause_metrics(blocked, violates)
                per_seed_baseline[baseline.name].append(metrics.recall)
                per_seed_baseline_fb[baseline.name].append(metrics.false_block_rate)
                result.add(
                    seed=seed,
                    system=system_name,
                    modality=deployment.bundle.modality,
                    clause="all",
                    baseline=baseline.name,
                    **metrics.to_dict(),
                )

    rng = np.random.default_rng(derive_seed(seeds[0], "rq1", "bootstrap"))
    for name in baseline_names:
        recall_interval = mean_interval(per_seed_baseline[name], resamples, confidence, rng)
        fb_interval = mean_interval(per_seed_baseline_fb[name], resamples, confidence, rng)
        result.metadata.setdefault("baseline_intervals", {})[name] = {
            "recall": recall_interval.to_dict(),
            "false_block_rate": fb_interval.to_dict(),
        }

    proposed = config.str_value("experiments.proposed_baseline_name")
    if proposed in per_seed_baseline:
        comparisons = compare_paired(
            {
                f"{proposed}_vs_{name}": (
                    per_seed_baseline[proposed],
                    per_seed_baseline[name],
                )
                for name in baseline_names
                if name != proposed
            },
            family_wise_alpha(config),
        )
        result.comparisons = [c.to_dict() for c in comparisons.values()]

    result.write(artifact_root(config))
    return result
