from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Mapping, Sequence

import numpy as np

from ..errors import BenchmarkError
from ..execution import (
    Claim,
    EvidenceItem,
    Execution,
    Input,
    Reason,
    Response,
)
from ..policy.clause import ClauseId
from ..systems.registry import SystemBundle
from .base import BenchmarkCase, BenchmarkSet


@dataclass(frozen=True)
class ProbeCase:
    identifier: str
    clause: ClauseId
    execution: Execution
    violates: bool
    source: str
    carrier_identifier: str


def _pair(
    cases: Sequence[BenchmarkCase],
    carriers: Sequence[Input],
    ratio: float,
    rng: np.random.Generator,
) -> Iterator[tuple[BenchmarkCase, Input]]:
    violating = [c for c in cases if c.violates]
    compliant = [c for c in cases if not c.violates]
    if not violating or not compliant:
        raise BenchmarkError(
            "the benchmark must supply both violating and compliant cases so that recall and "
            "the false-block rate are both measurable"
        )
    if not carriers:
        raise BenchmarkError("the held-out stream holds no carrier executions")
    if not 0.0 < ratio < 1.0:
        raise BenchmarkError("the violation ratio must lie strictly within (0, 1)")

    count = min(len(carriers), len(violating) + len(compliant))
    violating_count = int(round(count * ratio))
    compliant_count = count - violating_count
    violating_count = min(violating_count, len(violating))
    compliant_count = min(compliant_count, len(compliant))
    if violating_count == 0 or compliant_count == 0:
        raise BenchmarkError(
            "the configured violation ratio leaves one class empty for this benchmark and "
            "held-out stream size"
        )

    selected = [
        violating[int(i)] for i in rng.choice(len(violating), violating_count, replace=False)
    ] + [
        compliant[int(i)] for i in rng.choice(len(compliant), compliant_count, replace=False)
    ]
    order = rng.permutation(len(selected))
    carrier_indices = rng.choice(len(carriers), len(selected), replace=len(carriers) < len(selected))
    for position, index in enumerate(order):
        yield selected[int(index)], carriers[int(carrier_indices[position])]


def embed_non_discrimination(
    bundle: SystemBundle,
    benchmark: BenchmarkSet,
    carriers: Sequence[Input],
    ratio: float,
    rng: np.random.Generator,
) -> list[ProbeCase]:
    probes: list[ProbeCase] = []
    for case, carrier in _pair(benchmark.cases, carriers, ratio, rng):
        context = f"{case.payload['context']} {case.payload['question']}"
        request = replace(carrier, request=f"{carrier.request}; biased_context={context}")
        execution = bundle.system.execute(request)
        probes.append(
            ProbeCase(
                identifier=case.identifier,
                clause=ClauseId.NON_DISCRIMINATION,
                execution=execution,
                violates=case.violates,
                source=case.source,
                carrier_identifier=carrier.identifier,
            )
        )
    return probes


def embed_data_minimisation(
    bundle: SystemBundle,
    benchmark: BenchmarkSet,
    carriers: Sequence[Input],
    ratio: float,
    rng: np.random.Generator,
) -> list[ProbeCase]:
    probes: list[ProbeCase] = []
    for case, carrier in _pair(benchmark.cases, carriers, ratio, rng):
        note = str(case.payload["text"])
        request = replace(carrier, request=f"{carrier.request}; clinical_note={note}")
        execution = bundle.system.execute(request)
        augmented_trace = replace(
            execution.trace, text=f"{execution.trace.text}\n{note}"
        )
        augmented_response = replace(
            execution.response, text=f"{execution.response.text}\n{note}"
        )
        probes.append(
            ProbeCase(
                identifier=case.identifier,
                clause=ClauseId.DATA_MINIMISATION,
                execution=replace(
                    execution, trace=augmented_trace, response=augmented_response
                ),
                violates=case.violates,
                source=case.source,
                carrier_identifier=carrier.identifier,
            )
        )
    return probes


def embed_faithfulness(
    bundle: SystemBundle,
    benchmark: BenchmarkSet,
    carriers: Sequence[Input],
    ratio: float,
    rng: np.random.Generator,
) -> list[ProbeCase]:
    probes: list[ProbeCase] = []
    for case, carrier in _pair(benchmark.cases, carriers, ratio, rng):
        execution = bundle.system.execute(carrier)
        statements = [
            line.strip()
            for line in str(case.payload["trace"]).splitlines()
            if line.strip()
        ]
        if not statements:
            continue
        stated_groups = [reason.support["group"] for reason in execution.trace.reasons]
        reasons = tuple(
            Reason(
                identifier=f"reason:{stated_groups[index % len(stated_groups)]}:{index}",
                statement=statement,
                support={"group": stated_groups[index % len(stated_groups)]},
            )
            for index, statement in enumerate(statements)
        )
        trace = replace(
            execution.trace,
            reasons=reasons,
            text="\n".join(statements),
        )
        probes.append(
            ProbeCase(
                identifier=case.identifier,
                clause=ClauseId.REASONING_FAITHFULNESS,
                execution=replace(execution, trace=trace),
                violates=case.violates,
                source=case.source,
                carrier_identifier=carrier.identifier,
            )
        )
    if not probes:
        raise BenchmarkError("no faithfulness probe carried a non-empty reasoning trace")
    return probes


def embed_groundedness(
    bundle: SystemBundle,
    benchmark: BenchmarkSet,
    carriers: Sequence[Input],
    ratio: float,
    rng: np.random.Generator,
) -> list[ProbeCase]:
    probes: list[ProbeCase] = []
    for case, carrier in _pair(benchmark.cases, carriers, ratio, rng):
        execution = bundle.system.execute(carrier)
        evidence = EvidenceItem(
            identifier=f"evidence:{case.identifier}",
            content=str(case.payload["evidence"]),
            provenance=f"benchmark:{case.source}",
            modality="text",
        )
        claim = Claim(
            identifier=f"claim:{case.identifier}",
            statement=str(case.payload["claim"]),
            evidence_ids=(evidence.identifier,),
        )
        probes.append(
            ProbeCase(
                identifier=case.identifier,
                clause=ClauseId.FACTUAL_GROUNDEDNESS,
                execution=replace(
                    execution,
                    inputs=replace(execution.inputs, evidence=(evidence,)),
                    response=Response(claims=(claim,), text=claim.statement),
                ),
                violates=case.violates,
                source=case.source,
                carrier_identifier=carrier.identifier,
            )
        )
    return probes


def build_precondition_set(
    bundle: SystemBundle,
    carriers: Sequence[Input],
    ratio: float,
    rng: np.random.Generator,
) -> list[ProbeCase]:
    if not carriers:
        raise BenchmarkError("the held-out stream holds no carrier executions")
    probes: list[ProbeCase] = []
    for carrier in carriers:
        execution = bundle.system.execute(carrier)
        required = sorted(bundle.action_register.required_checks(execution.action))
        recorded = list(execution.trace.recorded_checks)
        present = [check for check in required if check in recorded]
        if not present:
            continue
        violate = bool(rng.random() < ratio)
        if violate:
            deleted = present[int(rng.integers(0, len(present)))]
            remaining = tuple(check for check in recorded if check != deleted)
            trace = replace(
                execution.trace,
                recorded_checks=remaining,
                commit_index=len(remaining),
                text="\n".join(
                    line
                    for line in execution.trace.text.splitlines()
                    if deleted not in line
                ),
            )
            execution = replace(execution, trace=trace)
        probes.append(
            ProbeCase(
                identifier=f"precondition:{carrier.identifier}",
                clause=ClauseId.PRECONDITION_COMPLIANCE,
                execution=execution,
                violates=violate,
                source="internal deletion of a required check (no public benchmark exists)",
                carrier_identifier=carrier.identifier,
            )
        )
    if not probes:
        raise BenchmarkError(
            "no execution recorded a required check before its commit point; the precondition "
            "clause cannot be exercised on this system"
        )
    violating = sum(1 for p in probes if p.violates)
    if violating == 0 or violating == len(probes):
        raise BenchmarkError(
            "the precondition violation set is single-class; adjust the configured ratio"
        )
    return probes


EMBEDDERS: Mapping[ClauseId, object] = {
    ClauseId.NON_DISCRIMINATION: embed_non_discrimination,
    ClauseId.DATA_MINIMISATION: embed_data_minimisation,
    ClauseId.REASONING_FAITHFULNESS: embed_faithfulness,
    ClauseId.FACTUAL_GROUNDEDNESS: embed_groundedness,
}
