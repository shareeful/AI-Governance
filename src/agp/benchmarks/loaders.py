from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from ..config import Config
from ..errors import BenchmarkError, MissingResourceError
from .base import BenchmarkCase, BenchmarkSet, read_jsonl, require_fields


def load_bbq(config: Config) -> BenchmarkSet:
    prefix = "benchmarks.non_discrimination"
    path = config.path(f"{prefix}.path")
    context_field = config.str_value(f"{prefix}.context_field")
    question_field = config.str_value(f"{prefix}.question_field")
    label_field = config.str_value(f"{prefix}.label_field")
    identifier_field = config.str_value(f"{prefix}.identifier_field")
    ambiguous_field = config.str_value(f"{prefix}.ambiguous_context_field")
    ambiguous_value = config.str_value(f"{prefix}.ambiguous_context_value")
    answer_fields = tuple(str(f) for f in config.list_value(f"{prefix}.answer_fields"))
    unknown_values = frozenset(str(v) for v in config.list_value(f"{prefix}.unknown_answer_values"))

    cases: list[BenchmarkCase] = []
    for row in read_jsonl(path):
        require_fields(
            row,
            (context_field, question_field, label_field, identifier_field, ambiguous_field),
            "BBQ",
        )
        answers = [str(row[field]) for field in answer_fields if field in row]
        gold_index = int(row[label_field])
        gold = answers[gold_index] if 0 <= gold_index < len(answers) else ""
        is_ambiguous = str(row[ambiguous_field]) == ambiguous_value
        cases.append(
            BenchmarkCase(
                identifier=str(row[identifier_field]),
                payload={
                    "context": str(row[context_field]),
                    "question": str(row[question_field]),
                    "answers": answers,
                    "gold": gold,
                    "unknown_values": sorted(unknown_values),
                    "ambiguous": is_ambiguous,
                },
                violates=is_ambiguous and gold not in unknown_values,
                source="BBQ",
            )
        )
    if not cases:
        raise BenchmarkError(f"BBQ benchmark at {path} yielded no cases")
    return BenchmarkSet(name="BBQ", clause="non_discrimination", cases=tuple(cases), external=True)


def load_i2b2(config: Config) -> BenchmarkSet:
    prefix = "benchmarks.data_minimisation"
    root = config.path(f"{prefix}.root")
    text_suffix = config.str_value(f"{prefix}.text_suffix")
    annotation_suffix = config.str_value(f"{prefix}.annotation_suffix")
    permitted_categories = frozenset(
        str(c) for c in config.list_value(f"{prefix}.permitted_categories")
    )

    cases: list[BenchmarkCase] = []
    for text_path in sorted(root.rglob(f"*{text_suffix}")):
        annotation_path = text_path.with_suffix(annotation_suffix)
        if not annotation_path.is_file():
            continue
        text = text_path.read_text(encoding="utf-8", errors="strict")
        categories = _parse_i2b2_annotations(annotation_path)
        excess = categories - permitted_categories
        cases.append(
            BenchmarkCase(
                identifier=text_path.stem,
                payload={"text": text, "categories": sorted(categories)},
                violates=bool(excess),
                source="i2b2/UTHealth",
            )
        )
    if not cases:
        raise MissingResourceError(
            f"no i2b2 document/annotation pairs found beneath {root}; the corpus is obtained "
            "under its own data-use agreement"
        )
    return BenchmarkSet(
        name="i2b2-UTHealth", clause="data_minimisation", cases=tuple(cases), external=True
    )


def _parse_i2b2_annotations(path: Path) -> frozenset[str]:
    import xml.etree.ElementTree as ElementTree

    try:
        tree = ElementTree.parse(path)
    except ElementTree.ParseError as exc:
        raise BenchmarkError(f"i2b2 annotation file {path} is not valid XML") from exc
    categories: set[str] = set()
    for element in tree.iter():
        type_attribute = element.get("TYPE")
        if type_attribute:
            categories.add(str(type_attribute))
    return frozenset(categories)


def load_faithfulness(config: Config) -> BenchmarkSet:
    prefix = "benchmarks.reasoning_faithfulness"
    cases: list[BenchmarkCase] = []
    for protocol in ("early_answering", "added_mistake", "biased_context"):
        path = config.path(f"{prefix}.{protocol}.path")
        identifier_field = config.str_value(f"{prefix}.{protocol}.identifier_field")
        trace_field = config.str_value(f"{prefix}.{protocol}.trace_field")
        answer_field = config.str_value(f"{prefix}.{protocol}.answer_field")
        perturbed_answer_field = config.str_value(f"{prefix}.{protocol}.perturbed_answer_field")
        for row in read_jsonl(path):
            require_fields(
                row,
                (identifier_field, trace_field, answer_field, perturbed_answer_field),
                f"faithfulness/{protocol}",
            )
            unchanged = str(row[answer_field]) == str(row[perturbed_answer_field])
            cases.append(
                BenchmarkCase(
                    identifier=f"{protocol}:{row[identifier_field]}",
                    payload={
                        "protocol": protocol,
                        "trace": str(row[trace_field]),
                        "answer": str(row[answer_field]),
                        "perturbed_answer": str(row[perturbed_answer_field]),
                    },
                    violates=unchanged,
                    source="Lanham et al.; Turpin et al.",
                )
            )
    if not cases:
        raise BenchmarkError("the faithfulness protocols yielded no cases")
    return BenchmarkSet(
        name="faithfulness-protocols",
        clause="reasoning_faithfulness",
        cases=tuple(cases),
        external=True,
    )


def load_groundedness(config: Config) -> BenchmarkSet:
    prefix = "benchmarks.factual_groundedness"
    cases: list[BenchmarkCase] = []

    fever_path = config.path(f"{prefix}.fever.path")
    fever_claim = config.str_value(f"{prefix}.fever.claim_field")
    fever_evidence = config.str_value(f"{prefix}.fever.evidence_field")
    fever_label = config.str_value(f"{prefix}.fever.label_field")
    fever_identifier = config.str_value(f"{prefix}.fever.identifier_field")
    supported_label = config.str_value(f"{prefix}.fever.supported_label")
    for row in read_jsonl(fever_path):
        require_fields(
            row, (fever_claim, fever_evidence, fever_label, fever_identifier), "FEVER"
        )
        cases.append(
            BenchmarkCase(
                identifier=f"fever:{row[fever_identifier]}",
                payload={
                    "claim": str(row[fever_claim]),
                    "evidence": _flatten_evidence(row[fever_evidence]),
                },
                violates=str(row[fever_label]) != supported_label,
                source="FEVER",
            )
        )

    halueval_path = config.path(f"{prefix}.halueval.path")
    hallucinated_field = config.str_value(f"{prefix}.halueval.hallucinated_answer_field")
    correct_field = config.str_value(f"{prefix}.halueval.correct_answer_field")
    knowledge_field = config.str_value(f"{prefix}.halueval.knowledge_field")
    for index, row in enumerate(read_jsonl(halueval_path)):
        require_fields(row, (hallucinated_field, correct_field, knowledge_field), "HaluEval")
        evidence = str(row[knowledge_field])
        cases.append(
            BenchmarkCase(
                identifier=f"halueval:{index}:hallucinated",
                payload={"claim": str(row[hallucinated_field]), "evidence": evidence},
                violates=True,
                source="HaluEval",
            )
        )
        cases.append(
            BenchmarkCase(
                identifier=f"halueval:{index}:correct",
                payload={"claim": str(row[correct_field]), "evidence": evidence},
                violates=False,
                source="HaluEval",
            )
        )
    if not cases:
        raise BenchmarkError("the groundedness benchmarks yielded no cases")
    return BenchmarkSet(
        name="FEVER+HaluEval",
        clause="factual_groundedness",
        cases=tuple(cases),
        external=True,
    )


def _flatten_evidence(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "\n".join(_flatten_evidence(item) for item in raw)
    if isinstance(raw, Mapping):
        return "\n".join(f"{k}: {_flatten_evidence(v)}" for k, v in raw.items())
    return str(raw)


def load_injection(config: Config) -> BenchmarkSet:
    prefix = "benchmarks.injection"
    cases: list[BenchmarkCase] = []

    advbench_path = config.path(f"{prefix}.advbench.path")
    advbench_column = config.str_value(f"{prefix}.advbench.prompt_column")
    with advbench_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if advbench_column not in (reader.fieldnames or ()):
            raise BenchmarkError(
                f"AdvBench file {advbench_path} has no column '{advbench_column}'"
            )
        for index, row in enumerate(reader):
            cases.append(
                BenchmarkCase(
                    identifier=f"advbench:{index}",
                    payload={"prompt": str(row[advbench_column])},
                    violates=True,
                    source="AdvBench",
                )
            )

    jbb_path = config.path(f"{prefix}.jailbreakbench.path")
    jbb_prompt = config.str_value(f"{prefix}.jailbreakbench.prompt_field")
    jbb_label = config.str_value(f"{prefix}.jailbreakbench.label_field")
    jbb_harmful = config.str_value(f"{prefix}.jailbreakbench.harmful_value")
    for index, row in enumerate(read_jsonl(jbb_path)):
        require_fields(row, (jbb_prompt, jbb_label), "JailbreakBench")
        cases.append(
            BenchmarkCase(
                identifier=f"jailbreakbench:{index}",
                payload={"prompt": str(row[jbb_prompt])},
                violates=str(row[jbb_label]) == jbb_harmful,
                source="JailbreakBench",
            )
        )
    if not cases:
        raise BenchmarkError("the injection benchmarks yielded no cases")
    return BenchmarkSet(name="AdvBench+JailbreakBench", clause="injection", cases=tuple(cases), external=True)


def load_safety(config: Config) -> BenchmarkSet:
    prefix = "benchmarks.safety"
    path = config.path(f"{prefix}.path")
    text_field = config.str_value(f"{prefix}.text_field")
    label_field = config.str_value(f"{prefix}.label_field")
    unsafe_value = config.str_value(f"{prefix}.unsafe_value")
    cases = [
        BenchmarkCase(
            identifier=f"safety:{index}",
            payload={"text": str(row[text_field])},
            violates=str(row[label_field]) == unsafe_value,
            source="configured safety corpus",
        )
        for index, row in enumerate(read_jsonl(path))
    ]
    if not cases:
        raise BenchmarkError(f"the safety corpus at {path} yielded no cases")
    return BenchmarkSet(name="safety-corpus", clause="safety", cases=tuple(cases), external=True)


def load_all(config: Config) -> dict[str, BenchmarkSet]:
    return {
        "non_discrimination": load_bbq(config),
        "data_minimisation": load_i2b2(config),
        "reasoning_faithfulness": load_faithfulness(config),
        "factual_groundedness": load_groundedness(config),
        "injection": load_injection(config),
        "safety": load_safety(config),
    }
