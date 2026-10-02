from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..errors import MissingResourceError


def load_result(root: Path, name: str) -> dict[str, Any]:
    path = root / "results" / f"{name}.json"
    if not path.is_file():
        raise MissingResourceError(
            f"no result file at {path}; run the corresponding experiment before reporting"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def frame_of(result: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(result["rows"])
    if frame.empty:
        raise MissingResourceError(
            f"experiment '{result['experiment']}' produced no rows"
        )
    return frame


def clause_by_system(root: Path) -> pd.DataFrame:
    frame = frame_of(load_result(root, "rq1_clause_enforcement"))
    clause_rows = frame[frame["clause"] != "all"]
    return (
        clause_rows.pivot_table(
            index="clause", columns="system", values="recall", aggfunc="mean"
        )
        .round(4)
    )


def modality_table(root: Path) -> pd.DataFrame:
    frame = frame_of(load_result(root, "rq1_clause_enforcement"))
    clause_rows = frame[frame["clause"] != "all"]
    grouped = clause_rows.groupby(["clause", "modality"]).agg(
        recall=("recall", "mean"),
        f1=("f1", "mean"),
        false_block_rate=("false_block_rate", "mean"),
        precision=("precision", "mean"),
    )
    pooled = clause_rows.groupby("clause").agg(
        pooled_precision=("precision", "mean"),
        pooled_f1=("f1", "mean"),
        pooled_false_block_rate=("false_block_rate", "mean"),
    )
    wide = grouped.unstack("modality")
    wide.columns = [f"{metric}_{modality}" for metric, modality in wide.columns]
    combined = wide.join(pooled)
    combined.loc["mean"] = combined.mean(numeric_only=True)
    return combined.round(4)


def integrity_table(root: Path) -> pd.DataFrame:
    frame = frame_of(load_result(root, "rq2_integrity_detection"))
    table = frame.pivot_table(
        index=["detector", "access", "runtime"],
        columns="attack_family",
        values="auc",
        aggfunc="mean",
    )
    return table.round(4)


def calibration_table(root: Path) -> pd.DataFrame:
    frame = frame_of(load_result(root, "rq3_threshold_validity"))
    return (
        frame.groupby(["clause", "alpha"])
        .agg(
            empirical_risk=("empirical_certified_violation_rate", "mean"),
            controlled=("controlled", "mean"),
            threshold=("threshold", "mean"),
        )
        .round(4)
    )


def drift_table(root: Path) -> pd.DataFrame:
    frame = frame_of(load_result(root, "rq4_durability_under_drift"))
    return (
        frame.groupby(["modality", "configuration"])
        .agg(
            mean_violation_rate=("certified_violation_rate", "mean"),
            min_violation_rate=("certified_violation_rate", "min"),
            max_violation_rate=("certified_violation_rate", "max"),
        )
        .round(4)
    )


def overhead_table(root: Path) -> pd.DataFrame:
    result = load_result(root, "rq5_runtime_overhead")
    components = result["metadata"]["components"]
    rows = [
        {
            "component": name,
            "added_latency_ms": payload["added_latency_ms"]["point"],
            "ci_lower_ms": payload["added_latency_ms"]["lower"],
            "ci_upper_ms": payload["added_latency_ms"]["upper"],
            "share": payload["share"],
        }
        for name, payload in components.items()
    ]
    frame = pd.DataFrame(rows)
    frame.loc[len(frame)] = {
        "component": "total_added",
        "added_latency_ms": result["metadata"]["total_added_latency_ms"],
        "ci_lower_ms": np.nan,
        "ci_upper_ms": np.nan,
        "share": 1.0,
    }
    frame.loc[len(frame)] = {
        "component": "throughput_retained_vs_ungoverned",
        "added_latency_ms": np.nan,
        "ci_lower_ms": np.nan,
        "ci_upper_ms": np.nan,
        "share": result["metadata"]["throughput_retained"],
    }
    return frame.round(4)


def ablation_table(root: Path) -> pd.DataFrame:
    result = load_result(root, "ablation")
    frame = frame_of(result)
    table = (
        frame.groupby("configuration")
        .agg(
            detection_recall=("detection_recall", "mean"),
            false_block_rate=("false_block_rate", "mean"),
            integrity_at_budget=("integrity_at_budget", "mean"),
            missed_referral_rate=("missed_referral_rate", "mean"),
        )
        .round(4)
    )
    significance = {
        comparison["label"]: comparison["significant"]
        for comparison in result.get("comparisons", [])
    }
    for metric in (
        "detection_recall",
        "false_block_rate",
        "integrity_at_budget",
        "missed_referral_rate",
    ):
        table[f"{metric}_significant"] = [
            significance.get(f"{configuration}:{metric}", False)
            for configuration in table.index
        ]
    return table


def baseline_table(root: Path) -> pd.DataFrame:
    result = load_result(root, "rq1_clause_enforcement")
    intervals = result["metadata"]["baseline_intervals"]
    rows = [
        {
            "baseline": name,
            "recall": payload["recall"]["point"],
            "recall_ci_lower": payload["recall"]["lower"],
            "recall_ci_upper": payload["recall"]["upper"],
            "false_block_rate": payload["false_block_rate"]["point"],
            "false_block_ci_lower": payload["false_block_rate"]["lower"],
            "false_block_ci_upper": payload["false_block_rate"]["upper"],
        }
        for name, payload in intervals.items()
    ]
    return pd.DataFrame(rows).round(4)


def write_all(root: Path, formats: Sequence[str]) -> list[Path]:
    builders = {
        "table_clause_by_system": clause_by_system,
        "table_modality": modality_table,
        "table_integrity": integrity_table,
        "table_calibration": calibration_table,
        "table_drift": drift_table,
        "table_overhead": overhead_table,
        "table_ablation": ablation_table,
        "table_baselines": baseline_table,
    }
    output_root = root / "tables"
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, builder in builders.items():
        frame = builder(root)
        for extension in formats:
            path = output_root / f"{name}.{extension}"
            if extension == "csv":
                frame.to_csv(path)
            elif extension == "tex":
                path.write_text(frame.to_latex(escape=False), encoding="utf-8")
            elif extension == "md":
                path.write_text(frame.to_markdown(), encoding="utf-8")
            else:
                raise ValueError(f"unsupported table format '{extension}'")
            written.append(path)
    return written
