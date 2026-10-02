from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .tables import frame_of, load_result


def _prepare(root: Path) -> Path:
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _save(figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    return path


def clause_heatmap(root: Path, extension: str) -> Path:
    import matplotlib.pyplot as plt

    frame = frame_of(load_result(root, "rq1_clause_enforcement"))
    clause_rows = frame[frame["clause"] != "all"]
    pivot = clause_rows.pivot_table(
        index="clause", columns="system", values="recall", aggfunc="mean"
    )
    modality = (
        clause_rows.groupby("system")["modality"].first().reindex(pivot.columns)
    )
    order = list(modality.sort_values(kind="stable").index)
    pivot = pivot[order]

    figure, axes = plt.subplots(figsize=(1.6 * len(pivot.columns) + 2, 0.7 * len(pivot) + 2))
    image = axes.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0)
    axes.set_xticks(range(len(pivot.columns)))
    axes.set_xticklabels(pivot.columns, rotation=45, ha="right")
    axes.set_yticks(range(len(pivot.index)))
    axes.set_yticklabels(pivot.index)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            axes.text(
                column,
                row,
                f"{pivot.iat[row, column]:.2f}",
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axes, label="violation-detection recall")
    axes.set_title("Violation-detection recall per clause and system")
    return _save(figure, _prepare(root) / f"fig_rq1_heatmap.{extension}")


def baseline_bars(root: Path, extension: str) -> Path:
    import matplotlib.pyplot as plt

    result = load_result(root, "rq1_clause_enforcement")
    intervals = result["metadata"]["baseline_intervals"]
    names = list(intervals)
    recall = [intervals[n]["recall"]["point"] for n in names]
    recall_error = [
        [intervals[n]["recall"]["point"] - intervals[n]["recall"]["lower"] for n in names],
        [intervals[n]["recall"]["upper"] - intervals[n]["recall"]["point"] for n in names],
    ]
    false_block = [intervals[n]["false_block_rate"]["point"] for n in names]
    false_block_error = [
        [
            intervals[n]["false_block_rate"]["point"] - intervals[n]["false_block_rate"]["lower"]
            for n in names
        ],
        [
            intervals[n]["false_block_rate"]["upper"] - intervals[n]["false_block_rate"]["point"]
            for n in names
        ],
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, recall, yerr=recall_error, capsize=4)
    axes[0].set_ylabel("violation-detection recall")
    axes[0].set_title("(a) recall")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(names, false_block, yerr=false_block_error, capsize=4)
    axes[1].set_ylabel("false-block rate")
    axes[1].set_title("(b) false blocks")
    axes[1].tick_params(axis="x", rotation=45)
    return _save(figure, _prepare(root) / f"fig_rq1_baselines.{extension}")


def integrity_roc(root: Path, extension: str) -> Path:
    import matplotlib.pyplot as plt

    result = load_result(root, "rq2_integrity_detection")
    roc = result["metadata"].get("roc", {})
    families = [f for f in ("A1", "A2") if f in roc]
    if not families:
        raise ValueError("no ROC curves were recorded for attack families A1 or A2")

    figure, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 4.5))
    if len(families) == 1:
        axes = [axes]
    for index, family in enumerate(families):
        axis = axes[index]
        for key, curve in roc[family].items():
            axis.plot(
                curve["false_positive_rate"],
                curve["true_positive_rate"],
                label=key.split(":", 1)[0],
                alpha=0.7,
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
        axis.set_xlabel("false-alarm rate")
        axis.set_ylabel("detection rate")
        axis.set_title(f"({chr(97 + index)}) {family}")
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), loc="lower right")
    return _save(figure, _prepare(root) / f"fig_rq2_roc.{extension}")


def attack_sensitivity(root: Path, extension: str) -> Path:
    import matplotlib.pyplot as plt

    frame = frame_of(load_result(root, "rq2_attack_strength_sensitivity"))
    families = sorted(frame["attack_family"].unique())
    figure, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 4.5))
    if len(families) == 1:
        axes = [axes]
    for index, family in enumerate(families):
        axis = axes[index]
        subset = frame[frame["attack_family"] == family]
        for detector, group in subset.groupby("detector"):
            summary = group.groupby("attack_strength")["detection_at_budget"].agg(
                ["mean", "std", "count"]
            )
            error = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
            axis.plot(summary.index, summary["mean"], marker="o", label=detector)
            axis.fill_between(
                summary.index,
                summary["mean"] - error,
                summary["mean"] + error,
                alpha=0.2,
            )
        axis.set_xlabel("attack strength")
        axis.set_ylabel("detection rate at the false-alarm budget")
        axis.set_title(f"({chr(97 + index)}) {family}")
        axis.legend()
    return _save(figure, _prepare(root) / f"fig_rq2_sensitivity.{extension}")


def calibration_curves(root: Path, extension: str) -> Path:
    import matplotlib.pyplot as plt

    frame = frame_of(load_result(root, "rq3_threshold_validity"))
    clauses = sorted(frame["clause"].unique())
    figure, axes = plt.subplots(1, len(clauses), figsize=(6 * len(clauses), 4.5))
    if len(clauses) == 1:
        axes = [axes]
    for index, clause in enumerate(clauses):
        axis = axes[index]
        subset = frame[frame["clause"] == clause]
        summary = subset.groupby("alpha")["empirical_certified_violation_rate"].agg(
            ["mean", "std", "count"]
        )
        error = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
        axis.errorbar(summary.index, summary["mean"], yerr=error, marker="o", capsize=3)
        limit = float(max(summary.index))
        axis.plot([0, limit], [0, limit], linestyle="--", color="grey", label="target")
        axis.set_xlabel("target alpha")
        axis.set_ylabel("empirical certified-violation rate")
        axis.set_title(f"({chr(97 + index)}) {clause}")
        axis.legend()
    return _save(figure, _prepare(root) / f"fig_rq3_calibration.{extension}")


def drift_curves(root: Path, extension: str) -> Path:
    import matplotlib.pyplot as plt

    result = load_result(root, "rq4_durability_under_drift")
    frame = frame_of(result)
    modalities = sorted(frame["modality"].unique())
    figure, axes = plt.subplots(1, len(modalities), figsize=(6 * len(modalities), 4.5))
    if len(modalities) == 1:
        axes = [axes]
    events = pd.DataFrame(result["metadata"].get("recertification_steps", []))
    for index, modality in enumerate(modalities):
        axis = axes[index]
        subset = frame[frame["modality"] == modality]
        for configuration, group in subset.groupby("configuration"):
            summary = group.groupby("step")["certified_violation_rate"].mean()
            axis.plot(summary.index, summary.values, marker="o", label=configuration)
        if not events.empty:
            systems = set(subset["system"].unique())
            for step in sorted(events[events["system"].isin(systems)]["step"].unique()):
                axis.axvline(step, linestyle=":", color="grey")
        axis.set_xlabel("operating window")
        axis.set_ylabel("certified-violation rate")
        axis.set_title(f"({chr(97 + index)}) {modality}")
        axis.legend()
    return _save(figure, _prepare(root) / f"fig_rq4_drift.{extension}")


def write_all(root: Path, extension: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    builders = (
        clause_heatmap,
        baseline_bars,
        integrity_roc,
        attack_sensitivity,
        calibration_curves,
        drift_curves,
    )
    return [builder(root, extension) for builder in builders]
