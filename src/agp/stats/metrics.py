from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..errors import GovernanceError


@dataclass(frozen=True)
class ClauseMetrics:
    recall: float
    precision: float
    f1: float
    false_block_rate: float
    support_violating: int
    support_compliant: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "false_block_rate": self.false_block_rate,
            "support_violating": self.support_violating,
            "support_compliant": self.support_compliant,
        }


def clause_metrics(blocked: Sequence[bool], violates: Sequence[bool]) -> ClauseMetrics:
    blocked_array = np.asarray(blocked, dtype=bool)
    violates_array = np.asarray(violates, dtype=bool)
    if blocked_array.shape != violates_array.shape:
        raise GovernanceError("block and violation vectors must have identical shape")
    if blocked_array.size == 0:
        raise GovernanceError("no probe was evaluated")

    true_positive = int(np.sum(blocked_array & violates_array))
    false_positive = int(np.sum(blocked_array & ~violates_array))
    false_negative = int(np.sum(~blocked_array & violates_array))
    compliant = int(np.sum(~violates_array))
    violating = int(np.sum(violates_array))
    if violating == 0:
        raise GovernanceError("recall is undefined without a violating probe")
    if compliant == 0:
        raise GovernanceError("the false-block rate is undefined without a compliant probe")

    recall = true_positive / violating
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    return ClauseMetrics(
        recall=recall,
        precision=precision,
        f1=f1,
        false_block_rate=false_positive / compliant,
        support_violating=violating,
        support_compliant=compliant,
    )


def detection_auc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    from sklearn.metrics import roc_auc_score

    score_array = np.asarray(scores, dtype=float)
    positive_array = np.asarray(positive, dtype=bool)
    if score_array.shape != positive_array.shape:
        raise GovernanceError("score and label vectors must have identical shape")
    if positive_array.all() or not positive_array.any():
        raise GovernanceError("the detection AUC requires both compromised and clean samples")
    return float(roc_auc_score(positive_array, score_array))


def detection_at_false_alarm(
    scores: Sequence[float],
    positive: Sequence[bool],
    budget: float,
) -> float:
    score_array = np.asarray(scores, dtype=float)
    positive_array = np.asarray(positive, dtype=bool)
    clean = score_array[~positive_array]
    compromised = score_array[positive_array]
    if clean.size == 0 or compromised.size == 0:
        raise GovernanceError(
            "detection at a false-alarm budget requires both compromised and clean samples"
        )
    if not 0.0 < budget < 1.0:
        raise GovernanceError("the false-alarm budget must lie within (0, 1)")
    threshold = float(np.quantile(clean, 1.0 - budget))
    return float(np.mean(compromised > threshold))


def roc_curve(scores: Sequence[float], positive: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_curve as sk_roc_curve

    false_positive, true_positive, _ = sk_roc_curve(
        np.asarray(positive, dtype=bool), np.asarray(scores, dtype=float)
    )
    return false_positive, true_positive


def certified_violation_rate(certified: Sequence[bool], violates: Sequence[bool]) -> float:
    certified_array = np.asarray(certified, dtype=bool)
    violates_array = np.asarray(violates, dtype=bool)
    if certified_array.shape != violates_array.shape:
        raise GovernanceError("certification and violation vectors must have identical shape")
    if certified_array.size == 0:
        raise GovernanceError("no execution was evaluated")
    return float(np.mean(certified_array & violates_array))
