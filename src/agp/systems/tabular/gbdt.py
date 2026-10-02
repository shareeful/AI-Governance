from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ...config import Config
from ...errors import ConfigurationError, MissingResourceError


@dataclass(frozen=True)
class GradientBoostingSpec:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    early_stopping_rounds: int
    num_classes: int

    @classmethod
    def from_config(cls, config: Config, prefix: str) -> "GradientBoostingSpec":
        return cls(
            n_estimators=config.int_value(f"{prefix}.n_estimators"),
            max_depth=config.int_value(f"{prefix}.max_depth"),
            learning_rate=config.float_value(f"{prefix}.learning_rate"),
            subsample=config.require_unit_interval(f"{prefix}.subsample"),
            colsample_bytree=config.require_unit_interval(f"{prefix}.colsample_bytree"),
            reg_lambda=config.float_value(f"{prefix}.reg_lambda"),
            early_stopping_rounds=config.int_value(f"{prefix}.early_stopping_rounds"),
            num_classes=config.int_value(f"{prefix}.num_classes"),
        )


def build_model(spec: GradientBoostingSpec, seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=spec.n_estimators,
        max_depth=spec.max_depth,
        learning_rate=spec.learning_rate,
        subsample=spec.subsample,
        colsample_bytree=spec.colsample_bytree,
        reg_lambda=spec.reg_lambda,
        objective="binary:logistic" if spec.num_classes == 2 else "multi:softprob",
        num_class=None if spec.num_classes == 2 else spec.num_classes,
        random_state=seed,
        early_stopping_rounds=spec.early_stopping_rounds,
        eval_metric="logloss" if spec.num_classes == 2 else "mlogloss",
        tree_method="hist",
    )


def train_model(
    spec: GradientBoostingSpec,
    features: np.ndarray,
    labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    seed: int,
    checkpoint: Path,
):
    if features.shape[0] == 0:
        raise MissingResourceError("the training split holds no rows")
    observed = np.unique(labels)
    if observed.size < 2:
        raise ConfigurationError(
            "the training split contains a single class; a decision model cannot be fitted"
        )
    model = build_model(spec, seed)
    model.fit(
        features,
        labels,
        eval_set=[(validation_features, validation_labels)],
        verbose=False,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(checkpoint))
    return model


def load_model(spec: GradientBoostingSpec, checkpoint: Path, seed: int):
    if not checkpoint.is_file():
        raise MissingResourceError(
            f"no trained tabular model at {checkpoint}; run 'agp train' before certification"
        )
    model = build_model(spec, seed)
    model.load_model(str(checkpoint))
    return model


def shap_values(model, features: np.ndarray, background: np.ndarray) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    values = explainer.shap_values(features, check_additivity=False)
    if isinstance(values, list):
        stacked = np.stack(values, axis=0)
        return np.abs(stacked).sum(axis=0)
    array = np.asarray(values)
    if array.ndim == 3:
        return np.abs(array).sum(axis=2)
    return np.abs(array)


def group_masses(
    attributions: np.ndarray,
    group_indices: Sequence[Sequence[int]],
) -> np.ndarray:
    return np.array(
        [float(np.abs(attributions[list(indices)]).sum()) for indices in group_indices],
        dtype=float,
    )
