from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ...checkers.scope import ActionRegister
from ...config import Config
from ...errors import ConfigurationError, InterfaceError
from ...execution import Input
from ..base import InstrumentedSystem, RationaleTemplates
from .dataset import TabularEncoder, TabularSchema
from .gbdt import GradientBoostingSpec, group_masses, shap_values


class TabularDecisionSystem(InstrumentedSystem):
    def __init__(
        self,
        name: str,
        config: Config,
        prefix: str,
        schema: TabularSchema,
        encoder: TabularEncoder,
        model,
        background: np.ndarray,
        action_register: ActionRegister,
        attribute_domains: Mapping[str, Sequence[Any]],
        class_names: tuple[str, ...],
    ) -> None:
        self.schema = schema
        self.encoder = encoder
        self.model = model
        self.background = background
        self.classes = class_names
        self.spec = GradientBoostingSpec.from_config(config, f"{prefix}.model")
        if len(self.classes) != self.spec.num_classes:
            raise ConfigurationError(
                f"{prefix}.model.class_names lists {len(self.classes)} names but num_classes is "
                f"{self.spec.num_classes}"
            )
        groups = tuple(schema.feature_groups)
        self.group_indices = tuple(
            encoder.column_indices(schema.feature_groups[group]) for group in groups
        )
        self.paraphrase_numeric_jitter = config.float_value(
            f"{prefix}.paraphrase.numeric_relative_jitter"
        )
        self.perturbation_mix = config.require_unit_interval(f"{prefix}.perturbation.mix_weight")
        self.precondition_fields = tuple(
            str(field) for field in config.list_value(f"{prefix}.preconditions.evidence_fields")
        )
        super().__init__(
            name=name,
            modality="tabular",
            action_register=action_register,
            feature_groups=groups,
            protected_attributes=tuple(config.list_value(f"{prefix}.protected_attributes")),
            attribute_domains=attribute_domains,
            minimum_necessary=frozenset(
                str(item) for item in config.list_value(f"{prefix}.minimum_necessary_personal_data")
            ),
            templates=RationaleTemplates.from_config(config, f"{prefix}.templates"),
            reason_count=config.int_value(f"{prefix}.reason_count"),
            recorded_check_policy=config.str_value(f"{prefix}.preconditions.recorded_check_policy"),
        )

    def request_from_row(self, row: Mapping[str, Any]) -> Input:
        metadata = {str(k): v for k, v in row.items()}
        metadata.setdefault("provenance", f"dataset:{self.name}")
        return Input(
            identifier=str(row[self.schema.identifier_column]),
            request="; ".join(
                f"{key}={value}"
                for key, value in sorted(metadata.items())
                if key != "provenance"
            ),
            features=self.encoder.transform_row(row),
            evidence=(),
            metadata=metadata,
            requested_action=self._requested_action(),
        )

    def _requested_action(self) -> str:
        actions = list(self.action_register)
        if not actions:
            raise ConfigurationError(f"system '{self.name}' has an empty action register")
        return actions[0]

    def _vector(self, request: Input) -> np.ndarray:
        vector = np.asarray(request.features, dtype=np.float32)
        if vector.ndim != 1:
            raise InterfaceError(
                f"system '{self.name}' expects a one-dimensional encoded row; got shape "
                f"{vector.shape}"
            )
        return vector

    def decision_scores(self, request: Input) -> np.ndarray:
        probabilities = self.model.predict_proba(self._vector(request)[None, :])
        return np.asarray(probabilities, dtype=float)[0]

    def predict_action(self, request: Input) -> str:
        return self.classes[int(np.argmax(self.decision_scores(request)))]

    def predict_action_with_groups_removed(
        self, request: Input, removed_groups: frozenset[str]
    ) -> str:
        if not removed_groups:
            return self.predict_action(request)
        vector = np.array(self._vector(request), copy=True)
        index_by_name = {name: position for position, name in enumerate(self.feature_groups)}
        neutral = self.background.mean(axis=0)
        for group in removed_groups:
            if group not in index_by_name:
                raise InterfaceError(f"unknown feature group '{group}' for system '{self.name}'")
            for column in self.group_indices[index_by_name[group]]:
                vector[column] = neutral[column]
        return self.predict_action(replace(request, features=vector))

    def attribution(self, request: Input) -> Mapping[str, float]:
        values = shap_values(self.model, self._vector(request)[None, :], self.background)[0]
        masses = group_masses(values, self.group_indices)
        return {name: float(mass) for name, mass in zip(self.feature_groups, masses)}

    def substitute_attribute(self, request: Input, attribute: str, value: Any) -> Input:
        if attribute not in self.protected_attributes:
            raise InterfaceError(
                f"'{attribute}' is not a documented protected attribute of system '{self.name}'"
            )
        row = dict(request.metadata)
        row[attribute] = value
        substituted = self.encoder.transform_row(row)
        return replace(
            request,
            features=substituted,
            metadata=row,
            request="; ".join(
                f"{key}={val}" for key, val in sorted(row.items()) if key != "provenance"
            ),
        )

    def perturbations(self, request: Input, count: int, rng: np.random.Generator) -> list[Input]:
        if count < 1:
            raise InterfaceError("the STRIP signal requires at least one perturbation")
        if self.background.shape[0] == 0:
            raise InterfaceError(
                f"system '{self.name}' holds an empty superimposition pool; the STRIP signal "
                "requires real records drawn from the calibration split"
            )
        vector = self._vector(request)
        outputs: list[Input] = []
        indices = rng.integers(0, self.background.shape[0], size=count)
        for index in indices:
            donor = self.background[int(index)]
            blended = (
                (1.0 - self.perturbation_mix) * vector + self.perturbation_mix * donor
            ).astype(np.float32)
            outputs.append(replace(request, features=blended))
        return outputs

    def paraphrases(self, request: Input, count: int, rng: np.random.Generator) -> list[Input]:
        if count < 1:
            raise InterfaceError("the paraphrase signal requires at least one variant")
        numeric_indices = self.encoder.column_indices(self.schema.numeric_columns)
        vector = self._vector(request)
        outputs: list[Input] = []
        for _ in range(count):
            variant = np.array(vector, copy=True)
            for index in numeric_indices:
                scale = 1.0 + float(
                    rng.uniform(-self.paraphrase_numeric_jitter, self.paraphrase_numeric_jitter)
                )
                variant[index] = variant[index] * scale
            outputs.append(replace(request, features=variant.astype(np.float32)))
        return outputs

    def recorded_checks(self, request: Input, action: str) -> tuple[str, ...]:
        required = sorted(self.action_register.required_checks(action))
        if self.recorded_check_policy == "complete":
            return tuple(required)
        satisfied: list[str] = []
        for check in required:
            matched = [field for field in self.precondition_fields if check.endswith(field)]
            if not matched:
                satisfied.append(check)
                continue
            if all(
                str(request.metadata.get(field, "")).strip() not in {"", "nan", "None"}
                for field in matched
            ):
                satisfied.append(check)
        return tuple(satisfied)


def background_matrix(frame: pd.DataFrame, encoder: TabularEncoder, size: int, rng) -> np.ndarray:
    if len(frame) == 0:
        raise ConfigurationError("the background split holds no rows")
    take = min(size, len(frame))
    indices = rng.choice(len(frame), size=take, replace=False)
    return encoder.transform(frame.iloc[sorted(int(i) for i in indices)])
