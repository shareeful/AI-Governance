from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ...config import Config
from ...errors import ConfigurationError, MissingResourceError


@dataclass(frozen=True)
class TabularSchema:
    target_column: str
    identifier_column: str
    group_column: str
    protected_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    feature_groups: Mapping[str, tuple[str, ...]]
    positive_class: str

    @classmethod
    def from_config(cls, config: Config, prefix: str) -> "TabularSchema":
        raw_groups = config.get(f"{prefix}.feature_groups")
        if not isinstance(raw_groups, Mapping) or not raw_groups:
            raise ConfigurationError(
                f"{prefix}.feature_groups must be a non-empty mapping of group name to columns"
            )
        groups = {
            str(name): tuple(str(c) for c in columns) for name, columns in raw_groups.items()
        }
        return cls(
            target_column=config.str_value(f"{prefix}.target_column"),
            identifier_column=config.str_value(f"{prefix}.identifier_column"),
            group_column=config.str_value(f"{prefix}.group_column"),
            protected_columns=tuple(str(c) for c in config.list_value(f"{prefix}.protected_columns")),
            numeric_columns=tuple(str(c) for c in config.list_value(f"{prefix}.numeric_columns")),
            categorical_columns=tuple(
                str(c) for c in config.list_value(f"{prefix}.categorical_columns")
            ),
            feature_groups=groups,
            positive_class=config.str_value(f"{prefix}.positive_class"),
        )

    def all_columns(self) -> tuple[str, ...]:
        return tuple(self.numeric_columns) + tuple(self.categorical_columns)


def load_table(config: Config, prefix: str, schema: TabularSchema) -> pd.DataFrame:
    path = config.path(f"{prefix}.path")
    separator = config.str_value(f"{prefix}.separator")
    if path.is_dir():
        raise ConfigurationError(f"{prefix}.path must point at a data file, not a directory")
    frame = pd.read_csv(path, sep=separator)
    required = set(schema.all_columns()) | {
        schema.target_column,
        schema.identifier_column,
        schema.group_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ConfigurationError(
            f"table at {path} lacks required columns: " + ", ".join(sorted(missing))
        )
    frame = frame.dropna(subset=[schema.target_column])
    if frame.empty:
        raise MissingResourceError(
            f"table at {path} holds no rows with a non-null target; the framework does not "
            "impute a target column"
        )
    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class TabularEncoder:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    categories: Mapping[str, tuple[str, ...]]
    numeric_means: Mapping[str, float]
    numeric_scales: Mapping[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, schema: TabularSchema) -> "TabularEncoder":
        categories = {
            column: tuple(sorted(str(v) for v in frame[column].dropna().unique()))
            for column in schema.categorical_columns
        }
        for column, values in categories.items():
            if not values:
                raise ConfigurationError(
                    f"categorical column '{column}' holds no observed values"
                )
        means = {column: float(frame[column].astype(float).mean()) for column in schema.numeric_columns}
        scales = {}
        for column in schema.numeric_columns:
            deviation = float(frame[column].astype(float).std(ddof=0))
            scales[column] = deviation if deviation > 0.0 else 1.0
        return cls(
            numeric_columns=schema.numeric_columns,
            categorical_columns=schema.categorical_columns,
            categories=categories,
            numeric_means=means,
            numeric_scales=scales,
        )

    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = list(self.numeric_columns)
        for column in self.categorical_columns:
            names.extend(f"{column}={value}" for value in self.categories[column])
        return tuple(names)

    def transform_row(self, row: Mapping[str, object]) -> np.ndarray:
        values: list[float] = []
        for column in self.numeric_columns:
            raw = row.get(column)
            numeric = self.numeric_means[column] if raw is None or pd.isna(raw) else float(raw)
            values.append((numeric - self.numeric_means[column]) / self.numeric_scales[column])
        for column in self.categorical_columns:
            observed = str(row.get(column))
            values.extend(1.0 if observed == value else 0.0 for value in self.categories[column])
        return np.asarray(values, dtype=np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.vstack([self.transform_row(row) for _, row in frame.iterrows()])

    def column_indices(self, columns: Sequence[str]) -> tuple[int, ...]:
        names = self.feature_names()
        selected: list[int] = []
        for index, name in enumerate(names):
            base = name.split("=", 1)[0]
            if base in columns:
                selected.append(index)
        if not selected:
            raise ConfigurationError(
                "no encoded feature corresponds to columns: " + ", ".join(columns)
            )
        return tuple(selected)

    def neutral_vector(self) -> np.ndarray:
        return np.zeros(len(self.feature_names()), dtype=np.float32)


def group_aware_partition(
    frame: pd.DataFrame,
    schema: TabularSchema,
    proportions: Mapping[str, float],
    rng: np.random.Generator,
    tolerance: float,
) -> dict[str, pd.DataFrame]:
    total = float(sum(proportions.values()))
    if abs(total - 1.0) > tolerance:
        raise ConfigurationError(
            f"partition proportions must sum to 1.0 within {tolerance}; got {total:.9f}"
        )
    groups = frame.groupby(schema.group_column).indices
    names = sorted(groups)
    rng.shuffle(names)
    assignments: dict[str, list[int]] = {name: [] for name in proportions}
    targets = {name: proportion * len(frame) for name, proportion in proportions.items()}
    counts = {name: 0 for name in proportions}
    for group_name in names:
        deficits = {name: targets[name] - counts[name] for name in proportions}
        chosen = max(deficits, key=lambda n: deficits[n])
        indices = list(groups[group_name])
        assignments[chosen].extend(indices)
        counts[chosen] += len(indices)
    partitions: dict[str, pd.DataFrame] = {}
    for name, indices in assignments.items():
        if not indices:
            raise ConfigurationError(
                f"partition '{name}' received no rows; adjust the configured proportions"
            )
        partitions[name] = frame.iloc[sorted(indices)].reset_index(drop=True)
    return partitions
