from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..execution import Input


@dataclass(frozen=True)
class CompromisedModel:
    family: str
    strength: float
    system: Any
    parameters: Mapping[str, Any]


class Attack(ABC):
    family: str

    @abstractmethod
    def compromise(self, bundle, config, seed: int, strength: float) -> CompromisedModel:
        raise NotImplementedError


def apply_patch(
    image: np.ndarray,
    size: int,
    row_offset: int,
    column_offset: int,
    value: float,
) -> np.ndarray:
    patched = np.array(image, copy=True)
    patched[:, row_offset : row_offset + size, column_offset : column_offset + size] = value
    return patched


def patched_input(request: Input, size: int, row: int, column: int, value: float) -> Input:
    from dataclasses import replace

    return replace(
        request, features=apply_patch(np.asarray(request.features), size, row, column, value)
    )
