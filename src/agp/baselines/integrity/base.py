from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...execution import Input


class AccessLevel(str):
    BLACK_BOX = "BB"
    WHITE_BOX = "WB"
    SHADOW_MODELS = "SM"
    OFFLINE_BATCH = "OB"


@dataclass(frozen=True)
class DetectorProfile:
    name: str
    access: tuple[str, ...]
    runtime: str


class IntegrityDetector(ABC):
    profile: DetectorProfile

    @abstractmethod
    def fit(self, reference_system, requests: Sequence[Input], rng: np.random.Generator) -> None:
        raise NotImplementedError

    @abstractmethod
    def score(self, system, requests: Sequence[Input], rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


def hidden_representations(system, requests: Sequence[Input]) -> np.ndarray:
    import torch

    network = getattr(system, "network", None)
    if network is None:
        raise AttributeError(
            f"system '{system.name}' exposes no network; this detector requires white-box access"
        )
    features: list[np.ndarray] = []
    handle_output: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        handle_output["value"] = output.detach()

    target = network.classifier if hasattr(network, "classifier") else network.fc
    penultimate = target[0] if isinstance(target, torch.nn.Sequential) else target
    handle = penultimate.register_forward_hook(hook)
    try:
        for request in requests:
            system.decision_scores(request)
            value = handle_output["value"]
            features.append(value.flatten().cpu().numpy())
    finally:
        handle.remove()
    return np.vstack(features)
