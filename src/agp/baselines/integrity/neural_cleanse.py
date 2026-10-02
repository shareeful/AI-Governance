from __future__ import annotations

from typing import Sequence

import numpy as np

from ...config import Config
from ...errors import ConfigurationError
from ...execution import Input
from .base import AccessLevel, DetectorProfile, IntegrityDetector


class NeuralCleanseDetector(IntegrityDetector):
    profile = DetectorProfile(
        name="Neural Cleanse", access=(AccessLevel.WHITE_BOX,), runtime="offline"
    )

    def __init__(self, config: Config) -> None:
        self.steps = config.int_value("baselines.integrity.neural_cleanse.optimisation_steps")
        self.learning_rate = config.float_value("baselines.integrity.neural_cleanse.learning_rate")
        self.mask_weight = config.float_value("baselines.integrity.neural_cleanse.mask_weight")
        self.batch_size = config.int_value("baselines.integrity.neural_cleanse.batch_size")
        self._reference_index: float | None = None

    def fit(self, reference_system, requests: Sequence[Input], rng: np.random.Generator) -> None:
        self._reference_index = self._anomaly_index(reference_system, requests)

    def score(self, system, requests: Sequence[Input], rng: np.random.Generator) -> np.ndarray:
        if self._reference_index is None:
            raise ConfigurationError("Neural Cleanse must be fitted on the certified model first")
        index = self._anomaly_index(system, requests)
        return np.full(len(requests), index - self._reference_index, dtype=float)

    def _anomaly_index(self, system, requests: Sequence[Input]) -> float:
        norms = [
            self._reverse_engineer(system, requests, target)
            for target in range(len(system.classes))
        ]
        values = np.asarray(norms, dtype=float)
        median = float(np.median(values))
        deviation = float(np.median(np.abs(values - median)))
        if deviation == 0.0:
            return 0.0
        return float(np.max((median - values) / (1.4826 * deviation)))

    def _reverse_engineer(self, system, requests: Sequence[Input], target: int) -> float:
        import torch

        network = getattr(system, "network", None)
        if network is None:
            raise ConfigurationError(
                "Neural Cleanse requires white-box access to the governed network"
            )
        device = system.spec.device
        batch = np.stack(
            [np.asarray(r.features, dtype=np.float32) for r in requests[: self.batch_size]]
        )
        tensor = torch.from_numpy(batch).float().to(device)
        mask = torch.zeros(
            (1, 1, tensor.shape[2], tensor.shape[3]), device=device, requires_grad=True
        )
        pattern = torch.zeros_like(tensor[:1], requires_grad=True)
        optimiser = torch.optim.Adam([mask, pattern], lr=self.learning_rate)
        targets = torch.full((tensor.shape[0],), target, dtype=torch.long, device=device)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(self.steps):
            optimiser.zero_grad()
            bounded_mask = torch.sigmoid(mask)
            bounded_pattern = torch.sigmoid(pattern)
            stamped = (1.0 - bounded_mask) * tensor + bounded_mask * bounded_pattern
            normalised = self._normalise(system, stamped)
            loss = criterion(network(normalised), targets) + self.mask_weight * bounded_mask.abs().sum()
            loss.backward()
            optimiser.step()
        return float(torch.sigmoid(mask).abs().sum().item())

    def _normalise(self, system, tensor):
        import torch

        mean = torch.tensor(system.normalisation_mean, device=tensor.device).view(1, -1, 1, 1)
        std = torch.tensor(system.normalisation_std, device=tensor.device).view(1, -1, 1, 1)
        return (tensor - mean) / std
