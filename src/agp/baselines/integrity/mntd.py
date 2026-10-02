from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ...config import Config
from ...errors import MissingResourceError
from ...execution import Input
from .base import AccessLevel, DetectorProfile, IntegrityDetector


class MetaNeuralTrojanDetector(IntegrityDetector):
    profile = DetectorProfile(
        name="MNTD",
        access=(AccessLevel.WHITE_BOX, AccessLevel.SHADOW_MODELS),
        runtime="offline",
    )

    def __init__(self, config: Config) -> None:
        self.query_count = config.int_value("baselines.integrity.mntd.query_count")
        self.shadow_root = config.path("baselines.integrity.mntd.shadow_model_root")
        self.clean_subdirectory = config.str_value("baselines.integrity.mntd.clean_subdirectory")
        self.trojaned_subdirectory = config.str_value(
            "baselines.integrity.mntd.trojaned_subdirectory"
        )
        self.meta_epochs = config.int_value("baselines.integrity.mntd.meta_epochs")
        self._meta = None
        self._queries: np.ndarray | None = None

    def fit(self, reference_system, requests: Sequence[Input], rng: np.random.Generator) -> None:
        from sklearn.linear_model import LogisticRegression

        clean_paths = sorted((self.shadow_root / self.clean_subdirectory).glob("*.pt"))
        trojaned_paths = sorted((self.shadow_root / self.trojaned_subdirectory).glob("*.pt"))
        if not clean_paths or not trojaned_paths:
            raise MissingResourceError(
                f"MNTD requires a pool of shadow models beneath {self.shadow_root} with "
                f"'{self.clean_subdirectory}' and '{self.trojaned_subdirectory}' subdirectories; "
                "train the pool with 'agp shadow-pool' before running RQ2"
            )
        take = min(self.query_count, len(requests))
        indices = rng.choice(len(requests), take, replace=False)
        self._queries = np.stack(
            [np.asarray(requests[int(i)].features, dtype=np.float32) for i in indices]
        )

        features: list[np.ndarray] = []
        labels: list[int] = []
        for path in clean_paths:
            features.append(self._representation(reference_system, path))
            labels.append(0)
        for path in trojaned_paths:
            features.append(self._representation(reference_system, path))
            labels.append(1)
        self._meta = LogisticRegression(max_iter=self.meta_epochs).fit(
            np.vstack(features), np.asarray(labels)
        )

    def _representation(self, reference_system, checkpoint: Path) -> np.ndarray:
        import torch

        from ...systems.imaging.backbone import build_backbone

        network = build_backbone(reference_system.spec)
        state = torch.load(checkpoint, map_location=reference_system.spec.device)
        network.load_state_dict(state["model_state"])
        network.eval()
        from ...systems.imaging.backbone import softmax_scores
        from ...systems.imaging.dataset import normalise

        normalised = normalise(
            self._queries, reference_system.normalisation_mean, reference_system.normalisation_std
        )
        return softmax_scores(network, normalised, reference_system.spec.device).flatten()

    def score(self, system, requests: Sequence[Input], rng: np.random.Generator) -> np.ndarray:
        if self._meta is None or self._queries is None:
            raise MissingResourceError("MNTD must be fitted on the shadow-model pool first")
        from ...systems.imaging.backbone import softmax_scores
        from ...systems.imaging.dataset import normalise

        normalised = normalise(
            self._queries, system.normalisation_mean, system.normalisation_std
        )
        representation = softmax_scores(
            system.network, normalised, system.spec.device
        ).flatten()
        probability = float(self._meta.predict_proba(representation[None, :])[0, 1])
        return np.full(len(requests), probability, dtype=float)
