from __future__ import annotations

from . import ablation, rq1, rq2, rq3, rq4, rq5
from .base import ExperimentResult, artifact_root, seeds_from, systems_from

EXPERIMENTS = {
    "rq1": rq1.run,
    "rq2": rq2.run,
    "rq2_sensitivity": rq2.run_sensitivity,
    "rq3": rq3.run,
    "rq4": rq4.run,
    "rq5": rq5.run,
    "ablation": ablation.run,
}

__all__ = [
    "EXPERIMENTS",
    "ExperimentResult",
    "ablation",
    "artifact_root",
    "rq1",
    "rq2",
    "rq3",
    "rq4",
    "rq5",
    "seeds_from",
    "systems_from",
]
