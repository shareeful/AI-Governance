from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..config import Config, require_sum_to_one
from ..errors import ConfigurationError
from ..execution import RiskBand
from ..policy.clause import ClauseId


@dataclass(frozen=True)
class RiskAssessment:
    score: float
    band: RiskBand
    overrides: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "band": self.band.value,
            "overrides": list(self.overrides),
        }


class RiskScorer:
    def __init__(self, config: Config) -> None:
        raw = config.get("trust.risk.weights")
        if not isinstance(raw, Mapping):
            raise ConfigurationError("trust.risk.weights must be a mapping")
        weights = {str(k): float(v) for k, v in raw.items()}
        if "attestation" not in weights:
            raise ConfigurationError(
                "trust.risk.weights must include an 'attestation' weight (w_att)"
            )
        clause_names = {c.value for c in ClauseId}
        unknown = set(weights) - clause_names - {"attestation"}
        if unknown:
            raise ConfigurationError(
                "trust.risk.weights contains unknown entries: " + ", ".join(sorted(unknown))
            )
        missing = clause_names - set(weights)
        if missing:
            raise ConfigurationError(
                "trust.risk.weights must assign a weight to every clause; missing: "
                + ", ".join(sorted(missing))
            )
        tolerance = config.float_value("trust.risk.weight_sum_tolerance")
        require_sum_to_one(weights, tolerance, "trust.risk.weights")
        if any(value < 0.0 for value in weights.values()):
            raise ConfigurationError("trust.risk.weights must be non-negative")

        self.attestation_weight = weights.pop("attestation")
        self.clause_weights = {ClauseId(name): value for name, value in weights.items()}
        self.rho = config.require_unit_interval("trust.risk.rho")

    def score(
        self,
        measures: Mapping[ClauseId, float],
        not_attested: bool,
    ) -> float:
        missing = set(self.clause_weights) - set(measures)
        if missing:
            raise ConfigurationError(
                "the risk score requires a measure for every weighted clause; missing: "
                + ", ".join(sorted(c.value for c in missing))
            )
        total = self.attestation_weight * float(not_attested)
        for clause, weight in self.clause_weights.items():
            total += weight * (1.0 - float(measures[clause]))
        return total

    def assess(
        self,
        measures: Mapping[ClauseId, float],
        not_attested: bool,
        clause_rejected: bool,
        action_is_high_risk: bool,
    ) -> RiskAssessment:
        value = self.score(measures, not_attested)
        overrides: list[str] = []
        if clause_rejected:
            overrides.append("clause_rejection")
        if not_attested:
            overrides.append("not_attested")
        if action_is_high_risk:
            overrides.append("action_register_high_risk")
        band = RiskBand.HIGH if overrides or value >= self.rho else RiskBand.LOW
        return RiskAssessment(score=value, band=band, overrides=tuple(overrides))
