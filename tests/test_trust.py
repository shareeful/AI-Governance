from __future__ import annotations

import json

import pytest
import yaml

from agp.config import Config
from agp.errors import ConfigurationError, LedgerIntegrityError
from agp.execution import RiskBand
from agp.guards.oversight import missed_referral_rate
from agp.policy.clause import ClauseId
from agp.trust.ledger import AppendOnlyLedger, generate_signing_key
from agp.trust.recertify import RecertificationCandidate, screen_candidates
from agp.trust.risk import RiskScorer

WEIGHTS = {
    "attestation": 0.30,
    "non_discrimination": 0.16,
    "data_minimisation": 0.14,
    "reasoning_faithfulness": 0.14,
    "precondition_compliance": 0.13,
    "factual_groundedness": 0.13,
}


def _config(tmp_path, **overrides):
    payload = {
        "trust": {
            "risk": {
                "weights": dict(WEIGHTS, **overrides),
                "weight_sum_tolerance": 1.0e-9,
                "rho": 0.5,
            }
        }
    }
    path = tmp_path / "risk.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return Config.load(path)


def _compliant() -> dict[ClauseId, float]:
    return {clause: 1.0 for clause in ClauseId}


def test_risk_weights_must_sum_to_one(tmp_path):
    with pytest.raises(ConfigurationError):
        RiskScorer(_config(tmp_path, attestation=0.50))


def test_fully_compliant_attested_execution_scores_zero(tmp_path):
    scorer = RiskScorer(_config(tmp_path))
    assert scorer.score(_compliant(), not_attested=False) == pytest.approx(0.0)


def test_not_attested_execution_carries_the_attestation_weight(tmp_path):
    scorer = RiskScorer(_config(tmp_path))
    assert scorer.score(_compliant(), not_attested=True) == pytest.approx(WEIGHTS["attestation"])


def test_any_clause_rejection_overrides_the_band(tmp_path):
    scorer = RiskScorer(_config(tmp_path))
    assessment = scorer.assess(
        _compliant(), not_attested=False, clause_rejected=True, action_is_high_risk=False
    )
    assert assessment.band is RiskBand.HIGH
    assert "clause_rejection" in assessment.overrides


def test_high_risk_action_overrides_the_band(tmp_path):
    scorer = RiskScorer(_config(tmp_path))
    assessment = scorer.assess(
        _compliant(), not_attested=False, clause_rejected=False, action_is_high_risk=True
    )
    assert assessment.band is RiskBand.HIGH


def test_low_band_requires_the_score_below_rho(tmp_path):
    scorer = RiskScorer(_config(tmp_path))
    assessment = scorer.assess(
        _compliant(), not_attested=False, clause_rejected=False, action_is_high_risk=False
    )
    assert assessment.band is RiskBand.LOW
    assert assessment.overrides == ()


def test_ledger_chains_and_verifies(tmp_path):
    private, public = generate_signing_key(tmp_path / "k.pem", tmp_path / "k.pub.pem")
    ledger = AppendOnlyLedger(tmp_path / "log.jsonl", private, public)
    first = ledger.append({"exec_id": "a", "verdict": "admit"})
    second = ledger.append({"exec_id": "b", "verdict": "withhold"})
    assert second.previous_hash == first.content_hash()
    assert ledger.verify() == 2


def test_ledger_detects_a_tampered_record(tmp_path):
    private, public = generate_signing_key(tmp_path / "k.pem", tmp_path / "k.pub.pem")
    path = tmp_path / "log.jsonl"
    ledger = AppendOnlyLedger(path, private, public)
    ledger.append({"exec_id": "a", "verdict": "admit"})
    ledger.append({"exec_id": "b", "verdict": "admit"})

    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["verdict"] = "withhold"
    lines[0] = json.dumps(first, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reopened = AppendOnlyLedger(path, private, public)
    with pytest.raises(LedgerIntegrityError):
        reopened.verify()


def test_ledger_survives_reopening(tmp_path):
    private, public = generate_signing_key(tmp_path / "k.pem", tmp_path / "k.pub.pem")
    path = tmp_path / "log.jsonl"
    AppendOnlyLedger(path, private, public).append({"exec_id": "a", "verdict": "admit"})
    reopened = AppendOnlyLedger(path, private, public)
    reopened.append({"exec_id": "b", "verdict": "admit"})
    assert reopened.verify() == 2


def test_screening_excludes_withheld_and_unattested_evidence(tmp_path):
    private, public = generate_signing_key(tmp_path / "k.pem", tmp_path / "k.pub.pem")
    ledger = AppendOnlyLedger(tmp_path / "log.jsonl", private, public)
    admitted = ledger.append({"exec_id": "a", "verdict": "admit"})
    withheld = ledger.append({"exec_id": "b", "verdict": "withhold"})
    unattested = ledger.append({"exec_id": "c", "verdict": "admit"})

    outcome = screen_candidates(
        [
            RecertificationCandidate(admitted, admitted=True, attested=True),
            RecertificationCandidate(withheld, admitted=False, attested=True),
            RecertificationCandidate(unattested, admitted=True, attested=False),
        ],
        contamination_hook=lambda record: (True, ""),
    )
    assert [r.identifier for r in outcome.approved] == ["a"]
    assert {reason for _, reason in outcome.rejected} == {
        "withheld_execution",
        "not_attested_execution",
    }


def test_missed_referral_rate_counts_unreferred_high_band_executions():
    assert missed_referral_rate([True, False, True], [True, True, False]) == pytest.approx(0.5)


def test_missed_referral_rate_is_undefined_without_high_band_executions():
    with pytest.raises(ValueError):
        missed_referral_rate([False, False], [False, False])
