from __future__ import annotations

from ..config import Config
from ..errors import ConfigurationError
from .a1_backdoor import TriggerBackdoorAttack, TriggerSpec
from .a2_label_flip import LabelFlipAttack
from .a3_substitution import SilentSubstitutionAttack
from .a4_adaptive import AttestationAwareAttack
from .base import Attack, CompromisedModel

ATTACK_FAMILIES = ("A1", "A2", "A3", "A4")


def build_attack(family: str, config: Config) -> Attack:
    if family == "A1":
        return TriggerBackdoorAttack(config)
    if family == "A2":
        return LabelFlipAttack(config)
    if family == "A3":
        return SilentSubstitutionAttack(config)
    if family == "A4":
        return AttestationAwareAttack(config)
    raise ConfigurationError(
        f"unknown attack family {family!r}; valid families are " + ", ".join(ATTACK_FAMILIES)
    )


__all__ = [
    "ATTACK_FAMILIES",
    "Attack",
    "AttestationAwareAttack",
    "CompromisedModel",
    "LabelFlipAttack",
    "SilentSubstitutionAttack",
    "TriggerBackdoorAttack",
    "TriggerSpec",
    "build_attack",
]
