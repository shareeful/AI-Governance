from __future__ import annotations


class GovernanceError(Exception):
    pass


class ConfigurationError(GovernanceError):
    pass


class MissingResourceError(ConfigurationError):
    pass


class CalibrationError(GovernanceError):
    pass


class NoAdmissibleThresholdError(CalibrationError):
    pass


class BindingError(GovernanceError):
    pass


class InterfaceError(GovernanceError):
    pass


class LedgerIntegrityError(GovernanceError):
    pass


class AttestationError(GovernanceError):
    pass


class BenchmarkError(GovernanceError):
    pass
