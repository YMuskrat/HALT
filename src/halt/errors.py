"""Explicit errors; missing research evidence is never a confidence signal."""


class HaltError(Exception):
    pass


class ConfigurationError(HaltError, ValueError):
    pass


class CapabilityError(HaltError):
    pass


class BudgetExceeded(HaltError):
    pass


class Cancelled(HaltError):
    pass


class ReplayUnavailable(HaltError):
    pass


class MethodError(HaltError):
    pass
