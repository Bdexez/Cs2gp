"""cs2gp - analyse et prediction de matchs CS2 professionnels."""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Analyzer",
    "EloConfig",
    "Fixture",
    "Tipster",
    "audit_history",
    "walk_forward",
]


def __getattr__(name: str):
    if name == "Analyzer":
        from .predict import Analyzer
        return Analyzer
    if name == "EloConfig":
        from .ratings import EloConfig
        return EloConfig
    if name in ("Fixture", "Tipster"):
        from . import pronostic
        return getattr(pronostic, name)
    if name == "audit_history":
        from .audit import audit_history
        return audit_history
    if name == "walk_forward":
        from .backtest import walk_forward
        return walk_forward
    raise AttributeError(f"module 'cs2gp' has no attribute {name!r}")
