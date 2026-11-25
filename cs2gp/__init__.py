"""cs2gp - analyse et prediction de matchs CS2 professionnels."""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Analyzer",
    "EloConfig",
    "Fixture",
    "Tipster",
    "audit_fixtures",
    "audit_history",
    "confusion_summary",
    "devig",
    "diff_analyzers",
    "inspect_analyzer",
    "roc_auc",
    "sharpness",
    "walk_forward",
]

_EXPORTS = {
    "Analyzer": (".predict", "Analyzer"),
    "EloConfig": (".ratings", "EloConfig"),
    "Fixture": (".pronostic", "Fixture"),
    "Tipster": (".pronostic", "Tipster"),
    "audit_fixtures": (".audit", "audit_fixtures"),
    "audit_history": (".audit", "audit_history"),
    "confusion_summary": (".metrics", "confusion_summary"),
    "devig": (".odds", "devig"),
    "diff_analyzers": (".snapshot", "diff_analyzers"),
    "inspect_analyzer": (".snapshot", "inspect_analyzer"),
    "roc_auc": (".metrics", "roc_auc"),
    "sharpness": (".metrics", "sharpness"),
    "walk_forward": (".backtest", "walk_forward"),
}


def __getattr__(name: str):
    if name in _EXPORTS:
        from importlib import import_module

        mod_name, attr = _EXPORTS[name]
        return getattr(import_module(mod_name, __name__), attr)
    raise AttributeError(f"module 'cs2gp' has no attribute {name!r}")
