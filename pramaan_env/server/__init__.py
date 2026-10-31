from pramaan_env.models import PramaanAction, PramaanObservation, PramaanState

__all__ = ["PramaanAction", "PramaanObservation", "PramaanState", "app", "main"]


def __getattr__(name: str):
    if name in {"app", "main"}:
        from pramaan_env.server.app import app, main

        return {"app": app, "main": main}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
