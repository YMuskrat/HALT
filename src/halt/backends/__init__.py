from halt.backends.scripted import ScriptedBackend

__all__ = ["ScriptedBackend", "TransformersBackend", "ReplayBackend"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "TransformersBackend":
        from halt.backends.transformers import TransformersBackend
        return TransformersBackend
    if name == "ReplayBackend":
        from halt.backends.replay import ReplayBackend
        return ReplayBackend
    raise AttributeError(name)
