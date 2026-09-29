"""Explicit, lazy discovery of installed engine adapters (trusted Python plugins)."""

from importlib.metadata import entry_points

from jev_runtime.errors import JevError

GROUP = "jev_runtime.backends"
BUILTINS = ("sglang", "vllm")


def backend_inventory() -> list[dict]:
    rows = [{"name": name, "provider": "jev-runtime-core"} for name in BUILTINS]
    for entry in entry_points(group=GROUP):
        rows.append(
            {"name": entry.name, "provider": entry.dist.name if entry.dist else entry.value}
        )
    return sorted(rows, key=lambda row: (row["name"], row["provider"]))


def create_backend(settings, api_key):
    entries = [entry for entry in entry_points(group=GROUP) if entry.name == settings.backend]
    if len(entries) > 1 or (entries and settings.backend in BUILTINS):
        raise JevError("backend_conflict", "Multiple providers registered this backend", 409)
    if settings.backend == "sglang":
        from jev_runtime.backends.sglang import SGLangHTTP

        return SGLangHTTP(settings.engine_url, settings.model_id, api_key)
    if settings.backend == "vllm":
        from jev_runtime.backends.vllm import VLLMHTTP

        return VLLMHTTP(settings.engine_url, api_key)
    if not entries:
        raise JevError(
            "backend_not_installed",
            f"Install a {GROUP} plugin for backend {settings.backend!r}",
            503,
        )
    # Only the selected installed provider executes; config cannot import an
    # arbitrary module or replace a built-in provider silently.
    backend = entries[0].load()(settings=settings, api_key=api_key)
    if not all(
        callable(getattr(backend, method, None)) for method in ("probe", "score", "cancel", "close")
    ):
        raise JevError("backend_contract", "Provider does not implement EngineAdapter", 503)
    return backend
