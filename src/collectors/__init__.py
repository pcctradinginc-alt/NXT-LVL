"""Data collectors — each exposes a `collect() -> dict` function.

Every collector is fault tolerant: network/parse errors are caught, logged as
a warning, and an empty or partial (but always JSON-serializable) result dict
is returned. The pipeline must never fail because a single free data source
is unavailable or rate-limited.
"""
