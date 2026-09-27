"""Process-wide HTTP connection pool for synchronous provider requests."""

from threading import Lock

import httpx


_client: httpx.Client | None = None
_client_lock = Lock()


def get_http_client() -> httpx.Client:
    """Return a thread-safe shared client with keep-alive connections enabled."""
    global _client
    if _client is not None and not _client.is_closed:
        return _client

    with _client_lock:
        if _client is None or _client.is_closed:
            _client = httpx.Client(
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                )
            )
        return _client


def close_http_client() -> None:
    """Close the shared pool during application shutdown."""
    global _client
    with _client_lock:
        if _client is not None and not _client.is_closed:
            _client.close()
        _client = None
