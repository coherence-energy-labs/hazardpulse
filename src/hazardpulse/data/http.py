"""Small HTTP helpers with on-disk caching for reproducible data pulls."""

from __future__ import annotations

import hashlib
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_USER_AGENT = "hazardpulse/0.1 (+https://github.com/coherence-energy-labs/hazardpulse)"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = PROJECT_ROOT / ".cache" / "http"


def _cache_path(url: str, namespace: str) -> Path:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    return CACHE_ROOT / namespace / digest


def fetch_bytes(
    url: str,
    *,
    timeout: int = 60,
    namespace: str = "default",
    use_cache: bool = True,
    refresh: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> bytes:
    """Fetch bytes from a URL with simple file caching."""

    cache_path = _cache_path(url, namespace)
    if use_cache and cache_path.exists() and not refresh:
        return cache_path.read_bytes()

    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as resp:
                data = resp.read()
            break
        except urllib.error.HTTPError as exc:
            # A 4xx answer is deterministic -- the object is missing or
            # forbidden, and asking again returns the same answer. Retrying it
            # (with back-off) only multiplied the cost of every miss by ~4x.
            # 408 and 429 are the transient exceptions.
            last_exc = exc
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                raise
            if attempt < 2:
                time.sleep(2 ** attempt)
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    else:
        raise last_exc  # type: ignore[misc]

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    return data


def fetch_text(
    url: str,
    *,
    timeout: int = 60,
    namespace: str = "default",
    encoding: str = "utf-8",
    errors: str = "replace",
    use_cache: bool = True,
    refresh: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    """Fetch text from a URL with simple file caching."""

    data = fetch_bytes(
        url,
        timeout=timeout,
        namespace=namespace,
        use_cache=use_cache,
        refresh=refresh,
        user_agent=user_agent,
    )
    return data.decode(encoding, errors=errors)
