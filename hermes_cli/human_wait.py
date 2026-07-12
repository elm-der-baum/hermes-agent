"""Shared timeout semantics for prompts that require a human response.

Positive values are finite seconds. Exactly zero disables automatic expiry.
Negative, malformed, or non-finite values fall back to a finite caller-supplied
default so configuration mistakes cannot silently disable safety timeouts.
"""

from __future__ import annotations

import math
from typing import Any


def _finite_default(default: Any) -> float:
    try:
        value = float(str(default))
    except (TypeError, ValueError, OverflowError):
        value = 60.0
    if not math.isfinite(value) or value <= 0:
        return 60.0
    return value


def normalize_human_timeout(value: Any, *, default: Any) -> float:
    """Return finite seconds or exactly ``0.0`` for unlimited waiting."""
    fallback = _finite_default(default)
    try:
        timeout = float(str(value))
    except (TypeError, ValueError, OverflowError):
        return fallback
    if not math.isfinite(timeout) or timeout < 0:
        return fallback
    return timeout


def wait_timeout(value: Any, *, default: Any) -> float | None:
    """Return a blocking-API timeout where ``None`` means no deadline."""
    timeout = normalize_human_timeout(value, default=default)
    return None if timeout == 0 else timeout
