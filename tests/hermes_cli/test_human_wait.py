"""Shared human-response timeout policy."""

import math

import pytest

from hermes_cli.human_wait import normalize_human_timeout, wait_timeout


def test_positive_timeout_remains_finite():
    assert normalize_human_timeout(12.5, default=60) == 12.5
    assert wait_timeout(12.5, default=60) == 12.5


def test_exact_zero_is_unlimited():
    assert normalize_human_timeout(0, default=60) == 0
    assert wait_timeout("0", default=60) is None


@pytest.mark.parametrize("value", [-1, "-5", math.inf, -math.inf, math.nan, "bad", None])
def test_invalid_or_negative_timeout_uses_finite_fallback(value):
    assert normalize_human_timeout(value, default=60) == 60
    assert wait_timeout(value, default=60) == 60
