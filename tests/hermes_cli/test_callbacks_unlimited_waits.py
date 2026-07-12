"""Unlimited-wait semantics for the modular CLI prompt callbacks."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cli as cli_module
from hermes_cli import callbacks


class _OneEmptyThen:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def get(self, timeout=None):
        self.calls += 1
        if self.calls == 1:
            raise queue.Empty
        return self.value


def _cli_stub():
    return SimpleNamespace(
        _app=SimpleNamespace(invalidate=MagicMock()),
        _clarify_state=None,
        _clarify_freetext=False,
        _clarify_deadline=0,
        _approval_state=None,
        _approval_deadline=0,
        _approval_lock=threading.Lock(),
    )


def test_zero_clarify_timeout_waits_for_response():
    cli = _cli_stub()
    response_queue = _OneEmptyThen("B")

    with patch.dict(cli_module.CLI_CONFIG, {"clarify": {"timeout": 0}}), \
         patch.object(callbacks.queue, "Queue", return_value=response_queue):
        result = callbacks.clarify_callback(cli, "Pick one", ["A", "B"])

    assert result == "B"
    assert response_queue.calls == 2


def test_zero_approval_timeout_waits_for_response():
    cli = _cli_stub()
    response_queue = _OneEmptyThen("once")

    with patch.dict(cli_module.CLI_CONFIG, {"approvals": {"timeout": 0}}), \
         patch.object(callbacks.queue, "Queue", return_value=response_queue):
        result = callbacks.approval_callback(cli, "rm -rf /tmp/example", "danger")

    assert result == "once"
    assert response_queue.calls == 2
