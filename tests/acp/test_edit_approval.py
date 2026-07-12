"""Tests for ACP pre-edit approval gating."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from acp_adapter.edit_approval import (
    EditProposal,
    build_acp_edit_tool_call,
    clear_edit_approval_requester,
    make_acp_edit_approval_requester,
    set_edit_approval_requester,
    should_auto_approve_edit,
)
from model_tools import handle_function_call


def teardown_function() -> None:
    clear_edit_approval_requester()


def test_acp_permission_tool_call_uses_edit_kind_and_diff_content():
    proposal = EditProposal(
        tool_name="write_file",
        path="demo.txt",
        old_text="old\n",
        new_text="new\n",
        arguments={"path": "demo.txt", "content": "new\n"},
    )

    tool_call = build_acp_edit_tool_call(proposal)

    assert tool_call.kind == "edit"
    assert tool_call.status == "pending"
    assert tool_call.rawInput == {"tool": "write_file", "arguments": proposal.arguments}
    assert len(tool_call.content) == 1
    diff = tool_call.content[0]
    assert diff.path == "demo.txt"
    assert diff.oldText == "old\n"
    assert diff.newText == "new\n"


def test_edit_approval_uses_global_unlimited_timeout():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    request_permission = AsyncMock(name="request_permission")
    future = MagicMock()
    future.result.return_value = SimpleNamespace(
        outcome=SimpleNamespace(outcome="selected", option_id="allow_once")
    )

    with patch(
        "hermes_cli.config.load_config",
        return_value={"approvals": {"timeout": 0}},
    ), patch(
        "agent.async_utils.safe_schedule_threadsafe",
        return_value=future,
    ) as schedule:
        requester = make_acp_edit_approval_requester(
            request_permission,
            loop,
            "session-1",
        )
        approved = requester(
            EditProposal(
                tool_name="write_file",
                path="demo.txt",
                old_text="old\n",
                new_text="new\n",
                arguments={},
            )
        )

    assert approved is True
    future.result.assert_called_once_with(timeout=0.25)
    schedule.call_args.args[0].close()


def test_interrupt_releases_unlimited_edit_approval():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    request_permission = AsyncMock(name="request_permission")
    future = MagicMock(spec=Future)
    future.result.side_effect = FutureTimeout()

    with patch(
        "agent.async_utils.safe_schedule_threadsafe",
        return_value=future,
    ) as schedule, patch("tools.interrupt.is_interrupted", return_value=True):
        requester = make_acp_edit_approval_requester(
            request_permission,
            loop,
            "session-1",
            timeout=0,
        )
        approved = requester(
            EditProposal(
                tool_name="write_file",
                path="demo.txt",
                old_text="old\n",
                new_text="new\n",
                arguments={},
            )
        )

    schedule.call_args.args[0].close()
    assert approved is False
    future.result.assert_not_called()
    future.cancel.assert_called_once()


def test_completed_edit_future_timeout_error_fails_closed_without_spinning():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    request_permission = AsyncMock(name="request_permission")
    future = Future()
    future.set_exception(FutureTimeout("ACP edit operation failed"))
    interrupted = threading.Event()

    with patch(
        "agent.async_utils.safe_schedule_threadsafe",
        return_value=future,
    ) as schedule, patch(
        "tools.interrupt.is_interrupted", side_effect=interrupted.is_set
    ):
        requester = make_acp_edit_approval_requester(
            request_permission, loop, "session-1", timeout=0
        )
        result = []
        thread = threading.Thread(
            target=lambda: result.append(
                requester(
                    EditProposal(
                        tool_name="write_file",
                        path="demo.txt",
                        old_text="old\n",
                        new_text="new\n",
                        arguments={},
                    )
                )
            ),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=0.2)
        was_alive = thread.is_alive()
        interrupted.set()
        thread.join(timeout=1)

    schedule.call_args.args[0].close()
    assert not was_alive, "completed TimeoutError future caused a busy loop"
    assert result == [False]


def test_write_file_rejection_does_not_mutate_existing_file(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")

    set_edit_approval_requester(lambda _proposal: False)

    result = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-reject",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "before\n"


def test_write_file_approval_mutates_and_request_includes_diff(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    proposals = []

    def approve(proposal):
        proposals.append(proposal)
        return True

    set_edit_approval_requester(approve)

    result = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-approve",
        )
    )

    assert result.get("bytes_written") == len("after\n")
    assert target.read_text(encoding="utf-8") == "after\n"
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.tool_name == "write_file"
    assert proposal.path == str(target)
    assert proposal.old_text == "before\n"
    assert proposal.new_text == "after\n"


def test_write_file_new_file_request_has_empty_old_text(tmp_path):
    target = tmp_path / "new.txt"
    proposals = []

    set_edit_approval_requester(lambda proposal: proposals.append(proposal) or True)

    result = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "created\n"},
            task_id="acp-edit-new-file",
        )
    )

    assert result.get("bytes_written") == len("created\n")
    assert target.read_text(encoding="utf-8") == "created\n"
    assert proposals[0].old_text is None
    assert proposals[0].new_text == "created\n"


def test_requester_exception_denies_and_does_not_mutate(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")

    def boom(_proposal):
        raise RuntimeError("zed disconnected")

    set_edit_approval_requester(boom)

    result = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-exception",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "before\n"


def test_patch_replace_rejection_does_not_mutate(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    set_edit_approval_requester(lambda _proposal: False)

    result = json.loads(
        handle_function_call(
            "patch",
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "beta\n",
                "new_string": "gamma\n",
            },
            task_id="acp-patch-reject",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_patch_v4a_rejection_does_not_mutate(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    set_edit_approval_requester(lambda _proposal: False)

    result = json.loads(
        handle_function_call(
            "patch",
            {
                "mode": "patch",
                "patch": (
                    "*** Begin Patch\n"
                    f"*** Update File: {target}\n"
                    "@@\n"
                    " alpha\n"
                    "-beta\n"
                    "+gamma\n"
                    "*** End Patch\n"
                ),
            },
            task_id="acp-patch-v4a-reject",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_patch_v4a_approval_request_includes_patch_targets(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    proposals = []

    set_edit_approval_requester(lambda proposal: proposals.append(proposal) or False)

    json.loads(
        handle_function_call(
            "patch",
            {
                "mode": "patch",
                "patch": (
                    "*** Begin Patch\n"
                    f"*** Update File: {target}\n"
                    "@@\n"
                    " alpha\n"
                    "-beta\n"
                    "+gamma\n"
                    "*** End Patch\n"
                ),
            },
            task_id="acp-patch-v4a-proposal",
        )
    )

    assert len(proposals) == 1
    assert proposals[0].tool_name == "patch"
    assert proposals[0].path == str(target)
    assert str(target) in proposals[0].new_text


def test_patch_replace_approval_request_includes_full_file_diff(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    proposals = []

    set_edit_approval_requester(lambda proposal: proposals.append(proposal) or True)

    result = json.loads(
        handle_function_call(
            "patch",
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "beta\n",
                "new_string": "gamma\n",
            },
            task_id="acp-patch-approve",
        )
    )

    assert result.get("success") is True
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert proposals[0].tool_name == "patch"
    assert proposals[0].old_text == "alpha\nbeta\n"
    assert proposals[0].new_text == "alpha\ngamma\n"


def test_workspace_auto_approval_allows_workspace_and_tmp_but_not_sensitive(tmp_path):
    workspace_file = tmp_path / "src.py"
    # Use tempfile.gettempdir() so this test exercises the same code path on
    # Linux (`/tmp`), macOS (`/private/var/folders/...`) and Windows
    # (`%LOCALAPPDATA%\Temp`). Before the fix this branch only worked on Linux.
    tmp_file = Path(tempfile.gettempdir()) / "hermes-acp-auto-approve-test.txt"
    env_file = tmp_path / ".env"

    assert should_auto_approve_edit(
        EditProposal("write_file", str(workspace_file), None, "x", {}),
        "workspace_session",
        str(tmp_path),
    )
    assert should_auto_approve_edit(
        EditProposal("write_file", str(tmp_file), None, "x", {}),
        "workspace_session",
        str(tmp_path),
    )
    assert not should_auto_approve_edit(
        EditProposal("write_file", str(env_file), None, "SECRET=x", {}),
        "session",
        str(tmp_path),
    )
