"""Malformed JSONL records must not take a thread's event reads offline.

``_iter_records`` promises to skip malformed or interrupted records. A line that
is valid JSON but not an event envelope (a scalar, array, or an object without an
integer ``seq``) is still a malformed record; the read paths must skip it instead
of raising and losing every event in the thread.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deerflow.runtime.events.store.jsonl import JsonlRunEventStore


def _run_path(tmp_path: Path) -> Path:
    return tmp_path / "threads/thread-1/runs/run-1.jsonl"


async def _seed_valid_event(store: JsonlRunEventStore) -> dict:
    return await store.put(
        thread_id="thread-1",
        run_id="run-1",
        event_type="llm.ai.response",
        category="message",
        content={"type": "ai", "id": "message-1", "content": "hello"},
    )


def _append_raw_lines(tmp_path: Path, lines: list[str]) -> None:
    path = _run_path(tmp_path)
    path.write_bytes(("\n".join([*lines, path.read_text(encoding="utf-8").strip()]) + "\n").encode("utf-8"))


@pytest.mark.anyio
async def test_non_object_json_lines_are_skipped_and_reads_survive(tmp_path: Path):
    store = JsonlRunEventStore(base_dir=tmp_path)
    saved = await _seed_valid_event(store)
    _append_raw_lines(tmp_path, ["null", "[]", '"scalar"', "123", "true"])

    reopened = JsonlRunEventStore(base_dir=tmp_path)

    assert await reopened.list_messages("thread-1") == [saved]
    assert await reopened.list_events("thread-1", "run-1") == [saved]
    assert await reopened.list_messages_by_run("thread-1", "run-1") == [saved]
    assert await reopened.count_messages("thread-1") == 1
    assert await reopened.get_message_seqs("thread-1", ["message:message-1"]) == {"message:message-1": saved["seq"]}
    assert await reopened.find_latest_ai_message_run_ids("thread-1", {"message-1"}) == {"message-1": "run-1"}

    following = await reopened.put(thread_id="thread-1", run_id="run-2", event_type="llm.ai.response", category="message", content="next")
    assert following["seq"] == saved["seq"] + 1


@pytest.mark.anyio
async def test_records_without_an_integer_seq_are_skipped(tmp_path: Path):
    store = JsonlRunEventStore(base_dir=tmp_path)
    saved = await _seed_valid_event(store)
    envelope = {"thread_id": "thread-1", "run_id": "run-1", "event_type": "llm.ai.response", "category": "message", "content": "x", "metadata": {}}
    _append_raw_lines(
        tmp_path,
        [
            json.dumps(envelope),
            json.dumps({**envelope, "seq": "42"}),
            json.dumps({**envelope, "seq": 42.5}),
            json.dumps({**envelope, "seq": None}),
        ],
    )

    reopened = JsonlRunEventStore(base_dir=tmp_path)

    assert await reopened.list_messages("thread-1") == [saved]
    assert await reopened.list_events("thread-1", "run-1") == [saved]
    assert await reopened.count_messages("thread-1") == 1

    following = await reopened.put(thread_id="thread-1", run_id="run-2", event_type="llm.ai.response", category="message", content="next")
    assert following["seq"] == saved["seq"] + 1
