#!/usr/bin/env python3
"""Cancel-time gateway history reconciliation (#7977 rework 2).

Drives REAL routes and REAL streaming functions with a fake /v1/runs gateway
that captures request bodies and applies the installed gateway's rules
(gateway/platforms/api_server_runs.py ~712-714 semantics):
  - a captured /v1/runs body WITH conversation_history uses exactly those rows
    (the gateway marks caller-supplied rows already-saved, _db_flush_collect);
  - a body WITHOUT conversation_history makes the gateway load its own stored
    transcript (which includes the user prompt + persisted incomplete snapshot
    it adopted when a run was cancelled).

Every scenario asserts on the CAPTURED REQUEST BODY's conversation_history,
not on WebUI state.

Scenarios:
  1. fork turn 1 -> history == parent prefix (master parity; fork question
     travels as `input`).
  2. [ROUND-2 KILLER] fork turn 1 cancelled with a streamed partial -> fork
     turn 2 -> history == parent prefix + [F-q1, partial]. RED on master:
     the interrupted turn is missing from the captured history.
  3. normal user Stop mid-run on a plain session -> next turn's history
     includes the reconciled user row + partial. RED on master (timing:
     nothing reconciles the model-facing context on Stop).
  4. truncate after an interrupted run -> history is the SHORTENED context
     (user edit wins).
  5. prefill configured + interrupted history -> captured history contains
     the context rows AND the prefill rows (nothing dropped).
  6. double-settle (same turn settled twice) -> assistant partial appended
     once. RED on master: the settle path never appends the partial at all.

Red/green evidence: on the un-fixed master state scenarios 2, 3, 5 and 6
fail (5 composes with the interrupted history) while 1 and 4 pass; with the
reconciliation in place all six pass.

Runs under pytest (conftest provisions isolated HERMES_HOME /
HERMES_WEBUI_STATE_DIR) and standalone:

  HERMES_HOME=/tmp/hermes-webui-agent-home \
  HERMES_WEBUI_STATE_DIR=/tmp/hermes-webui-agent-state \
  python3 tests/test_gateway_cancel_reconciliation.py
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch
from urllib.parse import urlparse

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Isolated trial state before importing api modules (AGENTS.md safety rule).
os.environ.setdefault("HERMES_HOME", "/tmp/hermes-webui-agent-home")
os.environ.setdefault("HERMES_WEBUI_STATE_DIR", "/tmp/hermes-webui-agent-state")
for _trial_dir in (
    os.environ["HERMES_WEBUI_STATE_DIR"],
    os.path.join(os.environ["HERMES_WEBUI_STATE_DIR"], "sessions"),
    os.environ["HERMES_HOME"],
):
    pathlib.Path(_trial_dir).mkdir(parents=True, exist_ok=True)

import api.config as cfg  # noqa: E402
import api.routes as routes  # noqa: E402
from api.models import Session, get_session  # noqa: E402

GATEWAY_STORE: dict[str, list] = {}  # session_id -> gateway's own stored transcript
CAPTURED_BODIES: list[dict] = []  # every /v1/runs admission body, in order


class _JsonResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self, _limit=None):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class _SseResponse:
    """SSE stream for a run: completed by default, or cancelled when armed."""

    cancelled_runs: set[str] = set()

    def __init__(self, run_id: str):
        self._run_id = run_id

    def __iter__(self):
        if self._run_id in _SseResponse.cancelled_runs:
            # Gateway-side cancel: persist the incomplete snapshot into the
            # gateway's own store (adopt/persist path) before reporting. The
            # worker resets STREAM_PARTIAL_TEXT at startup, so the partial it
            # reconciles is whatever THIS stream relayed before the terminal
            # event — stream a delta first, exactly like a real interrupted run.
            body = _last_admitted_body()
            sid = str(body.get("session_id") or "")
            GATEWAY_STORE.setdefault(sid, []).extend(_gateway_adopted_rows(body))
            _SseResponse.cancelled_runs.discard(self._run_id)
            yield b'data: {"event":"message.delta","delta":"gateway cut"}\n\n'
            yield b'data: {"event":"run.cancelled"}\n\n'
        else:
            yield b'data: {"event":"run.completed","output":"turn answer","usage":{"input_tokens":1,"output_tokens":1}}\n\n'
        yield b"data: [DONE]\n\n"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _last_admitted_body() -> dict:
    return CAPTURED_BODIES[-1] if CAPTURED_BODIES else {}


def _gateway_adopted_rows(body: dict) -> list:
    """Rows the gateway persists for an interrupted run: prompt + partial."""
    prompt = body.get("input")
    if isinstance(prompt, list):
        prompt = prompt[-1].get("content") if prompt else ""
    return [
        {"role": "user", "content": str(prompt or "")},
        {"role": "assistant", "content": "gateway-persisted partial"},
    ]


def fake_urlopen(req, *, timeout=None):
    url = req.full_url
    if url.endswith("/v1/runs"):
        body = json.loads(req.data.decode("utf-8"))
        CAPTURED_BODIES.append(body)
        return _JsonResponse({"run_id": f"run-{len(CAPTURED_BODIES)}"})
    if "/events" in url:
        # .../v1/runs/<run_id>/events — the run id sits BEFORE /events, not at
        # the tail (the old rsplit parsed the literal "events" and the
        # cancelled_runs arming could never match).
        run_id = url.split("/v1/runs/", 1)[1].split("/events", 1)[0]
        return _SseResponse(run_id)
    raise AssertionError(f"unexpected gateway request: {url}")


class _FakeQueue:
    def __init__(self):
        self.events = []

    def put_nowait(self, item):
        self.events.append(item)

    def note_last_event_id(self, event_id):
        return None


class _FakeHandler:
    """Minimal handler for routes.handle_post (pattern from tests/test_465)."""

    def __init__(self, body: dict):
        self.status = None
        self.sent_headers = []
        self.body = bytearray()
        self.wfile = self
        self.rfile = io.BytesIO(json.dumps(body).encode("utf-8"))
        self.headers = {"Content-Length": str(len(self.rfile.getvalue()))}
        self.request = None

    def send_response(self, status):
        self.status = status

    def send_header(self, name, value):
        self.sent_headers.append((name, value))

    def end_headers(self):
        pass

    def write(self, data):
        self.body.extend(data)

    def json_body(self):
        return json.loads(bytes(self.body).decode("utf-8"))


def drive_route(path: str, body: dict):
    """Drive the real routes.handle_post; return (status, payload)."""
    handler = _FakeHandler(body)
    cap = {}
    with patch.object(routes, "_check_csrf", lambda _h: True), \
            patch.object(routes, "read_body", lambda _h: body), \
            patch.object(routes, "j", lambda h, obj, *a, **k: cap.update(ok=obj, status=k.get("status", 200))), \
            patch.object(routes, "bad", lambda h, msg, code=400: cap.update(bad=(msg, code))):
        routes.handle_post(handler, urlparse(path))
    return cap.get("status"), cap.get("ok"), cap.get("bad")


def make_session(title, context, *, messages=None, **kwargs):
    s = Session(title=title, workspace="/tmp/hermes-webui-agent-home", messages=messages if messages is not None else list(context), context_messages=list(context), **kwargs)
    s.save()
    return s


def set_mid_turn_stream_state(session, stream_id, *, partial, prompt):
    """Register a live gateway stream the way the gateway worker + route layer do."""
    q = _FakeQueue()
    with cfg.STREAMS_LOCK:
        cfg.STREAMS[stream_id] = q
        cfg.CANCEL_FLAGS[stream_id] = threading.Event()
        cfg.STREAM_PARTIAL_TEXT[stream_id] = partial
        cfg.STREAM_REASONING_TEXT[stream_id] = ""
        cfg.STREAM_LIVE_TOOL_CALLS[stream_id] = []
    cfg.register_active_run(stream_id, session_id=session.session_id, backend="gateway", phase="gateway-request")
    cfg.register_stream_owner(stream_id, session.session_id)
    session.active_stream_id = stream_id
    session.pending_user_message = prompt
    started = time.time()
    session.pending_started_at = started
    session.pending_attachments = []
    session.pending_user_source = "webui"
    session.gateway_run = {"run_id": f"run-{stream_id}", "stream_id": stream_id}
    session.save()
    return started


def teardown_stream_state(stream_id):
    with cfg.STREAMS_LOCK:
        cfg.STREAMS.pop(stream_id, None)
        cfg.CANCEL_FLAGS.pop(stream_id, None)
        cfg.STREAM_PARTIAL_TEXT.pop(stream_id, None)
        cfg.STREAM_REASONING_TEXT.pop(stream_id, None)
        cfg.STREAM_LIVE_TOOL_CALLS.pop(stream_id, None)
        cfg.STREAM_LAST_EVENT_ID.pop(stream_id, None)
    cfg.unregister_active_run(stream_id)


def admit_run(session_id, *, msg_text, prefill_messages=None, context_messages=None, stream_id=None):
    """Drive the real _run_gateway_runs_api_streaming; return the captured body."""
    from api.gateway_chat import _STREAM_RUN_IDS, _run_gateway_runs_api_streaming

    del CAPTURED_BODIES[:]
    stream_id = stream_id or f"stream-admit-{session_id}"
    with cfg.STREAMS_LOCK:
        cfg.STREAM_PARTIAL_TEXT[stream_id] = ""
        cfg.STREAM_REASONING_TEXT[stream_id] = ""
    session = get_session(session_id) if context_messages is None else None
    try:
        with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            final_text, usage = _run_gateway_runs_api_streaming(
                session_id=session_id,
                msg_text=msg_text,
                model="test-model",
                workspace="/tmp",
                stream_id=stream_id,
                base_url="http://gw:8642",
                api_key="secret",
                prefill_messages=list(prefill_messages or []),
                body_extras={},
                put_gateway_event=lambda *_a, **_k: None,
                cancel_event=threading.Event(),
                session=session,
            )
        assert final_text == "turn answer", final_text
        return CAPTURED_BODIES[0]
    finally:
        with cfg.STREAMS_LOCK:
            cfg.STREAM_PARTIAL_TEXT.pop(stream_id, None)
            cfg.STREAM_REASONING_TEXT.pop(stream_id, None)
        _STREAM_RUN_IDS.pop(stream_id, None)


def history_rows(body):
    return [
        (row.get("role"), row.get("content"))
        for row in (body.get("conversation_history") or [])
    ]


PARENT_CONTEXT = [
    {"role": "user", "content": "P-u1"},
    {"role": "assistant", "content": "P-a1"},
]


class ReconciliationHarness(unittest.TestCase):
    def setUp(self):
        del CAPTURED_BODIES[:]
        GATEWAY_STORE.clear()
        _SseResponse.cancelled_runs.clear()
        with cfg.STREAMS_LOCK:
            cfg.STREAMS.clear()

    def tearDown(self):
        with cfg.STREAMS_LOCK:
            for key in ("STREAMS", "CANCEL_FLAGS", "STREAM_PARTIAL_TEXT", "STREAM_REASONING_TEXT", "STREAM_LIVE_TOOL_CALLS", "STREAM_LAST_EVENT_ID"):
                getattr(cfg, key).clear()
        cfg.ACTIVE_RUNS.clear()

    # ── Scenario 1: fork turn 1 (master parity) ─────────────────────────────

    def test_1_fork_first_turn_sends_parent_prefix(self):
        parent = make_session("parent", PARENT_CONTEXT)
        status, ok, _bad = drive_route("/api/session/branch", {"session_id": parent.session_id})
        self.assertEqual(status, 200, ok)
        fork_id = ok["session_id"]

        body = admit_run(fork_id, msg_text="F-q1")

        self.assertEqual(
            history_rows(body),
            [("user", "P-u1"), ("assistant", "P-a1")],
            f"fork turn 1 must send the copied parent prefix, got {history_rows(body)}",
        )
        self.assertEqual(body.get("input"), "F-q1")

    # ── Scenario 2: [ROUND-2 KILLER] fork turn 1 cancelled -> turn 2 ─────────

    def test_2_fork_cancelled_turn_flows_into_turn2_history(self):
        from api.streaming import cancel_stream

        parent = make_session("parent", PARENT_CONTEXT)
        _status, ok, _bad = drive_route("/api/session/branch", {"session_id": parent.session_id})
        fork_id = ok["session_id"]

        stream_id = "stream-fork-cancel"
        fork = get_session(fork_id)
        set_mid_turn_stream_state(fork, stream_id, partial="F-partial…", prompt="F-q1")
        # Gateway adopted/persisted the interrupted turn in its own store.
        GATEWAY_STORE[fork_id] = [
            {"role": "user", "content": "F-q1"},
            {"role": "assistant", "content": "F-partial…"},
        ]

        self.assertTrue(cancel_stream(stream_id))

        body = admit_run(fork_id, msg_text="F-q2")

        self.assertEqual(
            history_rows(body),
            [
                ("user", "P-u1"),
                ("assistant", "P-a1"),
                ("user", "F-q1"),
                ("assistant", "F-partial…"),
            ],
            "ROUND-2 KILLER CASE: fork turn 2 history must include the "
            f"reconciled interrupted turn, got {history_rows(body)}",
        )
        # The reconciled snapshot must not lose the parent prefix.
        self.assertEqual(body.get("input"), "F-q2")
        teardown_stream_state(stream_id)

    # ── Scenario 3: normal user Stop on a plain session ──────────────────────

    def test_3_user_stop_reconciles_rows_into_next_history(self):
        from api.streaming import cancel_stream

        session = make_session("plain", [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
        ])
        stream_id = "stream-stop"
        set_mid_turn_stream_state(session, stream_id, partial="partial answer text", prompt="q1")

        self.assertTrue(cancel_stream(stream_id))

        body = admit_run(session.session_id, msg_text="q2")

        self.assertEqual(
            history_rows(body),
            [
                ("user", "u0"),
                ("assistant", "a0"),
                ("user", "q1"),
                ("assistant", "partial answer text"),
            ],
            "user Stop must reconcile the interrupted turn into the next "
            f"captured history, got {history_rows(body)}",
        )
        teardown_stream_state(stream_id)

    # ── Scenario 4: truncate after an interrupted run (user edit wins) ───────

    def test_4_truncate_after_interrupt_sends_shortened_history(self):
        from api.streaming import cancel_stream

        session = make_session("trunc", [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
        ])
        stream_id = "stream-trunc"
        set_mid_turn_stream_state(session, stream_id, partial="to be truncated", prompt="q1")
        self.assertTrue(cancel_stream(stream_id))
        teardown_stream_state(stream_id)

        status, ok, bad = drive_route(
            "/api/session/truncate",
            {"session_id": session.session_id, "keep_count": 2},
        )
        self.assertIsNone(bad, bad)
        self.assertEqual(status, 200, ok)

        body = admit_run(session.session_id, msg_text="fresh question")

        self.assertEqual(
            history_rows(body),
            [("user", "u0"), ("assistant", "a0")],
            "user edit (truncate) must win: reconciled rows stay deleted, "
            f"got {history_rows(body)}",
        )

    # ── Scenario 5: prefill composes with reconciled history ─────────────────

    def test_5_prefill_composes_with_reconciled_history(self):
        from api.streaming import cancel_stream

        session = make_session("prefill", [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
        ])
        stream_id = "stream-prefill"
        set_mid_turn_stream_state(session, stream_id, partial="half done", prompt="q1")
        self.assertTrue(cancel_stream(stream_id))
        teardown_stream_state(stream_id)

        prefill = [
            {"role": "system", "content": "prefill system"},
            {"role": "user", "content": "prefill question"},
            {"role": "assistant", "content": "prefill reply"},
        ]
        body = admit_run(session.session_id, msg_text="q2", prefill_messages=prefill)

        self.assertEqual(
            history_rows(body),
            [
                ("user", "u0"),
                ("assistant", "a0"),
                ("user", "q1"),
                ("assistant", "half done"),
                ("user", "prefill question"),
                ("assistant", "prefill reply"),
            ],
            "prefill must COMPOSE with the reconciled context rows, got "
            f"{history_rows(body)}",
        )
        self.assertEqual(body.get("instructions"), "prefill system")

    # ── Scenario 6: double-settle appends the partial once ───────────────────

    def test_6_double_settle_appends_partial_once(self):
        from api.gateway_chat import _settle_gateway_cancelled_turn

        session = make_session("double", [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
        ])
        stream_id = "stream-double"
        started = set_mid_turn_stream_state(session, stream_id, partial="settled partial", prompt="q1")

        _settle_gateway_cancelled_turn(session.session_id, stream_id)

        # Simulate the same turn being settled again (worker replay / refetch):
        # restore the exact mid-turn identity the settle path guards on.
        reloaded = get_session(session.session_id)
        reloaded.active_stream_id = stream_id
        reloaded.pending_user_message = "q1"
        reloaded.pending_started_at = started
        reloaded.pending_attachments = []
        reloaded.pending_user_source = "webui"
        reloaded.save()
        _settle_gateway_cancelled_turn(session.session_id, stream_id)

        ctx = get_session(session.session_id).context_messages
        partial_rows = [
            row for row in ctx
            if row.get("role") == "assistant" and row.get("content") == "settled partial"
        ]
        self.assertEqual(
            len(partial_rows), 1,
            f"double settle must append the partial exactly once, got {len(partial_rows)}: {ctx}",
        )
        user_rows = [
            row for row in ctx
            if row.get("role") == "user" and row.get("content") == "q1"
        ]
        self.assertEqual(
            len(user_rows), 1,
            f"double settle must not duplicate the user row, got {len(user_rows)}: {ctx}",
        )
        teardown_stream_state(stream_id)

    # ── Round-3 re-gate (2026-10-05 22:07 UTC): CORE items 1-3 ───────────────

    def test_7_empty_context_seeded_from_transcript_before_interrupt(self):
        """Item 1: Stop on an imported-CLI-shaped session keeps the whole history.

        context_messages is EMPTY while session.messages holds the prior
        conversation. The reconcile must seed the canonical transcript
        projection BEFORE appending the interrupted turn; otherwise the next
        turn's outbound history is a two-row world that replaces the gateway's
        stored transcript and the earlier conversation drops out of model
        context.
        """
        from api.streaming import cancel_stream

        session = make_session(
            "imported", [],
            messages=[
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
            ],
        )
        # Imported CLI shape: empty model-facing context, full transcript.
        session.context_messages = []
        session.save()

        stream_id = "stream-import"
        set_mid_turn_stream_state(session, stream_id, partial="imported partial", prompt="q1")
        self.assertTrue(cancel_stream(stream_id))
        teardown_stream_state(stream_id)

        body = admit_run(session.session_id, msg_text="q2")

        self.assertEqual(
            history_rows(body),
            [
                ("user", "u0"),
                ("assistant", "a0"),
                ("user", "q1"),
                ("assistant", "imported partial"),
            ],
            "empty context must be seeded from the transcript projection before "
            f"the interrupted turn is appended, got {history_rows(body)}",
        )
        self.assertEqual(body.get("input"), "q2")

    def test_8_settle_attaches_partial_to_its_own_turn_not_older_prompt(self):
        """Item 2: the settle must reconcile while the pending identity is live.

        A queued prompt B runs (eager checkpoint in the transcript) while an
        older user row (turn A) is still unanswered in the context. Persisting
        the cancellation BEFORE the reconcile cleared the pending identity:
        the pending-turn materializer's token dedup then returned early
        without mirroring B into the context, and the partial-append guard —
        seeing only "any trailing user row" — attached B's partial under A.
        """
        from api.gateway_chat import _settle_gateway_cancelled_turn
        from api.routes import _checkpoint_user_message_for_eager_session_save

        session = make_session("queued", [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "A-old"},
        ])
        stream_id = "stream-queued"
        started = set_mid_turn_stream_state(session, stream_id, partial="B partial", prompt="qB")
        # Eager save mode: the turn's user row is checkpointed into the
        # transcript (with the active-turn token) at chat-start.
        _checkpoint_user_message_for_eager_session_save(
            session, "qB", [], started,
        )
        session.save()

        _settle_gateway_cancelled_turn(session.session_id, stream_id)

        ctx = get_session(session.session_id).context_messages
        self.assertEqual(
            history_rows({"conversation_history": [
                {"role": row.get("role"), "content": row.get("content")}
                for row in ctx
            ]}),
            [
                ("user", "u0"),
                ("assistant", "a0"),
                ("user", "A-old"),
                ("user", "qB"),
                ("assistant", "B partial"),
            ],
            "the partial must attach to B's own row; older rows untouched, "
            f"got {ctx}",
        )
        teardown_stream_state(stream_id)

    def _prepare_regenerate_start(self, session, msg_text):
        """Run the production regenerate-start steps and persist the result.

        Mirrors _start_regeneration_stream_locked: plan, apply (truncate at
        the retained row), then _prepare_chat_start_session_for_stream, which
        restamps the retained context row with the NEW stream's active-turn
        token, then saves.
        """
        from api.routes import (
            _RETAINED_CONTEXT_USER_UNSET,
            _prepare_chat_start_session_for_stream,
        )
        from api.session_ops import apply_regeneration_plan, plan_regeneration

        fresh = get_session(session.session_id)
        plan = plan_regeneration(fresh)
        self.assertIsNotNone(plan)
        applied, retained_context_user = apply_regeneration_plan(
            fresh, plan, return_context_user=True,
        )
        self.assertTrue(applied)
        retained_user = fresh.messages[-1]
        stream_id = f"stream-regen-{session.session_id[-6:]}"
        _prepare_chat_start_session_for_stream(
            fresh,
            msg=plan.turn.message_text,
            attachments=[],
            workspace="/tmp",
            model="test-model",
            model_provider="",
            stream_id=stream_id,
            source="webui",
            retained_user=retained_user,
            retained_context_user=(
                retained_context_user
                if retained_context_user is not None
                else _RETAINED_CONTEXT_USER_UNSET
            ),
            defer_save=True,
        )
        # _start_regeneration_stream_locked saves right after prepare; the
        # worker's get_session must see the restamped row.
        fresh.save()
        return stream_id

    def test_9_regenerate_after_stop_sends_prompt_exactly_once(self):
        """Item 3a: Stop then regenerate - the gateway sees the prompt once.

        Pinned to the post-Stop regenerate-admission state the maintainer's
        repro produced: the retained interrupted-turn user row (restamped at
        regenerate start with the NEW stream's token) followed by the
        reconciled partial. Sending that row in conversation_history as well
        as `input` made the gateway see the prompt twice.
        """
        from api.process_event_utils import build_active_turn_token

        session = make_session("regen", [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
        ])
        stop_stream = "stream-stop-regen"
        started = set_mid_turn_stream_state(
            session, stop_stream, partial="cut off mid", prompt="q1",
        )
        teardown_stream_state(stop_stream)

        # Regenerate start: the plan truncates at the retained row and
        # _prepare_chat_start_session_for_stream restamps it with the NEW
        # stream's token; the reconciled partial trails it.
        regen_stream = "stream-regen-once"
        token = build_active_turn_token(regen_stream, started)
        session.context_messages = [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {
                "role": "user", "content": "q1", "_recovered": True,
                "_active_turn_token": token, "timestamp": int(started),
            },
            {
                "role": "assistant", "content": "cut off mid", "_partial": True,
                "_gateway_cancelled_stream": stop_stream,
            },
        ]
        session.active_stream_id = regen_stream
        session.pending_user_message = "q1"
        session.pending_started_at = started
        session.pending_attachments = []
        session.pending_user_source = "webui"
        session.save()

        body = admit_run(session.session_id, msg_text="q1", stream_id=regen_stream)

        seen = history_rows(body).count(("user", "q1"))
        if body.get("input") == "q1":
            seen += 1
        self.assertEqual(
            seen, 1,
            f"regenerate must deliver the prompt exactly once, got {seen}: "
            f"input={body.get('input')!r} history={history_rows(body)}",
        )
        self.assertEqual(body.get("input"), "q1")
        teardown_stream_state(regen_stream)

    def _gateway_worker_env(self):
        return patch.dict(os.environ, {
            "HERMES_WEBUI_CHAT_BACKEND": "gateway",
            "HERMES_WEBUI_GATEWAY_USE_RUNS_API": "1",
        })

    def _drive_full_worker(self, session_id, msg_text, stream_id):
        """Drive the real _run_gateway_chat_streaming end to end."""
        import api.gateway_chat as gateway_chat
        from api.config import create_stream_channel

        with cfg.STREAMS_LOCK:
            cfg.STREAMS[stream_id] = create_stream_channel()
        with patch.object(gateway_chat, "gateway_supports_approval", lambda *a, **k: True), \
                patch.object(gateway_chat.urllib.request, "urlopen", side_effect=fake_urlopen):
            gateway_chat._run_gateway_chat_streaming(
                session_id, msg_text, "test-model", "/tmp", stream_id, [],
            )

    def test_10_regenerate_writeback_saves_single_user_row(self):
        """Item 3b: post-regeneration success writeback keeps ONE prompt row."""
        from api.streaming import cancel_stream

        session = make_session("regen-writeback", [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
        ])
        stop_stream = "stream-stop-wb"
        set_mid_turn_stream_state(session, stop_stream, partial="cut off mid", prompt="q1")
        self.assertTrue(cancel_stream(stop_stream))
        teardown_stream_state(stop_stream)

        stream_id = self._prepare_regenerate_start(session, "q1")
        with self._gateway_worker_env():
            self._drive_full_worker(session.session_id, "q1", stream_id)

        saved = get_session(session.session_id)
        ctx = saved.context_messages
        user_rows = [
            row for row in ctx
            if isinstance(row, dict) and row.get("role") == "user" and row.get("content") == "q1"
        ]
        self.assertEqual(
            len(user_rows), 1,
            f"writeback must save exactly one q1 user row, got {len(user_rows)}: {ctx}",
        )
        assistant_rows = [
            row for row in ctx
            if isinstance(row, dict) and row.get("role") == "assistant"
            and row.get("content") == "turn answer"
        ]
        self.assertEqual(
            len(assistant_rows), 1,
            f"writeback must save the regenerated answer once, got {ctx}",
        )
        self.assertIsNone(saved.gateway_run)
        self.assertIsNone(saved.active_stream_id)
        teardown_stream_state(stream_id)

    def test_11_run_cancelled_event_settles_reconciles_and_matches_store(self):
        """Item 6 headline: the worker's run.cancelled path reconciles context.

        The gateway cancels the run mid-stream; the worker must settle the
        cancelled turn (prompt + streamed partial into context), clear the run
        record, and the next turn's history must carry the interrupted turn
        matching the gateway's own persisted store.
        """
        session = make_session("worker-cancel", [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
        ])
        stream_id = "stream-worker-cancel"
        set_mid_turn_stream_state(session, stream_id, partial="gateway cut", prompt="q1")

        # Arm a gateway-side cancel for the admitted run id (run-1).
        _SseResponse.cancelled_runs.add("run-1")
        with self._gateway_worker_env():
            self._drive_full_worker(session.session_id, "q1", stream_id)

        saved = get_session(session.session_id)
        self.assertIsNone(saved.gateway_run, "cancel settle must clear the run record")
        # The gateway adopted/persisted the interrupted turn in its own store.
        self.assertEqual(
            GATEWAY_STORE.get(session.session_id),
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "gateway-persisted partial"},
            ],
            f"gateway store must hold the adopted interrupted turn, got {GATEWAY_STORE}",
        )
        # WebUI's mirror: prompt + streamed partial under it.
        ctx = saved.context_messages
        self.assertEqual(
            history_rows({"conversation_history": [
                {"role": row.get("role"), "content": row.get("content")}
                for row in ctx
            ]}),
            [
                ("user", "u0"),
                ("assistant", "a0"),
                ("user", "q1"),
                ("assistant", "gateway cut"),
            ],
            f"run.cancelled must reconcile prompt + partial into context, got {ctx}",
        )
        teardown_stream_state(stream_id)

        # Next turn: the outbound history carries the interrupted turn.
        body = admit_run(session.session_id, msg_text="q2")
        self.assertEqual(
            history_rows(body),
            [
                ("user", "u0"),
                ("assistant", "a0"),
                ("user", "q1"),
                ("assistant", "gateway cut"),
            ],
            f"next turn must keep the story after a gateway-side cancel, got {history_rows(body)}",
        )

    # ── Round-4 re-gate (2026-10-06 14:38 UTC): CORE items 1-2 + SILENT item 3

    def test_12_eager_stop_history_pairs_partial_with_its_own_question(self):
        """Item 1: the recovered-question filter must not orphan the partial.

        Eager-save Stop with an older unanswered A-old in context: the stored
        context is correct ([A-old, qB, B partial]) but the history builder's
        recovered-row rule drops qB (a _recovered row preceded by a user
        row), so the Gateway receives [A-old, B partial] — the partial paired
        with the wrong question. The reconcile must clear the owning row's
        recovered flag right before appending the partial under it, so the
        partial rides with its own question.
        """
        from api.gateway_chat import _settle_gateway_cancelled_turn
        from api.routes import _checkpoint_user_message_for_eager_session_save

        session = make_session("queued-history", [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "A-old"},
        ])
        stream_id = "stream-queued-hist"
        started = set_mid_turn_stream_state(session, stream_id, partial="B partial", prompt="qB")
        _checkpoint_user_message_for_eager_session_save(
            session, "qB", [], started,
        )
        session.save()

        _settle_gateway_cancelled_turn(session.session_id, stream_id)
        teardown_stream_state(stream_id)

        body = admit_run(session.session_id, msg_text="q2")

        self.assertEqual(
            history_rows(body),
            [
                ("user", "u0"),
                ("assistant", "a0"),
                ("user", "A-old"),
                ("user", "qB"),
                ("assistant", "B partial"),
            ],
            "the partial must ride with its OWN question: qB must not be "
            f"dropped from the outbound history, got {history_rows(body)}",
        )
        self.assertEqual(body.get("input"), "q2")



if __name__ == "__main__":
    unittest.main(verbosity=2)
