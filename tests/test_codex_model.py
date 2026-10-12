"""SDK boundary checks: subscription-only, evidence isolation, failures and cancellation."""
import argparse
import asyncio
import json
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from openai_codex.generated.v2_all import (
    ItemCompletedNotification,
    ItemStartedNotification,
    CommandExecutionSource,
    ReasoningEffort,
    ThreadItem,
    ThreadTokenUsage,
    ThreadTokenUsageUpdatedNotification,
    TokenUsageBreakdown,
    Turn,
    TurnCompletedNotification,
    TurnItemsView,
    TurnStatus,
)
from openai_codex.errors import InternalRpcError

from crypto_quant.research.factor_mining.codex_model import CodexModel, DEFAULT_MODEL, PROVIDER
from crypto_quant.research.factor_mining.model import ApiCallError, RecoverableToolError


class Payload:
    def __init__(self, data, **attrs):
        self.data = data
        self.__dict__.update(attrs)

    def model_dump(self, **kwargs):
        return self.data


def event(method, data, **attrs):
    return NS(method=method, payload=Payload(data, **attrs))


def events(*, status=TurnStatus.completed):
    return [event("item/completed", {"item": {"type": "agentMessage", "phase": "final_answer", "text": '{"ok":true}'}}),
            event("thread/tokenUsage/updated", {"tokenUsage": {"last": {"inputTokens": 20, "outputTokens": 5}}}),
            event("turn/completed", {"turn": {"status": status.value, "error": None}}, turn=NS(status=status))]


class FakeCodex:
    def __init__(self, stream_events=None):
        self.calls = []
        self.closed = False
        self.stream_events = events() if stream_events is None else stream_events
        self.start = AsyncMock()
        self.initialize = AsyncMock()
        self.account_read = AsyncMock(return_value=NS(account=NS(root=NS(type="chatgpt"))))
        selected = NS(model=DEFAULT_MODEL, supported_reasoning_efforts=[NS(reasoning_effort=ReasoningEffort.max)],
                      service_tiers=[])
        self.model_list = AsyncMock(return_value=NS(data=[selected]))

    async def request(self, *args, **kwargs):
        return NS(config=Payload({"mcp_servers": {"private": {}}, "plugins": {"personal": {}}}))

    async def thread_start(self, params):
        self.calls.append(params)
        return NS(thread=NS(id=f"thread-{len(self.calls)}"), model=params["model"], model_provider="openai",
                  service_tier=params.get("serviceTier"))

    async def turn_start(self, thread_id, prompt, params):
        self.prompt, self.turn_options = prompt, params
        self.pending_events = iter(self.stream_events)
        return NS(turn=NS(id="turn"))

    async def next_turn_notification(self, turn_id):
        return next(self.pending_events)

    async def close(self):
        self.closed = True


class FakeSyncCodex:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.close_count = 0
        self.on_close = None

    def start(self):
        pass

    def initialize(self):
        pass

    def account_read(self):
        return NS(account=NS(root=NS(type="chatgpt")))

    def model_list(self):
        return NS(data=[NS(model=DEFAULT_MODEL,
                           supported_reasoning_efforts=[NS(reasoning_effort=ReasoningEffort.max)],
                           service_tiers=[])])

    def request(self, method, params, *, response_model):
        self.calls.append((method, params, response_model))
        return NS(config=NS(model_dump=lambda **kwargs: {
            "mcp_servers": {"private": {}}, "plugins": {"personal": {}}}))

    def thread_start(self, params):
        self.thread_params = params
        return NS(thread=NS(id="thread-native"), model=params["model"], model_provider="openai",
                  service_tier=params.get("serviceTier"))

    def close(self):
        self.close_count += 1
        self.closed = True
        if self.on_close is not None:
            self.on_close()


def native_event(method, payload):
    return NS(method=method, payload=payload)


def native_item(item_type, **fields):
    return ThreadItem.model_validate({"type": item_type, "id": f"item-{item_type}", **fields})


def native_usage():
    usage = TokenUsageBreakdown(cached_input_tokens=0, input_tokens=20, output_tokens=5,
                                reasoning_output_tokens=0, total_tokens=25)
    return ThreadTokenUsage(last=usage, total=usage)


class CodexModelTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "system", "content": "Only use stage A evidence."},
                         {"role": "user", "content": '证据\n"quoted"'},
                         {"role": "assistant", "content": '{"read_records": ["A-1"]}'},
                         {"role": "user", "content": "Full original A-1 record"}]

    def test_native_tool_loop_retries_only_recoverable_tool_errors(self):
        fake = FakeSyncCodex()
        callback = {}
        tool_responses = []
        answer = '{"result":{"candidate_ref":"FM-CAND-1","definition":"A test factor."},"read_records":[]}'

        class Stream:
            def __iter__(self):
                params = {"arguments": {"limit": 20}, "callId": "call-1", "threadId": "thread-native",
                          "tool": "read_page", "turnId": "turn-native"}
                tool_responses.append(callback["handler"]("item/tool/call", params))
                yield native_event("item/started", ItemStartedNotification(
                    item=native_item("dynamicToolCall", arguments={}, status="completed", tool="read_page"),
                    started_at_ms=1, thread_id="thread-native", turn_id="turn-native"))
                params = {**params, "arguments": {"limit": 5}, "callId": "call-2"}
                tool_responses.append(callback["handler"]("item/tool/call", params))
                yield native_event("item/completed", ItemCompletedNotification(
                    item=native_item("contextCompaction"), completed_at_ms=2,
                    thread_id="thread-native", turn_id="turn-native"))
                params = {**params, "arguments": {"limit": 3}, "callId": "call-3"}
                tool_responses.append(callback["handler"]("item/tool/call", params))
                yield native_event("item/completed", ItemCompletedNotification(
                    item=native_item("agentMessage", text=answer, phase="final_answer"),
                    completed_at_ms=3, thread_id="thread-native", turn_id="turn-native"))
                yield native_event("thread/tokenUsage/updated", ThreadTokenUsageUpdatedNotification(
                    token_usage=native_usage(), thread_id="thread-native", turn_id="turn-native"))
                yield native_event("turn/completed", TurnCompletedNotification(
                    thread_id="thread-native", turn=Turn(id="turn-native", status=TurnStatus.completed,
                                                          items=[], items_view=TurnItemsView.full)))

            def close(self):
                pass

        turn = NS(id="turn-native", stream=lambda: Stream())
        thread = NS(turn=Mock())
        thread.turn.side_effect = lambda *args, **kwargs: turn

        def make_client(config, *, approval_handler):
            callback["handler"] = approval_handler
            return fake

        schema = {"type": "object", "properties": {"limit": {"type": "integer"}},
                  "required": ["limit"], "additionalProperties": False}
        output_schema = {"type": "object", "required": ["result", "read_records"],
                         "properties": {"result": {"type": "object"},
                                        "read_records": {"type": "array", "maxItems": 0}},
                         "additionalProperties": False}
        invocations = []

        def handler(name, arguments):
            invocations.append((name, arguments))
            if len(invocations) == 1:
                raise RecoverableToolError("Page exceeds response budget; narrow the pointer or reduce limit.")
            return {"records": [f"A-{len(invocations) - 1}"]}

        with patch("crypto_quant.research.factor_mining.codex_model.CodexClient", side_effect=make_client), \
                patch("crypto_quant.research.factor_mining.codex_model._config", return_value=object()), \
                patch("crypto_quant.research.factor_mining.codex_model.Thread", return_value=thread) as thread_type:
            reply = CodexModel().complete_with_tools(
                self.messages, tools=[{"type": "function", "name": "read_page", "description": "Read a page",
                                       "inputSchema": schema}],
                tool_handler=handler, max_output_tokens=None, session_id="native-tools",
                output_schema=output_schema)

        self.assertEqual(reply.text, answer)
        self.assertEqual([item[0] for item in invocations], ["read_page"] * 3)
        self.assertFalse(tool_responses[0]["success"])
        self.assertIn("narrow the pointer", tool_responses[0]["contentItems"][0]["text"])
        self.assertTrue(tool_responses[1]["success"])
        self.assertTrue(tool_responses[2]["success"])
        self.assertTrue(fake.closed)
        self.assertEqual(fake.thread_params["dynamicTools"][0]["name"], "read_page")
        self.assertEqual(fake.thread_params["dynamicTools"][0]["inputSchema"], schema)
        self.assertEqual(thread_type.return_value.turn.call_args.kwargs["output_schema"], output_schema)
        self.assertIn("contextCompaction", [event["params"]["item"]["type"]
                                            for event in reply.raw_response["events"]
                                            if "item" in event["params"]])
        self.assertEqual(reply.raw_response["tool_calls"][0]["error_type"], "RecoverableToolError")
        self.assertEqual(reply.raw_response["tool_calls"][1]["result"], {"records": ["A-1"]})
        self.assertEqual(reply.raw_response["tool_calls"][2]["result"], {"records": ["A-2"]})
        self.assertEqual([arguments for _, arguments in invocations],
                         [{"limit": 20}, {"limit": 5}, {"limit": 3}])

    def test_native_tool_handler_unexpected_error_fails_closed(self):
        fake = FakeSyncCodex()
        callback = {}

        class Stream:
            def __iter__(self):
                callback["handler"]("item/tool/call", {"arguments": {"limit": 1}, "callId": "call-1",
                    "threadId": "thread-native", "tool": "read_page", "turnId": "turn-native"})
                return iter(())

            def close(self):
                pass

        turn = NS(id="turn-native", stream=lambda: Stream())
        thread = NS(turn=Mock())
        thread.turn.side_effect = lambda *args, **kwargs: turn

        def make_client(config, *, approval_handler):
            callback["handler"] = approval_handler
            return fake

        with patch("crypto_quant.research.factor_mining.codex_model.CodexClient", side_effect=make_client), \
                patch("crypto_quant.research.factor_mining.codex_model._config", return_value=object()), \
                patch("crypto_quant.research.factor_mining.codex_model.Thread", return_value=thread):
            with self.assertRaises(ApiCallError) as caught:
                CodexModel().complete_with_tools(
                    self.messages,
                    tools=[{"type": "function", "name": "read_page", "description": "Read a page",
                            "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}},
                                            "required": ["limit"], "additionalProperties": False}}],
                    tool_handler=lambda *_: (_ for _ in ()).throw(ValueError("corrupt record")),
                    max_output_tokens=None, session_id="native-tools-fail-closed",
                    output_schema={"type": "object", "properties": {"result": {"type": "object"},
                                                                        "read_records": {"type": "array"}},
                                   "required": ["result", "read_records"], "additionalProperties": False})
        self.assertFalse(caught.exception.retryable)
        self.assertIn("corrupt record", caught.exception.diagnostics["error"])
        self.assertEqual(caught.exception.diagnostics["tool_calls"][0]["error_type"], "ValueError")
        self.assertTrue(fake.closed)

    def test_native_protocol_violations_fail_closed(self):
        cases = {
            "unknown_server_request": "unsupported Codex server request",
            "unregistered_tool": "unregistered dynamic tool",
            "foreign_thread": "another thread",
            "foreign_turn": "another turn",
            "unknown_item": "unsupported item",
        }
        for case, message in cases.items():
            with self.subTest(case=case):
                fake = FakeSyncCodex()
                callback = {}
                handler_calls = []

                class Stream:
                    def __iter__(self):
                        if case == "unknown_item":
                            yield native_event("item/started", ItemStartedNotification(
                                item=native_item("commandExecution", command="ls", command_actions=[],
                                                  cwd="/", status="completed",
                                                  source=CommandExecutionSource.agent),
                                started_at_ms=1, thread_id="thread-native", turn_id="turn-native"))
                            return
                        params = {"arguments": {"limit": 1}, "callId": "call-1",
                                  "threadId": "thread-native", "tool": "read_page",
                                  "turnId": "turn-native"}
                        method = "item/tool/call"
                        if case == "unknown_server_request":
                            method, params = "execCommandApproval", {}
                        elif case == "unregistered_tool":
                            params["tool"] = "delete_record"
                        elif case == "foreign_thread":
                            params["threadId"] = "thread-other"
                        elif case == "foreign_turn":
                            params["turnId"] = "turn-other"
                        callback["handler"](method, params)
                        return iter(())

                    def close(self):
                        pass

                turn = NS(id="turn-native", stream=lambda: Stream())
                thread = NS(turn=Mock())
                thread.turn.side_effect = lambda *args, **kwargs: turn

                def make_client(config, *, approval_handler):
                    callback["handler"] = approval_handler
                    return fake

                with patch("crypto_quant.research.factor_mining.codex_model.CodexClient",
                           side_effect=make_client), \
                        patch("crypto_quant.research.factor_mining.codex_model._config", return_value=object()), \
                        patch("crypto_quant.research.factor_mining.codex_model.Thread", return_value=thread):
                    with self.assertRaises(ApiCallError) as caught:
                        CodexModel().complete_with_tools(
                            self.messages,
                            tools=[{"type": "function", "name": "read_page", "description": "Read a page",
                                    "inputSchema": {"type": "object",
                                                    "properties": {"limit": {"type": "integer"}},
                                                    "required": ["limit"], "additionalProperties": False}}],
                            tool_handler=lambda *args: handler_calls.append(args),
                            max_output_tokens=None, session_id=f"native-invalid-{case}",
                            output_schema={"type": "object", "properties": {"result": {"type": "object"},
                                                                                 "read_records": {"type": "array"}},
                                           "required": ["result", "read_records"],
                                           "additionalProperties": False})
                self.assertFalse(caught.exception.retryable)
                self.assertIn(message, caught.exception.diagnostics["error"])
                self.assertTrue(caught.exception.diagnostics["generation_started"])
                self.assertEqual(handler_calls, [])
                self.assertTrue(fake.closed)

    def test_native_tool_worker_deadline_and_final_notification_timeout(self):
        tool_schema = {"type": "object", "properties": {"limit": {"type": "integer"}},
                       "required": ["limit"], "additionalProperties": False}
        for wait_at in ("tool", "final_notification"):
            with self.subTest(wait_at=wait_at):
                fake = FakeSyncCodex()
                callback = {}
                released = threading.Event()
                entered = threading.Event()
                reader_failed = threading.Event()
                reader_errors = []
                handler_finished = threading.Event()
                callback_responses = []
                if wait_at == "final_notification":
                    fake.on_close = released.set

                class Stream:
                    def __iter__(self):
                        entered.set()
                        if wait_at == "tool":
                            def reader_loop():
                                try:
                                    response = callback["handler"]("item/tool/call", {
                                        "arguments": {"limit": 1}, "callId": "call-1",
                                        "threadId": "thread-native", "tool": "read_page",
                                        "turnId": "turn-native"})
                                    callback_responses.append(response)
                                except BaseException as exc:
                                    reader_errors.append(exc)
                                finally:
                                    # CodexClient._reader_loop calls _router.fail_all after this error.
                                    reader_failed.set()

                            threading.Thread(target=reader_loop, daemon=True).start()
                            if not reader_failed.wait(2):
                                raise TimeoutError("simulated reader loop did not leave the tool callback")
                            if reader_errors:
                                raise reader_errors[0]
                        else:
                            released.wait(2)
                            raise RuntimeError("Codex app-server process closed")
                        return iter(())

                    def close(self):
                        pass

                turn = NS(id="turn-native", stream=lambda: Stream())
                thread = NS(turn=Mock())
                thread.turn.side_effect = lambda *args, **kwargs: turn

                def make_client(config, *, approval_handler):
                    callback["handler"] = approval_handler
                    return fake

                def tool_handler(*args):
                    entered.set()
                    released.wait(3)
                    handler_finished.set()
                    return {"records": []}

                with patch("crypto_quant.research.factor_mining.codex_model.CodexClient",
                           side_effect=make_client), \
                        patch("crypto_quant.research.factor_mining.codex_model._config", return_value=object()), \
                        patch("crypto_quant.research.factor_mining.codex_model.Thread", return_value=thread):
                    with self.assertRaises(ApiCallError) as caught:
                        CodexModel(timeout_seconds=1).complete_with_tools(
                            self.messages,
                            tools=[{"type": "function", "name": "read_page", "description": "Read a page",
                                    "inputSchema": tool_schema}],
                            tool_handler=tool_handler, max_output_tokens=None,
                            session_id=f"native-timeout-{wait_at}",
                            output_schema={"type": "object", "properties": {"result": {"type": "object"},
                                                                                 "read_records": {"type": "array"}},
                                           "required": ["result", "read_records"],
                                           "additionalProperties": False})
                self.assertTrue(entered.is_set())
                self.assertFalse(caught.exception.retryable)
                self.assertEqual(caught.exception.diagnostics["error_type"], "TimeoutError")
                self.assertTrue(caught.exception.diagnostics["generation_started"])
                self.assertTrue(fake.closed)
                self.assertEqual(fake.close_count, 1)
                if wait_at == "tool":
                    self.assertFalse(released.is_set(), "CodexClient.close must not release tool work")
                    self.assertEqual(callback_responses, [], "the timed-out result must not reach Codex")
                    self.assertTrue(reader_failed.is_set())
                    self.assertIsInstance(reader_errors[0], TimeoutError)
                    self.assertEqual(caught.exception.diagnostics["tool_calls"][0]["error_type"],
                                     "TimeoutError")
                    released.set()
                    self.assertTrue(handler_finished.wait(1))
                else:
                    self.assertTrue(released.is_set(), "closing Codex should unblock the notification wait")

    def call(self, fake):
        with patch("crypto_quant.research.factor_mining.codex_model.AsyncCodexClient", return_value=fake):
            return CodexModel().complete(self.messages, max_output_tokens=None, session_id="A-evaluator")

    def test_explicit_runtime_is_used_and_roundtrips_actual_version(self):
        from crypto_quant.research.factor_mining.model_config import model_from_settings
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "codex"
            binary.write_text("#!/bin/sh\necho 'codex-cli 0.159.2'\n")
            binary.chmod(0o755)
            model = CodexModel(codex_bin=str(binary))
            self.assertEqual(model.settings()["runtime_version"], "0.159.2")
            self.assertEqual(model.settings()["codex_bin"], str(binary.resolve()))
            self.assertEqual(model_from_settings(model.settings()).settings(), model.settings())
            fake = FakeCodex()
            with patch("crypto_quant.research.factor_mining.codex_model.AsyncCodexClient",
                       return_value=fake) as client:
                reply = model.complete(self.messages, max_output_tokens=None, session_id="native-runtime")
            self.assertEqual(client.call_args.args[0].codex_bin, str(binary.resolve()))
            self.assertEqual(reply.raw_response["settings"], model.settings())
            self.assertNotIn("codex_bin", CodexModel().settings())

    def test_explicit_runtime_rejects_invalid_executable_and_mimo_routing(self):
        from crypto_quant.research.factor_mining.model_config import add_model_arguments, model_from_args
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "codex"
            binary.write_text("#!/bin/sh\necho 'codex-cli 0.159.2'\n")
            with self.assertRaisesRegex(ValueError, "executable|可执行"):
                CodexModel(codex_bin=str(binary))
            binary.chmod(0o755)
            parser = argparse.ArgumentParser()
            add_model_arguments(parser)
            with self.assertRaisesRegex(ValueError, "codex-bin|codex_bin"):
                model_from_args(parser.parse_args(["--provider", "mimo", "--codex-bin", str(binary)]))
            model = model_from_args(parser.parse_args([
                "--provider", "codex", "--model", "gpt-6-luna", "--reasoning-effort", "max",
                "--codex-bin", str(binary)]))
            self.assertEqual(model.model, "gpt-6-luna")
            self.assertEqual(model.reasoning_effort, "max")
            self.assertEqual(model.settings()["runtime_version"], "0.159.2")

    def test_real_sdk_shape_maps_usage_and_preserves_all_transcript_content(self):
        fake = FakeCodex()
        reply = self.call(fake)
        self.assertEqual(json.loads(fake.prompt), self.messages[1:])
        self.assertIn(self.messages[0]["content"], fake.calls[0]["baseInstructions"])
        self.assertEqual(fake.calls[0]["sandbox"], "read-only")
        self.assertEqual(fake.calls[0]["approvalPolicy"], "never")
        self.assertTrue(fake.calls[0]["ephemeral"])
        self.assertEqual(fake.calls[0]["config"]["mcp_servers"], {"private": {"enabled": False}})
        self.assertEqual(fake.calls[0]["config"]["plugins"], {"personal": {"enabled": False}})
        self.assertEqual(reply.usage["last"]["outputTokens"], 5)
        self.assertEqual(reply.finish_reason, "stop")
        self.assertEqual(fake.calls[0]["model"], "gpt-5.6-luna")
        self.assertEqual(fake.turn_options, {"effort": "max"})
        self.assertEqual(reply.raw_response["settings"]["provider"], PROVIDER)
        self.assertTrue(fake.closed)
        again = self.call(fake)
        self.assertNotEqual(reply.raw_response["thread_id"], again.raw_response["thread_id"])

    def test_api_key_login_does_not_start_a_turn(self):
        fake = FakeCodex()
        fake.account_read.return_value = NS(account=NS(root=NS(type="apiKey")))
        with self.assertRaisesRegex(ApiCallError, "subscription login"):
            self.call(fake)
        self.assertEqual(fake.calls, [])
        self.assertTrue(fake.closed)

    def test_fast_tier_is_explicit_preserves_effort_and_roundtrips_settings(self):
        from crypto_quant.research.factor_mining.model_config import model_from_settings
        fake = FakeCodex()
        selected = fake.model_list.return_value.data[0]
        selected.model = "gpt-6.1-sol"
        selected.service_tiers = [NS(id="priority")]
        selected.supported_reasoning_efforts = [NS(reasoning_effort=ReasoningEffort.high)]
        model = CodexModel("gpt-6.1-sol", reasoning_effort="high", service_tier="priority")
        self.assertEqual(model_from_settings(model.settings()).settings(), model.settings())
        with patch("crypto_quant.research.factor_mining.codex_model.AsyncCodexClient", return_value=fake) as client:
            reply = model.complete(self.messages, max_output_tokens=None, session_id="fast-evaluator")
        self.assertIn('features.fast_mode=true', client.call_args.args[0].config_overrides)
        self.assertEqual(fake.calls[0]["serviceTier"], "priority")
        self.assertEqual(fake.turn_options, {"effort": "high", "serviceTier": "priority"})
        self.assertEqual(reply.raw_response["accepted_service_tier"], "priority")
        self.assertEqual(reply.raw_response["settings"]["service_tier"], "priority")
        self.assertNotIn("service_tier", CodexModel().settings())

    def test_fast_tier_must_be_supported_and_accepted_before_generation(self):
        for tiers in ([], None):
            with self.subTest(tiers=tiers):
                fake = FakeCodex()
                fake.model_list.return_value.data[0].service_tiers = tiers
                with patch("crypto_quant.research.factor_mining.codex_model.AsyncCodexClient", return_value=fake):
                    with self.assertRaisesRegex(ApiCallError, "does not support requested Fast"):
                        CodexModel(service_tier="priority").complete(
                            self.messages, max_output_tokens=None, session_id="unsupported-fast")
                self.assertEqual(fake.calls, [])
        fake = FakeCodex()
        fake.model_list.return_value.data[0].service_tiers = [NS(id="priority")]
        fake.thread_start = AsyncMock(return_value=NS(thread=NS(id="thread"), model=DEFAULT_MODEL,
                                                      model_provider="openai", service_tier=None))
        fake.turn_start = AsyncMock()
        with patch("crypto_quant.research.factor_mining.codex_model.AsyncCodexClient", return_value=fake):
            with self.assertRaisesRegex(ApiCallError, "did not accept the requested Fast"):
                CodexModel(service_tier="priority").complete(
                    self.messages, max_output_tokens=None, session_id="rejected-fast")
        fake.turn_start.assert_not_awaited()
        self.assertTrue(fake.closed)

    def test_failure_and_interruption_cannot_be_returned_as_success(self):
        for status in (TurnStatus.failed, TurnStatus.interrupted):
            with self.subTest(status=status):
                fake = FakeCodex(events(status=status))
                with self.assertRaises(ApiCallError) as caught:
                    self.call(fake)
                self.assertFalse(caught.exception.retryable)
                self.assertEqual(caught.exception.diagnostics["turn_id"], "turn")
                self.assertTrue(fake.closed)

    def test_tools_and_compaction_reject_call_with_audit_evidence(self):
        for kind in ("commandExecution", "mcpToolCall", "contextCompaction", "fileChange"):
            with self.subTest(kind=kind):
                fake = FakeCodex([event("item/started", {"item": {"type": kind}})])
                with self.assertRaises(ApiCallError) as caught:
                    self.call(fake)
                self.assertEqual(caught.exception.diagnostics["events"][0]["params"]["item"]["type"], kind)
                self.assertTrue(fake.closed)

    def test_unknown_model_or_effort_fails_before_generation(self):
        fake = FakeCodex()
        fake.model_list.return_value.data = []
        with self.assertRaisesRegex(ApiCallError, "account catalog"):
            self.call(fake)
        self.assertEqual(fake.calls, [])

    def test_contract_output_cap_is_not_silently_ignored(self):
        with patch("crypto_quant.research.factor_mining.codex_model.AsyncCodexClient") as sdk:
            with self.assertRaisesRegex(ValueError, "output_tokens=null"):
                CodexModel().complete(self.messages, max_output_tokens=100, session_id="capped")
            sdk.assert_not_called()

    def test_preparation_timeout_closes_owned_sdk_and_is_retryable(self):
        fake = FakeCodex()
        async def stalled():
            await asyncio.sleep(10)
        fake.account_read.side_effect = stalled
        with patch("crypto_quant.research.factor_mining.codex_model.AsyncCodexClient", return_value=fake):
            with self.assertRaises(ApiCallError) as caught:
                CodexModel(timeout_seconds=1).complete(self.messages, max_output_tokens=None, session_id="timeout")
        self.assertEqual(caught.exception.diagnostics["error_type"], "TimeoutError")
        self.assertTrue(caught.exception.retryable)
        fake.account_read.assert_awaited_once()
        self.assertTrue(fake.closed)

    def test_workspace_routing_timeout_is_retryable_before_generation(self):
        fake = FakeCodex()
        fake.account_read.side_effect = InternalRpcError(-32603, "workspace routing discovery timed out")
        with self.assertRaises(ApiCallError) as caught:
            self.call(fake)
        self.assertTrue(caught.exception.retryable)
        self.assertFalse(caught.exception.diagnostics["generation_started"])
        self.assertEqual(fake.calls, [])
        self.assertTrue(fake.closed)

    def test_other_account_rpc_errors_are_not_retryable(self):
        for message in ("workspace routing discovery unauthorized (401)", "selected workspace missing from routing discovery"):
            with self.subTest(message=message):
                fake = FakeCodex()
                fake.account_read.side_effect = InternalRpcError(-32603, message)
                with self.assertRaises(ApiCallError) as caught:
                    self.call(fake)
                self.assertFalse(caught.exception.retryable)
                self.assertEqual(fake.calls, [])
                self.assertTrue(fake.closed)

    def test_timeout_after_generation_starts_is_not_retryable(self):
        fake = FakeCodex()
        fake.next_turn_notification = AsyncMock(side_effect=TimeoutError())
        with self.assertRaises(ApiCallError) as caught:
            self.call(fake)
        self.assertFalse(caught.exception.retryable)
        self.assertTrue(caught.exception.diagnostics["generation_started"])
        self.assertTrue(fake.closed)

    def test_turn_start_timeout_is_not_retryable_when_acceptance_is_unknown(self):
        fake = FakeCodex()
        fake.turn_start = AsyncMock(side_effect=TimeoutError())
        with self.assertRaises(ApiCallError) as caught:
            self.call(fake)
        self.assertFalse(caught.exception.retryable)
        self.assertTrue(caught.exception.diagnostics["generation_started"])
        self.assertTrue(fake.closed)

    def test_cancellation_closes_sdk(self):
        fake = FakeCodex()
        fake.account_read.side_effect = asyncio.CancelledError()
        with patch("crypto_quant.research.factor_mining.codex_model.AsyncCodexClient", return_value=fake):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(CodexModel()._complete(self.messages, "cancel"))
        self.assertTrue(fake.closed)

    def test_factor_cli_defaults_use_mimo_and_codex_remains_available(self):
        from crypto_quant.research.factor_mining.cli import add_arguments
        from crypto_quant.research.factor_mining.model_config import model_from_args
        parser = argparse.ArgumentParser(); add_arguments(parser)
        args = parser.parse_args(["explore", "--contract", "c.json", "--universe", "u.csv"])
        model = model_from_args(args)
        self.assertEqual(model.settings()["provider"], "xiaomi-token-plan")
        self.assertEqual(model.model, "mimo-v2.6-flash")
        args.provider = "codex"
        self.assertEqual(model_from_args(args).model, "gpt-5.6-luna")
        self.assertEqual(model_from_args(args).reasoning_effort, "max")

    def test_goal_resume_rejects_old_provider_or_changed_sdk_without_running(self):
        from crypto_quant.research.factor_mining.cli import execute
        settings = CodexModel().settings()
        for change in ({"provider": "opencode-go"}, {"sdk_version": "0.0.0"}):
            with self.subTest(change=change):
                saved = {**settings, **change}
                runner = NS(model_settings=saved)
                with patch("crypto_quant.research.factor_mining.cli.GoalRunner", return_value=runner), patch(
                        "crypto_quant.research.factor_mining.cli._run_goal") as run:
                    with self.assertRaises(ValueError):
                        execute(NS(mining_action="goal-resume", goal_dir="historical-goal"))
                    run.assert_not_called()
                self.assertEqual(saved, {**settings, **change})


if __name__ == "__main__":
    unittest.main()
