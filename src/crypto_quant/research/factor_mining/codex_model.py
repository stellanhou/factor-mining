"""Official Codex Python SDK adapter using the operator's ChatGPT subscription."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from openai_codex import CodexConfig, Thread
from openai_codex.async_client import AsyncCodexClient
from openai_codex.client import CodexClient
from openai_codex.errors import CodexError, InternalRpcError
from openai_codex.generated.v2_all import (
    ConfigReadResponse,
    ItemCompletedNotification,
    ItemStartedNotification,
    ReasoningEffort,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
    TurnStatus,
)

from .contracts import require, text
from .model import ApiCallError, ModelReply, RecoverableToolError
from .structured_output import validate_schema


DEFAULT_MODEL = "gpt-5.6-luna"
PROVIDER = "codex-sdk-chatgpt"


def _config(directory, codex_bin=None, *, service_tier=None, experimental_api=None):
    # These are process-local overrides: the user's Codex settings and login stay intact.
    # Research evidence is supplied explicitly. Codex may retain its global AGENTS.md;
    # project documents, personal memories and research-external tools are disabled.
    disabled = ("shell_tool", "unified_exec", "apply_patch_freeform", "view_image", "code_mode",
                "code_mode_host", "multi_agent", "multi_agent_v2", "apps", "plugins", "browser_use",
                "computer_use", "image_generation", "memories", "memory_tool", "hooks", "codex_hooks",
                "plugin_hooks", "skill_search", "tool_suggest", "workspace_dependencies", "goals")
    overrides = (
        'model_provider="openai"', 'forced_login_method="chatgpt"',
        'chatgpt_base_url="https://chatgpt.com/backend-api"',
        'agents.enabled=false', 'web_search="disabled"',
        'project_doc_max_bytes=0', 'skills.include_instructions=false',
        'features.skip_host_skill_discovery=true', 'developer_instructions=""',
        *(f"features.{name}=false" for name in disabled),
    )
    if service_tier is not None:
        overrides += ('features.fast_mode=true',)
    options = {"cwd": directory, "config_overrides": overrides,
               "env": {"OPENAI_API_KEY": "", "CODEX_API_KEY": ""}}
    if codex_bin is not None:
        options["codex_bin"] = codex_bin
    if experimental_api is not None:
        options["experimental_api"] = experimental_api
    return CodexConfig(**options)


def _codex_binary(path):
    require(isinstance(path, (str, os.PathLike)), "codex_bin must be a filesystem path")
    raw_path = os.fspath(path)
    require(isinstance(raw_path, str) and bool(raw_path.strip()), "codex_bin must be a nonempty path")
    try:
        binary = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"codex_bin does not identify an existing file: {raw_path}") from exc
    require(binary.is_file() and os.access(binary, os.X_OK),
            "codex_bin must identify an executable file")
    try:
        result = subprocess.run([str(binary), "--version"], check=True, capture_output=True,
                                text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"could not read version from codex_bin: {binary}") from exc
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"(?<![\w.])(?:v)?(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)(?![\w.])", output)
    require(match is not None, f"codex_bin --version returned no parseable version: {binary}")
    return str(binary), match.group(1)


async def _subscription(codex):
    await codex.start()
    await codex.initialize()
    account = (await codex.account_read()).account
    require(account is not None and account.root.type == "chatgpt",
            "Codex requires a ChatGPT subscription login; run codex login first (API keys are not used)")


async def _thread_config(codex):
    # Empty tables merge with user settings in Codex; explicitly disable each inherited
    # server/plugin before starting a thread. Never print or persist their configuration.
    effective = await codex.request("config/read", {"includeLayers": False}, response_model=ConfigReadResponse)
    config = effective.config.model_dump(mode="json")
    return {section: {name: {"enabled": False} for name in config.get(section, {})}
            for section in ("mcp_servers", "plugins")}


def _subscription_sync(codex):
    codex.start()
    codex.initialize()
    account = codex.account_read().account
    require(account is not None and account.root.type == "chatgpt",
            "Codex requires a ChatGPT subscription login; run codex login first (API keys are not used)")


def _thread_config_sync(codex):
    effective = codex.request("config/read", {"includeLayers": False}, response_model=ConfigReadResponse)
    config = effective.config.model_dump(mode="json")
    return {section: {name: {"enabled": False} for name in config.get(section, {})}
            for section in ("mcp_servers", "plugins")}


def _validate_dynamic_tools(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    require(isinstance(tools, list) and bool(tools), "Codex tool call requires at least one tool")
    by_name = {}
    native_specs = []
    for tool in tools:
        require(isinstance(tool, dict)
                and set(tool) == {"type", "name", "description", "inputSchema"},
                "Codex dynamic tools require type, name, description, and inputSchema")
        require(tool["type"] == "function", "only native function dynamic tools are supported")
        name = tool["name"]
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name) is not None,
                "dynamic tool name is invalid")
        require(name not in by_name, "dynamic tool names must be unique")
        require(isinstance(tool["description"], str) and bool(tool["description"].strip()),
                "dynamic tool description must be nonempty")
        schema = tool["inputSchema"]
        require(isinstance(schema, dict) and schema.get("type") == "object",
                "dynamic tool inputSchema must describe an object")
        encoded_schema = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        native = {"type": "function", "name": name, "description": tool["description"],
                  "inputSchema": json.loads(encoded_schema)}
        by_name[name] = native
        native_specs.append(native)
    return native_specs, by_name


class CodexModel:
    def __init__(self, model=DEFAULT_MODEL, *, reasoning_effort="max", timeout_seconds=300,
                 codex_bin=None, service_tier=None):
        self.model = text(model, "model")
        require(reasoning_effort in {"low", "medium", "high", "xhigh", "max"}, "unsupported reasoning effort")
        require(type(timeout_seconds) is int and timeout_seconds > 0, "timeout must be positive")
        require(service_tier in {None, "priority"}, "supported service tier is priority (Fast)")
        self.service_tier = service_tier
        self.reasoning_effort, self.timeout_seconds = reasoning_effort, timeout_seconds
        self.codex_bin, self.codex_runtime_version = (
            _codex_binary(codex_bin) if codex_bin is not None else (None, None))

    def settings(self):
        settings = {"provider": PROVIDER, "model": self.model, "reasoning_effort": self.reasoning_effort,
                    "timeout_seconds": self.timeout_seconds, "sdk_version": version("openai-codex"),
                    "runtime_version": (self.codex_runtime_version if self.codex_bin is not None
                                        else version("openai-codex-cli-bin"))}
        if self.codex_bin is not None:
            settings["codex_bin"] = self.codex_bin
        if self.service_tier is not None:
            settings["service_tier"] = self.service_tier
        return settings

    def available_models(self):
        return asyncio.run(self._models())

    async def _models(self):
        with tempfile.TemporaryDirectory(prefix="quant-codex-") as directory:
            codex = AsyncCodexClient(_config(directory, self.codex_bin, service_tier=self.service_tier))
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await _subscription(codex)
                    catalog = await codex.model_list()
                    return {"provider": PROVIDER, "authentication": "chatgpt",
                            **catalog.model_dump(mode="json")}
            finally:
                await codex.close()

    def complete(self, messages, *, max_output_tokens, session_id):
        # Codex turns have no per-call output token cap. Never silently ignore a contract cap.
        require(max_output_tokens is None, "Codex SDK requires output_tokens=null; per-call output caps are unsupported")
        require(bool(messages) and messages[0]["role"] == "system"
                and all(m["role"] in {"user", "assistant"} and isinstance(m["content"], str)
                        for m in messages[1:]) and isinstance(messages[0]["content"], str)
                and messages[-1]["role"] == "user", "invalid Codex conversation")
        return asyncio.run(self._complete(messages, session_id))

    def complete_with_tools(self, messages: list[dict[str, str]], *,
                            tools: list[dict[str, Any]],
                            tool_handler: Callable[[str, dict[str, Any]], Any],
                            max_output_tokens: int | None, session_id: str,
                            output_schema: dict[str, Any]) -> ModelReply:
        require(max_output_tokens is None,
                "Codex SDK requires output_tokens=null; per-call output caps are unsupported")
        require(bool(messages) and messages[0]["role"] == "system"
                and all(m["role"] in {"user", "assistant"} and isinstance(m["content"], str)
                        for m in messages[1:]) and isinstance(messages[0]["content"], str)
                and messages[-1]["role"] == "user", "invalid Codex conversation")
        require(callable(tool_handler), "tool_handler must be callable")
        dynamic_tools, tools_by_name = _validate_dynamic_tools(tools)
        require(isinstance(output_schema, dict) and output_schema.get("type") == "object",
                "output_schema must describe a JSON object")
        json.dumps(output_schema, ensure_ascii=False, allow_nan=False)
        return self._complete_with_tools(messages, dynamic_tools, tools_by_name, tool_handler,
                                         session_id, output_schema)

    def _complete_with_tools(self, messages, tools, tools_by_name, tool_handler,
                             session_id, output_schema):
        trace = {"session_id": session_id, "settings": self.settings(), "events": [],
                 "tool_calls": [], "generation_started": False}
        with tempfile.TemporaryDirectory(prefix="quant-codex-tools-") as directory:
            timeout_state = {"expired": False, "finished": False}
            timeout_lock = threading.Lock()
            client_close_lock = threading.Lock()
            client_closed = {"value": False}
            trace_lock = threading.Lock()
            thread_state = {"thread_id": None, "turn_id": None, "pending_turn_ids": set()}
            thread_state_lock = threading.Lock()
            timeout_cleanup_errors = []
            client = None

            def dispatch_server_request(method, params):
                if method != "item/tool/call":
                    raise RuntimeError(f"unsupported Codex server request: {method}")
                require(isinstance(params, dict), "dynamic tool request params must be an object")
                required = {"arguments", "callId", "threadId", "tool", "turnId"}
                require(required <= set(params), "dynamic tool request is missing required fields")
                require(isinstance(params["callId"], str) and bool(params["callId"]),
                        "dynamic tool callId must be nonempty")
                require(isinstance(params["threadId"], str) and isinstance(params["turnId"], str),
                        "dynamic tool request threadId and turnId must be strings")
                require(params.get("namespace") in (None, ""),
                        "namespaced dynamic tools are not configured for this call")
                with thread_state_lock:
                    require(params["threadId"] == thread_state["thread_id"],
                            "dynamic tool request belongs to another thread")
                    expected_turn = thread_state["turn_id"]
                    if expected_turn is None:
                        thread_state["pending_turn_ids"].add(params["turnId"])
                    else:
                        require(params["turnId"] == expected_turn,
                                "dynamic tool request belongs to another turn")
                name = params["tool"]
                require(isinstance(name, str) and name in tools_by_name,
                        "Codex requested an unregistered dynamic tool")
                arguments = params["arguments"]
                require(isinstance(arguments, dict), "dynamic tool arguments must be an object")
                extra_fields = []
                parsed_arguments = validate_schema(
                    arguments, tools_by_name[name]["inputSchema"], extra_fields
                )
                require(not extra_fields, "dynamic tool arguments contain undeclared fields")
                call = {"call_id": params["callId"], "tool": name,
                        "arguments": parsed_arguments}
                with trace_lock:
                    trace["tool_calls"].append(call)

                def timeout_tool_call():
                    message = (f"Codex dynamic-tool call exceeded the {self.timeout_seconds}s turn deadline")
                    with trace_lock:
                        call["error_type"] = "TimeoutError"
                        call["error"] = message
                    raise TimeoutError(message)

                worker_result = {}
                worker_finished = threading.Event()

                def run_tool_handler():
                    try:
                        result = tool_handler(name, parsed_arguments)
                        result_text = json.dumps(result, ensure_ascii=False, allow_nan=False,
                                                 separators=(",", ":"))
                        worker_result["result_text"] = result_text
                        worker_result["normalized_result"] = json.loads(result_text)
                    except BaseException as exc:
                        worker_result["error"] = exc
                    finally:
                        worker_result["completed_at"] = time.monotonic()
                        worker_finished.set()

                worker = threading.Thread(target=run_tool_handler,
                                          name=f"codex-tool-{params['callId']}", daemon=True)
                worker.start()
                if not worker_finished.wait(max(0.0, deadline - time.monotonic())):
                    timeout_tool_call()
                if worker_result["completed_at"] > deadline or timeout_state["expired"]:
                    timeout_tool_call()
                try:
                    if "error" in worker_result:
                        raise worker_result["error"]
                except RecoverableToolError as exc:
                    with trace_lock:
                        call["error_type"] = type(exc).__name__
                        call["error"] = str(exc)
                    require(bool(str(exc)), "recoverable tool errors must explain the correction")
                    return {"success": False,
                            "contentItems": [{"type": "inputText", "text": str(exc)}]}
                except Exception as exc:
                    with trace_lock:
                        call["error_type"] = type(exc).__name__
                        call["error"] = str(exc)
                    raise
                result_text = worker_result["result_text"]
                normalized_result = worker_result["normalized_result"]
                with trace_lock:
                    call["result"] = normalized_result
                return {"success": True,
                        "contentItems": [{"type": "inputText", "text": result_text}]}

            client = CodexClient(
                _config(directory, self.codex_bin, service_tier=self.service_tier,
                        experimental_api=True),
                approval_handler=dispatch_server_request,
            )

            def close_client():
                with client_close_lock:
                    if client_closed["value"]:
                        return
                    client.close()
                    client_closed["value"] = True

            def expire():
                with timeout_lock:
                    if timeout_state["finished"]:
                        return
                    timeout_state["expired"] = True
                try:
                    close_client()
                except Exception as exc:
                    timeout_cleanup_errors.append(f"{type(exc).__name__}: {exc}")

            deadline = time.monotonic() + self.timeout_seconds
            timer = threading.Timer(self.timeout_seconds, expire)
            timer.daemon = True
            timer.start()

            def check_deadline():
                with timeout_lock:
                    if time.monotonic() >= deadline:
                        timeout_state["expired"] = True
                    expired = timeout_state["expired"]
                if expired:
                    close_client()
                    raise TimeoutError(f"Codex dynamic-tool turn exceeded {self.timeout_seconds}s")

            try:
                check_deadline()
                _subscription_sync(client)
                check_deadline()
                catalog = client.model_list()
                selected = next((model for model in catalog.data if model.model == self.model), None)
                require(selected is not None, "requested model is not in the Codex account catalog")
                require(any(item.reasoning_effort.value == self.reasoning_effort
                            for item in selected.supported_reasoning_efforts),
                        "model does not support requested reasoning effort")
                if self.service_tier is not None:
                    require(selected.service_tiers is not None
                            and any(tier.id == self.service_tier for tier in selected.service_tiers),
                            "model does not support requested Fast service tier")
                check_deadline()
                instructions = (messages[0]["content"] + "\n\n"
                    "You are a bounded research role. Use only the supplied conversation and the "
                    "registered read-only A-evidence tools. Do not use files, external sources, "
                    "other sessions, or any unregistered tool. The user provides a JSON "
                    "conversation transcript with user/assistant roles. Return only the requested "
                    "JSON object after any needed evidence reads.")
                prompt = json.dumps(messages[1:], ensure_ascii=False)
                trace["request"] = {"base_instructions": instructions, "input": prompt,
                                    "dynamic_tools": tools, "output_schema": output_schema}
                thread_options = {"model": self.model, "modelProvider": "openai", "cwd": directory,
                    "baseInstructions": instructions, "developerInstructions": "", "ephemeral": True,
                    "sandbox": "read-only", "approvalPolicy": "never",
                    "config": _thread_config_sync(client), "dynamicTools": tools}
                if self.service_tier is not None:
                    thread_options["serviceTier"] = self.service_tier
                started = client.thread_start(thread_options)
                require(started.model == self.model and started.model_provider == "openai",
                        "Codex started with an unexpected model/provider")
                if self.service_tier is not None:
                    require(started.service_tier == self.service_tier,
                            "Codex did not accept the requested Fast service tier")
                    trace["accepted_service_tier"] = started.service_tier
                trace["thread_id"] = started.thread.id
                with thread_state_lock:
                    thread_state["thread_id"] = started.thread.id
                check_deadline()
                trace["generation_started"] = True
                turn = Thread(client, started.thread.id).turn(
                    prompt,
                    effort=ReasoningEffort(self.reasoning_effort),
                    output_schema=output_schema,
                    service_tier=self.service_tier,
                )
                trace["turn_id"] = turn.id
                with thread_state_lock:
                    pending_turn_ids = thread_state["pending_turn_ids"]
                    require(not pending_turn_ids or pending_turn_ids == {turn.id},
                            "dynamic tool request belongs to another turn")
                    thread_state["turn_id"] = turn.id
                completed = None
                usage = None
                answer = None
                items = []
                stream = turn.stream()
                try:
                    for event in stream:
                        check_deadline()
                        payload_object = event.payload
                        if hasattr(payload_object, "model_dump"):
                            payload = payload_object.model_dump(mode="json", by_alias=True)
                        else:
                            payload = {"params": payload_object.params}
                        trace["events"].append({"method": event.method, "params": payload})
                        if isinstance(payload_object, (ItemStartedNotification,
                                                       ItemCompletedNotification)):
                            item = payload["item"]
                            require(item["type"] in {"userMessage", "agentMessage", "reasoning",
                                                     "dynamicToolCall", "contextCompaction"},
                                    "Codex attempted an unregistered tool or unsupported item")
                            if isinstance(payload_object, ItemCompletedNotification):
                                items.append(item)
                                if item["type"] == "agentMessage" \
                                        and item.get("phase") in {None, "final_answer"}:
                                    answer = item.get("text")
                        elif isinstance(payload_object, ThreadTokenUsageUpdatedNotification):
                            usage = payload["tokenUsage"]
                        elif isinstance(payload_object, TurnCompletedNotification):
                            require(payload_object.turn.id == turn.id,
                                    "Codex completed an unexpected turn")
                            completed = payload_object.turn
                finally:
                    stream.close()
                check_deadline()
                require(completed is not None, "Codex did not return a completed turn")
                require(completed.status == TurnStatus.completed,
                        f"Codex turn completed with status {completed.status.value}")
                require(usage is not None, "Codex completed without token usage")
                require(bool(answer and answer.strip()), "Codex completed without a final answer")
                trace["items"] = items
                with timeout_lock:
                    if timeout_state["expired"] or time.monotonic() >= deadline:
                        timeout_state["expired"] = True
                        raise TimeoutError(
                            f"Codex dynamic-tool turn exceeded {self.timeout_seconds}s"
                        )
                    timeout_state["finished"] = True
                return ModelReply(answer, usage,
                                  self.model, "stop", self.reasoning_effort, trace)
            except Exception as exc:
                timed_out = timeout_state["expired"]
                trace["error_type"] = "TimeoutError" if timed_out else type(exc).__name__
                trace["error"] = (f"Codex dynamic-tool turn exceeded {self.timeout_seconds}s"
                                  if timed_out else str(exc))
                if timeout_cleanup_errors:
                    trace["timeout_cleanup_errors"] = list(timeout_cleanup_errors)
                retryable = not trace["generation_started"] and (
                    timed_out or isinstance(exc, TimeoutError)
                    or isinstance(exc, InternalRpcError)
                    and exc.message == "workspace routing discovery timed out")
                raise ApiCallError(f"Codex SDK call failed: {trace['error_type']}: {trace['error']}",
                                   retryable=retryable, diagnostics=trace) from exc
            finally:
                timer.cancel()
                close_client()
                if timer.is_alive() and threading.current_thread() is not timer:
                    timer.join(timeout=2.5)

    async def _complete(self, messages, session_id):
        trace = {"session_id": session_id, "settings": self.settings(), "events": [],
                 "generation_started": False}
        with tempfile.TemporaryDirectory(prefix="quant-codex-") as directory:
            codex = AsyncCodexClient(_config(directory, self.codex_bin, service_tier=self.service_tier))
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await _subscription(codex)
                    catalog = await codex.model_list()
                    selected = next((m for m in catalog.data if m.model == self.model), None)
                    require(selected is not None, "requested model is not in the Codex account catalog")
                    require(any(r.reasoning_effort.value == self.reasoning_effort
                                for r in selected.supported_reasoning_efforts),
                            "model does not support requested reasoning effort")
                    if self.service_tier is not None:
                        require(selected.service_tiers is not None
                                and any(tier.id == self.service_tier for tier in selected.service_tiers),
                                "model does not support requested Fast service tier")
                    # Fresh ephemeral threads isolate A/B, roles and corrections. The transcript
                    # is lossless; the SDK accepts turn text, not an arbitrary chat-message array.
                    instructions = (messages[0]["content"] + "\n\n"
                        "You are a bounded research role. Use only the supplied conversation. "
                        "Do not use tools, files, external sources or other sessions. "
                        "The user provides a JSON conversation transcript with user/assistant roles. "
                        "Answer its final user turn with only the requested JSON object.")
                    prompt = json.dumps(messages[1:], ensure_ascii=False)
                    trace["request"] = {"base_instructions": instructions, "input": prompt}
                    thread_options = {"model": self.model, "modelProvider": "openai",
                        "cwd": directory, "baseInstructions": instructions, "developerInstructions": "",
                        "ephemeral": True, "sandbox": "read-only", "approvalPolicy": "never",
                        "config": await _thread_config(codex)}
                    turn_options = {"effort": self.reasoning_effort}
                    if self.service_tier is not None:
                        thread_options["serviceTier"] = self.service_tier
                        turn_options["serviceTier"] = self.service_tier
                    started = await codex.thread_start(thread_options)
                    require(started.model == self.model and started.model_provider == "openai",
                            "Codex started with an unexpected model/provider")
                    if self.service_tier is not None:
                        require(started.service_tier == self.service_tier,
                                "Codex did not accept the requested Fast service tier")
                        trace["accepted_service_tier"] = started.service_tier
                    trace["thread_id"] = started.thread.id
                    trace["generation_started"] = True
                    turn = await codex.turn_start(started.thread.id, prompt, turn_options)
                    trace["turn_id"] = turn.turn.id
                    answer, usage, completed = None, None, None
                    while completed is None:
                        event = await codex.next_turn_notification(turn.turn.id)
                        payload = event.payload.model_dump(mode="json", by_alias=True)
                        trace["events"].append({"method": event.method, "params": payload})
                        if event.method in {"item/started", "item/completed"}:
                            item = payload["item"]
                            require(item["type"] in {"userMessage", "agentMessage", "reasoning"},
                                    "Codex attempted a tool or context compaction; research call rejected")
                            if event.method == "item/completed" and item["type"] == "agentMessage":
                                if item.get("phase") in {None, "final_answer"}:
                                    answer = item["text"]
                        elif event.method == "thread/tokenUsage/updated":
                            usage = payload["tokenUsage"]
                        elif event.method == "turn/completed":
                            completed = event.payload.turn
                    require(completed is not None, "Codex did not return a completed turn")
                    if completed.status != TurnStatus.completed:
                        raise ApiCallError("Codex turn did not complete", retryable=False, diagnostics=trace)
                    require(usage is not None, "Codex completed without token usage")
                    require(bool(answer and answer.strip()), "Codex completed without a final answer")
                    return ModelReply(answer, usage, self.model, "stop", self.reasoning_effort, trace)
            except (TimeoutError, CodexError, ValueError) as exc:
                # The gateway owns bounded retries. Preparation timeouts are safe to repeat;
                # once turn/start is attempted, generation may already have begun remotely.
                trace["error_type"] = type(exc).__name__
                trace["error"] = str(exc)
                retryable = not trace["generation_started"] and (
                    isinstance(exc, TimeoutError) or isinstance(exc, InternalRpcError)
                    and exc.message == "workspace routing discovery timed out")
                raise ApiCallError(f"Codex SDK call failed: {type(exc).__name__}: {exc}",
                                   retryable=retryable, diagnostics=trace) from exc
            finally:
                await codex.close()
