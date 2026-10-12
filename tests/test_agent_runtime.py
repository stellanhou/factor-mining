"""Migration contracts: exact messages, strict parsing and natural graph termination."""
import json
import time
from unittest.mock import Mock

from langgraph.graph import StateGraph, START, END
import pytest

from crypto_quant.research.factor_mining.model import ModelReply
from crypto_quant.research.factor_mining.runtime import invoke_role, GRAPH_CONFIG
from crypto_quant.research.progress import ProgressLog
from crypto_quant.research.factor_mining.structured_output import ReplyModel, parse_reply, OutputFormatError


def test_prompt_chain_keeps_literal_evidence_and_role_order():
    messages = [{"role": "system", "content": "合同 {not_a_template}"},
                {"role": "user", "content": '{"evidence":"原文\\n{foo}"}'},
                {"role": "assistant", "content": '{"bad":true}'},
                {"role": "user", "content": "纠正格式，保留原证据"}]
    model = Mock()
    expected = ModelReply('{"ok":true}', {"tokens": 3}, "fixture")
    model.complete.return_value = expected
    assert invoke_role(model, messages, max_output_tokens=None, session_id="unchanged") is expected
    model.complete.assert_called_once_with(messages, max_output_tokens=None, session_id="unchanged")


def test_graph_does_not_impose_a_new_research_round_cap():
    graph = StateGraph(dict)
    graph.add_node("step", lambda state: {"n": state["n"] + 1})
    graph.add_edge(START, "step")
    graph.add_conditional_edges("step", lambda state: END if state["n"] == 1100 else "step", [END, "step"])
    assert graph.compile().invoke({"n": 0}, config=GRAPH_CONFIG)["n"] == 1100


def test_langchain_parser_keeps_strict_json_contract():
    class Reply(ReplyModel):
        accepted: bool
    assert parse_reply('{"accepted":true}', Reply) == {"accepted": True}
    for content in ['{"accepted":"true"}', '```json\n{"accepted":true}\n```', '{"accepted":true']:
        with pytest.raises(OutputFormatError):
            parse_reply(content, Reply)


def test_progress_is_visible_live_and_kept_outside_frozen_run(tmp_path, capsys):
    run = tmp_path / "research-1"
    run.mkdir()
    progress = ProgressLog.for_run(run)
    progress.heartbeat_seconds = 0.01
    with progress.span("strategy.model_call", heartbeat=True, role="design", attempt=1):
        time.sleep(0.03)
    events = [json.loads(line) for line in progress.path.read_text().splitlines()]
    assert progress.path == tmp_path / "research-1.progress.jsonl"
    assert [event["event"] for event in events][0] == "started"
    assert "heartbeat" in [event["event"] for event in events]
    assert events[-1]["event"] == "completed"
    assert events[-1]["elapsed_seconds"] > 0
    output = capsys.readouterr().err
    assert "开始 strategy.model_call" in output
    assert "进行中 strategy.model_call" in output
    assert "完成 strategy.model_call" in output


def test_progress_failure_does_not_save_exception_message(tmp_path, capsys):
    progress = ProgressLog(tmp_path / "progress.jsonl")
    with pytest.raises(RuntimeError, match="sensitive content"):
        with progress.span("factor.model_call", role="calculator"):
            raise RuntimeError("sensitive content")
    record = json.loads(progress.path.read_text().splitlines()[-1])
    assert record["event"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert "sensitive content" not in progress.path.read_text()
    assert "sensitive content" not in capsys.readouterr().err
