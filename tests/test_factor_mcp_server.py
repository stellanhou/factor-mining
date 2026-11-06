import asyncio

from mcp.types import CallToolRequestParams
import pytest

from crypto_quant.research.factor_mining.mcp_server import create_server
from crypto_quant.research.factor_mining.model import RecoverableToolError
from crypto_quant.research.factor_mining.records import EvidenceIntegrityError, RecordStore
from crypto_quant.research.factor_mining.research_tools import AResearchEvidenceTools


def test_recoverable_evidence_read_error_is_returned_as_a_tool_error():
    class Adapter:
        role = "A"
        tool_definitions = [{
            "name": "read_a_evidence",
            "inputSchema": {"type": "object", "required": [], "properties": {}},
        }]

        def call(self, _name, _arguments):
            raise RecoverableToolError("no content was returned; narrow the JSON pointer to /summary")

    server = create_server(Adapter())
    handler = server.get_request_handler("tools/call")
    result = asyncio.run(handler.handler(None, CallToolRequestParams(
        name="read_a_evidence", arguments={},
    )))

    assert result.is_error is True
    assert result.content[0].text == "no content was returned; narrow the JSON pointer to /summary"


def test_invalid_a_pointer_is_recoverable_but_record_integrity_errors_fail_fast(tmp_path):
    store = RecordStore(tmp_path / "a_records")
    record_id = store.append("valid-evidence", "fact", {"value": 123})
    tools = AResearchEvidenceTools(store, {record_id})

    with pytest.raises(RecoverableToolError, match="component 'missing' is missing"):
        tools.read(record_id=record_id, pointer="/missing", offset=0, limit=1)

    (store.root / f"{record_id}.json").write_text("{", encoding="utf-8")
    with pytest.raises(EvidenceIntegrityError):
        tools.read(record_id=record_id, pointer="/value", offset=0, limit=1)
