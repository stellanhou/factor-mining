"""Contracts derived from the real factor and Goal role schemas."""
import pytest

from crypto_quant.research.factor_mining.structured_output import OutputFormatError, validate_schema
from crypto_quant.research.factor_mining.agent_evaluator import CANDIDATE_DECISION_SCHEMA
from crypto_quant.research.factor_mining.goal import TASK_SCHEMA


def test_factor_and_goal_schemas_report_independent_errors_together():
    for schema, value, fields in (
        (CANDIDATE_DECISION_SCHEMA, {"disposition": "decide", "continue_optimization": "false"},
         ["disposition", "continue_optimization", "candidate_id"]),
        (TASK_SCHEMA, {"action": "decide", "dependencies": [{"field": "not-a-field", "min_valid_rows": 0}]},
         ["action", "field", "min_valid_rows", "task"]),
    ):
        with pytest.raises(OutputFormatError) as exc:
            validate_schema(value, schema, [])
        for field in fields:
            assert field in str(exc.value)


def test_nested_limits_types_and_extra_field_audit():
    schema = {"type": "object", "properties": {
        "rows": {"type": "array", "minItems": 1, "maxItems": 2, "items": {
            "type": "object", "properties": {"count": {"type": "integer", "minimum": 1}}, "required": ["count"]}},
        "weight": {"type": "number", "exclusiveMinimum": 0}}, "required": ["rows", "weight"]}
    for value in ({"rows": [], "weight": float("inf")},
                  {"rows": [{"count": True}], "weight": "1"},
                  {"rows": [{"count": 0}], "weight": 0}):
        with pytest.raises(OutputFormatError) as exc:
            validate_schema(value, schema, [])
        assert "rows" in str(exc.value) and "weight" in str(exc.value)
    extras = []
    result = validate_schema({"rows": [{"count": 1, "note": "ignored"}], "weight": 1, "other": 2}, schema, extras)
    assert result == {"rows": [{"count": 1}], "weight": 1}
    assert extras == ["/rows/0/note", "/other"]
