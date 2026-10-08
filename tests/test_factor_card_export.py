from types import SimpleNamespace
from pathlib import Path
import json
from crypto_quant.features.factor_expressions import compile_expression
from crypto_quant.research.factor_mining.workflow import FactorMiner

def test_export_matches_multifactor_card_contract(tmp_path):
    expression = "div(perp_close, ts_mean(perp_close, 3))"
    executed = compile_expression(expression).description()
    miner = object.__new__(FactorMiner)
    miner.root = tmp_path
    miner.spec = SimpleNamespace(
        run_id="workflow-fixture",
        universe_provenance="fixed local cohort",
        data_usage_review="historical evidence is exploratory",
        admission_scheme="plan3",
        fdr_method="BH",
        fdr_alpha=0.05,
    )
    card = miner._idea_card(
        "candidate-0001",
        {
            "definition": {
                "name": "current workflow card",
                "meaning": "test meaning",
                "direction": -1,
                "hypothesis": "test hypothesis",
            },
            "executed": executed,
            "retained_horizons": [24],
            "a_decision": {"disposition": "retain"},
            "a_evaluation_ref": {"record_id": "candidate-0001-evaluation"},
        },
        {
            "retained_horizons": [24],
            "horizons": {
                "24": {"summary": {"rank_ic": {"mean": 0.1}}, "coverage": {},
                       "factor_archive": {"root": "factor_archive_v2"}},
            },
        },
        {
            "mechanism": "test mechanism",
            "falsifiers": ["test falsifier"],
            "limitations": ["test limitation"],
            "next_steps": ["test next step"],
            "conditions": ["test condition"],
        },
        {
            "eligible_for_idea_pool": True,
            "validation_status": "passed",
            "passed_horizons": [24],
            "retained_horizons": [24],
            "tracks": [],
            "tests": [],
        },
        "2026-10-02T00:00:00Z",
    )
    fixture = Path(__file__).parent / "fixtures/fm_v6_workflow_card.json"
    expected = json.loads(fixture.read_text(encoding="utf-8"))
    expected["source"]["research_directory"] = str(tmp_path.resolve())
    assert card == expected
