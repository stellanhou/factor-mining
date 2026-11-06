"""The Host's private evidence-check mode returns the fixed native binding."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from crypto_quant.research.factor_mining import mcp_server


def _adapter(tmp_path: Path, *, role: str = "coordinator"):
    evidence = {"status": "incomplete", "objective_met": False, "gaps": ["missing A seal"]}
    binding = SimpleNamespace(
        research_id="synthetic-research-run",
        run_root=tmp_path / "runs" / "synthetic-research-run",
        contract_sha256="a" * 64,
        goal={"goal_id": "synthetic-domain-goal", "objective": "Synthetic acceptance only."},
    )
    actions = SimpleNamespace(check_goal_evidence=lambda: evidence)
    return SimpleNamespace(binding=binding, actions=actions, role=role), evidence


def test_internal_check_returns_binding_and_recomputed_evidence(monkeypatch, capsys, tmp_path):
    adapter, evidence = _adapter(tmp_path)
    calls = []

    def load(binding_path, role):
        calls.append((binding_path, role))
        return adapter

    monkeypatch.setattr(mcp_server, "load_native_adapter", load)
    mcp_server.main([
        "--role", "coordinator", "--binding", str(tmp_path / "binding.json"),
        "--check-goal-evidence",
    ])

    result = json.loads(capsys.readouterr().out)
    assert calls == [(str(tmp_path / "binding.json"), "coordinator")]
    assert result == {
        "research_id": "synthetic-research-run",
        "run_root": str(adapter.binding.run_root),
        "contract_sha256": "a" * 64,
        "goal": adapter.binding.goal,
        "evidence": evidence,
    }


def test_internal_check_is_coordinator_only(monkeypatch, tmp_path):
    adapter, _ = _adapter(tmp_path, role="A")
    monkeypatch.setattr(mcp_server, "load_native_adapter", lambda _binding, _role: adapter)

    with pytest.raises(ValueError, match="coordinator-only"):
        mcp_server.main([
            "--role", "A", "--binding", str(tmp_path / "binding.json"),
            "--check-goal-evidence",
        ])
