"""Dependency boundary required by the DeepSeek Harness entry point."""

import tomllib
import subprocess
import sys
from pathlib import Path


def test_codex_sdk_is_optional_for_the_dsh_install():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "openai-codex==0.154.0" not in project["dependencies"]
    assert project["optional-dependencies"]["codex"] == ["openai-codex==0.154.0"]


def test_dsh_domain_modules_import_without_the_codex_sdk():
    script = """
import importlib
import sys
sys.modules['openai_codex'] = None
for module in (
    'crypto_quant.research.factor_mining.records',
    'crypto_quant.research.factor_mining.model_config',
    'crypto_quant.research.factor_mining.mcp_server',
):
    importlib.import_module(module)
from crypto_quant.research.factor_mining.model_config import _codex_model
try:
    _codex_model()
except RuntimeError as error:
    assert 'install factor-mining[codex]' in str(error)
else:
    raise AssertionError('Codex selection must explain the missing optional SDK')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
