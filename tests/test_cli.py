import json
import runpy
import sys

import pytest
from mcp.server import MCPServer
from typer.testing import CliRunner

from llmparser.cli import app
from llmparser.corpus import CorpusError, SourceMode
from llmparser.mcp.config import SourceSpec, claude_desktop_config, load_store

runner = CliRunner()


def test_source_spec_parsing(tmp_path):
    path = tmp_path / "train.conllu"
    spec = SourceSpec.parse(str(path))
    assert (spec.name, spec.path, spec.mode, spec.default_confidence) == ("train", path.resolve(), SourceMode.READ_ONLY, None)
    spec = SourceSpec.parse(f"pred={path}:rw")
    assert (spec.name, spec.mode) == ("pred", SourceMode.READ_WRITE)
    spec = SourceSpec.parse(f"silver={path}:ro:0.8")
    assert spec.default_confidence == 0.8
    assert spec.to_arg() == f"silver={path.resolve()}:ro:0.8"
    assert SourceSpec.parse(f"{path}:rw").to_arg() == f"train={path.resolve()}:rw"
    assert SourceSpec.parse("C:/data/x.conllu:rw").path.name == "x.conllu"
    with pytest.raises(CorpusError, match="Invalid confidence"):
        SourceSpec.parse(f"{path}:ro:high")
    with pytest.raises(CorpusError, match="no path"):
        SourceSpec.parse("x=:rw")


def test_load_store_and_claude_desktop_config(train_path, pred_path):
    specs = [SourceSpec.parse(str(train_path)), SourceSpec.parse(f"pred={pred_path}:rw")]
    store = load_store(specs, autosave=False)
    assert [s.name for s in store.list_sources()] == ["train", "pred"]
    config = claude_desktop_config(specs, name="ud", read_only=True, root_key="LRoot")
    entry = config["mcpServers"]["ud"]
    assert entry["command"] == sys.executable
    assert entry["args"][:6] == ["-m", "llmparser", "mcp", "serve", "--name", "ud"]
    assert entry["args"][-3:] == ["--read-only", "--root-key", "LRoot"]


def test_mcp_config_command(train_path, tmp_path):
    result = runner.invoke(app, ["mcp", "config", "-s", str(train_path), "-s", f"{tmp_path / 'missing.conllu'}:rw"])
    assert result.exit_code == 0
    assert "does not exist" in result.stderr
    assert json.loads(result.stdout)["mcpServers"]["conllu"]["args"][-1].endswith("missing.conllu:rw")


def test_mcp_serve_command(monkeypatch, train_path, pred_path):
    calls = []
    monkeypatch.setattr(MCPServer, "run", lambda self, transport, **kwargs: calls.append((self.name, transport, kwargs)))
    result = runner.invoke(app, ["mcp", "serve", "-s", str(train_path), "--read-only"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["mcp", "serve", "-s", f"{pred_path}:rw", "--transport", "streamable-http", "--port", "9000", "--name", "ud"])
    assert result.exit_code == 0, result.output
    assert calls == [("conllu", "stdio", {}), ("ud", "streamable-http", {"host": "127.0.0.1", "port": 9000})]


def test_serve_reports_bad_sources(tmp_path):
    result = runner.invoke(app, ["mcp", "serve", "-s", str(tmp_path / "missing.conllu")])
    assert result.exit_code == 1
    assert "not found" in result.stderr
    result = runner.invoke(app, ["mcp", "serve", "-s", "x.conllu:ro:bad"])
    assert result.exit_code == 2
    assert "Invalid confidence" in result.stderr


def test_validate_command(train_path, pred_path):
    result = runner.invoke(app, ["validate", str(pred_path), "-s", str(train_path)])
    assert result.exit_code == 0, result.output
    assert "WARNING p1:2" in result.stdout
    assert result.stdout.strip().endswith("0 errors, 2 warnings.")
    pred_path.write_text("# sent_id = a\n1\ta\ta\tBAD\t_\t_\t0\troot\t_\t_\n2\tb\tb\tX\t_\t_\t0\troot\t_\t_\n\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(pred_path), "--limit", "5"])
    assert result.exit_code == 1
    assert "ERROR   a        " in result.stdout
    assert "invalid-upos" in result.stdout


def test_module_entry_point(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["llmparser", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("llmparser", run_name="__main__")
    assert exit_info.value.code == 0
    assert "Usage: llmparser" in capsys.readouterr().out
