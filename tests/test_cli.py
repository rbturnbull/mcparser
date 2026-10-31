import json
import runpy
import sys

import pytest
from mcp.server import MCPServer
from typer.testing import CliRunner

from mcparser.cli import app
from mcparser.corpus import CorpusError, SourceMode
from mcparser.mcp.config import SourceSpec, claude_desktop_config, load_store

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
    assert entry["args"][:6] == ["-m", "mcparser", "mcp", "serve", "--name", "ud"]
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
    monkeypatch.setattr(sys, "argv", ["mcparser", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("mcparser", run_name="__main__")
    assert exit_info.value.code == 0
    assert "Usage: mcparser" in capsys.readouterr().out


# ---------------------------------------------------------------- annotation commands


@pytest.fixture
def scripted_team(monkeypatch):
    """Replace the agent team with scripted models: the scorer rates everything 0.9."""
    from mcparser import cli
    from mcparser.agents.team import AgentTeam
    from scripted import scored, scripted

    created = []

    def make_team(store, target, settings):
        created.append(settings)
        return AgentTeam(store, target, settings, models={"scorer": scripted(scored(0.9)), "*": scripted()})

    monkeypatch.setattr(cli, "make_team", make_team)
    return created


def test_run_command(scripted_team, train_path, pred_path, tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text("rounds: 2\nroles:\n  head:\n    model: big\n", encoding="utf-8")
    report = tmp_path / "reports" / "report.json"
    args = [
        "run", str(pred_path), "-s", str(train_path), "-c", str(config), "--model", "m", "--model-for", "lemma=small",
        "--roles", "lemma,head", "--threshold", "0.5", "--sentence", "p2", "--report", str(report),
        "--strong-model", "s", "--efficient-model", "e", "--efficient-roles", "upos,head",
        "--strong-thinking", "low", "--efficient-thinking", "off",
        "--light-model", "h", "--light-roles", "xpos", "--light-thinking", "minimal", "--batch-sentences", "3",
    ]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    settings = scripted_team[0]
    assert (settings.model, settings.model_for("lemma"), settings.model_for("head"), settings.rounds, settings.threshold) == ("m", "small", "big", 2, 0.5)
    assert not settings.enabled("upos")
    assert (settings.strong_model, settings.efficient_model, settings.efficient_roles) == ("s", "e", ["upos", "head"])
    assert (settings.thinking_for("lemma"), settings.thinking_for("head")) == ("low", False)
    assert (settings.light_model, settings.thinking_for("xpos"), settings.batch_sentences) == ("h", "minimal", 3)
    assert "  function:" in result.stdout  # usage per model
    assert "Round 1: scored 1 sentences, 0 agent tasks" in result.stdout
    assert "Final mean confidence 0.900" in result.stdout
    assert "[round 1] scorer p2: done" in result.stderr
    assert "Rules settled" in result.stdout
    assert json.loads(report.read_text())["rounds"][0]["scored"] == 1
    log = pred_path.with_name("pred.conllu.runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 1 and json.loads(log[0])["total"]["tasks"] == 1
    assert "Conf=" in pred_path.read_text(encoding="utf-8")


def test_score_command_and_missing_sources(scripted_team, pred_path):
    result = runner.invoke(
        app, ["score", str(pred_path), "--concurrency", "2", "--efficient-model", "e", "--efficient-thinking", "on", "--batch-sentences", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "no trusted --source" in result.stderr
    assert "Round 1: scored 2 sentences, 0 agent tasks" in result.stdout
    settings = scripted_team[0]
    assert (settings.concurrency, settings.efficient_model, settings.efficient_thinking, settings.batch_sentences) == (2, "e", True, 1)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--model-for", "parsing=x"], "Expected ROLE=MODEL"),
        (["--model-for", "lemma"], "Expected ROLE=MODEL"),
        (["--roles", "lemma,parsing"], "Roles must be among"),
        (["--efficient-roles", "parsing"], "Roles must be among"),
        (["--strong-thinking", "max"], "Thinking must be"),
        (["-c", "missing.yaml"], "No such file"),
    ],
)
def test_run_rejects_bad_options(pred_path, args, message):
    result = runner.invoke(app, ["run", str(pred_path), *args])
    assert result.exit_code == 2
    assert message in result.stderr


def test_run_reports_configuration_errors(monkeypatch, pred_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["score", str(pred_path), "--model", "anthropic:claude-opus-5-5"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.stderr
    result = runner.invoke(app, ["run", str(pred_path), "--sentence", "nope"])
    assert result.exit_code == 1
    assert "No sentence 'nope'" in result.stderr


def test_init_command(tmp_path, train_path):
    raw = tmp_path / "raw.txt"
    raw.write_text("Vino al mercado.\nthe dogs\n", encoding="utf-8")
    output = tmp_path / "out.conllu"
    result = runner.invoke(app, ["init", str(raw), "-o", str(output), "-s", str(train_path)])
    assert result.exit_code == 0, result.output
    assert "Wrote 2 sentences" in result.stdout
    assert "Baseline: 1 multiword splits, 7 words analysed, 0 words not found" in result.stdout
    assert "2-3\tal" in output.read_text(encoding="utf-8")

    result = runner.invoke(app, ["init", str(raw), "-o", str(output)])
    assert result.exit_code == 1 and "use --force" in result.stderr
    result = runner.invoke(app, ["init", str(raw), "-o", str(output), "--force", "--no-baseline", "-s", str(train_path)])
    assert result.exit_code == 0
    assert "Baseline" not in result.stdout
    assert not output.with_name("out.conllu.history.jsonl").exists()

    bad = tmp_path / "bad.yaml"
    bad.write_text("- {verse: 1}\n", encoding="utf-8")
    result = runner.invoke(app, ["init", str(bad), "-o", str(tmp_path / "x.conllu")])
    assert result.exit_code == 1 and "no 'text'" in result.stderr


def test_evaluate_command(tmp_path, train_path, pred_path):
    result = runner.invoke(app, ["evaluate", str(pred_path), str(train_path)])
    assert result.exit_code == 0, result.output
    assert "0 sentences matched." in result.stdout
    assert "Missing from the predictions: t1, t2, t3" in result.stdout
    assert "Not in the gold file: p1, p2" in result.stdout
    result = runner.invoke(app, ["evaluate", str(train_path), str(train_path), "--json"])
    assert json.loads(result.stdout)["scores"][0]["f1"] == 1.0
    result = runner.invoke(app, ["evaluate", str(train_path), str(train_path)])
    assert "Missing" not in result.stdout and "Words      100.00%" in result.stdout
    result = runner.invoke(app, ["evaluate", str(tmp_path / "missing.conllu"), str(train_path)])
    assert result.exit_code == 1 and "Error" in result.stderr
    result = runner.invoke(app, ["evaluate", str(train_path), str(train_path), "--calibration"])
    assert "Accuracy of cells by confidence:" in result.stdout and "0.95-1.00" in result.stdout


def test_settings_command(tmp_path):
    result = runner.invoke(app, ["settings"])
    assert result.stdout.startswith("strong_model: anthropic:claude-opus-5-5\nefficient_model: anthropic:claude-sonnet-5-5\n")
    config = tmp_path / "s.yaml"
    config.write_text("rounds: 7\n", encoding="utf-8")
    assert "rounds: 7" in runner.invoke(app, ["settings", "-c", str(config)]).stdout


def test_init_from_conllu_and_max_sentences(tmp_path, train_path, scripted_team):
    output = tmp_path / "new" / "dirs" / "blank.conllu"  # parent directories are created
    result = runner.invoke(app, ["init", str(train_path), "-o", str(output), "--max-sentences", "2", "--no-baseline"])
    assert result.exit_code == 0, result.output
    text = output.read_text(encoding="utf-8")
    assert text.count("# sent_id") == 2 and "\tdog\t" not in text and "2-3\tdel\t_" in text
    later = tmp_path / "later.conllu"
    result = runner.invoke(app, ["init", str(train_path), "-o", str(later), "--skip", "1", "--max-sentences", "1", "--no-baseline"])
    assert result.exit_code == 0 and later.read_text(encoding="utf-8").startswith("# sent_id = t2")
    result = runner.invoke(app, ["score", str(output), "-s", str(train_path), "--max-sentences", "1"])
    assert "Round 1: scored 1 sentences" in result.stdout
    result = runner.invoke(app, ["run", str(output), "-s", str(train_path), "--sentence", "t2", "--max-sentences", "5", "--rounds", "1"])
    assert "scored 1 sentences" in result.stdout
    result = runner.invoke(app, ["evaluate", str(output), str(train_path), "--predicted-only"])
    assert "2 sentences matched." in result.stdout and "Missing" not in result.stdout
    result = runner.invoke(app, ["evaluate", str(output), str(train_path), "--annotated-only"])
    assert "0 sentences matched." in result.stdout


def test_cli_corpus_options(monkeypatch, tmp_path, train_path, scripted_team, pred_path):
    corpus = tmp_path / "text.txt"
    corpus.write_text("vino al mercado\n", encoding="utf-8")
    paths = [corpus]
    spec = f"es={paths[0]}"
    result = runner.invoke(app, ["mcp", "config", "-s", str(train_path), "--corpus", spec])
    assert json.loads(result.stdout)["mcpServers"]["conllu"]["args"][-2:] == ["--corpus", f"es={paths[0].resolve()}"]
    result = runner.invoke(app, ["mcp", "config", "-s", str(train_path), "--corpus", "es="])
    assert result.exit_code == 2 and "--corpus" in result.output
    result = runner.invoke(app, ["run", str(pred_path), "-s", str(train_path), "--corpus", spec, "--no-corpus-tool", "--sentence", "p2"])
    assert result.exit_code == 0, result.output
    assert scripted_team[-1].corpus_tool is False
    assert "Indexing corpus 'es'" in result.stderr and "words," in result.stderr
    result = runner.invoke(app, ["score", str(pred_path), "-s", str(train_path), "--corpus", spec, "--sentence", "p2"])
    assert result.exit_code == 0, result.output
    served = []
    monkeypatch.setattr(MCPServer, "run", lambda self, transport, **kwargs: served.append(self.instructions))
    result = runner.invoke(app, ["mcp", "serve", "-s", str(train_path), "--corpus", spec])
    assert result.exit_code == 0, result.output
    assert "Text corpora" in served[0]
