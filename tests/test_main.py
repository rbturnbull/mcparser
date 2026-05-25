from pathlib import Path
from types import SimpleNamespace

import llmloader
from typer.testing import CliRunner

from llmparser.database import load_database
from llmparser.main import app
from llmparser.io import conllu_to_dict


class DummyDatabase:
    def __init__(self, examples):
        self._examples = list(examples)

    def __call__(self, _sentence: str, k: int = 100):
        return self._examples[:k]


def test_db_command_builds_database(tmp_path: Path, patch_embeddings_and_chroma) -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"
    db_path = tmp_path / "db"

    runner = CliRunner()
    result = runner.invoke(app, ["db", str(conllu_path), str(db_path)])

    assert result.exit_code == 0

    
    expected_docs = conllu_to_dict(conllu_path)
    loaded_db = load_database(db_path)
    assert loaded_db.docs_count == len(expected_docs)


def test_convert_to_yaml_command_writes_file(tmp_path: Path) -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"
    yaml_path = tmp_path / "out.yml"

    runner = CliRunner()
    result = runner.invoke(app, ["convert-to-yaml", str(conllu_path), str(yaml_path)])

    assert result.exit_code == 0
    assert yaml_path.exists()
    assert "'5.7':\n" in yaml_path.read_text()


def test_parse_command_writes_conllu(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.yml"
    output_path = tmp_path / "out.yml"
    input_path.write_text("item1:\n  text: Hello world.\n")
    examples = [SimpleNamespace(metadata={"conllu": "1\tHello\t_\tINTJ\t_\t_\t0\troot\t_\t_"})]

    original_load = llmloader.load
    monkeypatch.setattr(llmloader, "load", lambda *_args, **_kwargs: original_load("dummy"))
    monkeypatch.setattr(
        "llmparser.main.load_database",
        lambda *_args, **_kwargs: DummyDatabase(examples),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["parse", str(input_path), "--db", str(tmp_path / "db"), "--output", str(output_path), "--model", "dummy"],
    )

    assert result.exit_code == 0
    output_text = output_path.read_text()
    assert "conllu:" in output_text
    assert "Hello world." in output_text


def test_correct_command_writes_correction_yaml(tmp_path: Path, monkeypatch) -> None:
    prediction_path = tmp_path / "predicted.conllu"
    output_path = tmp_path / "corrected.yml"
    prediction_path.write_text(
        """# sent_id = s1
# text = Hello world.
1	Hello	hello	INTJ	_	_	0	root	_	_
"""
    )
    examples = [SimpleNamespace(metadata={"conllu": "1\tHello\thello\tINTJ\t_\t_\t0\troot\t_\t_"})]

    monkeypatch.setattr(
        "llmparser.main.load_database",
        lambda *_args, **_kwargs: DummyDatabase(examples),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["correct", str(prediction_path), "--db", str(tmp_path / "db"), "--output", str(output_path)],
    )

    assert result.exit_code == 0
    output_text = output_path.read_text()
    assert "s1:" in output_text
    assert "audit:" in output_text
    assert "conllu:" in output_text


def test_correct_command_can_run_rule_based_lemma_checker(tmp_path: Path, monkeypatch) -> None:
    prediction_path = tmp_path / "predicted.conllu"
    output_path = tmp_path / "corrected.yml"
    prediction_path.write_text(
        """# sent_id = s1
# text = ἐδόκεε
1	ἐδόκεε	ἐδόκεε	VERB	V-	_	0	root	_	_
"""
    )
    examples = [
        {
            "sent_id": "train1",
            "conllu": """# sent_id = train1
# text = ἐδόκεε
1	ἐδόκεε	δοκέω	VERB	V-	_	0	root	_	_
""",
        },
        {
            "sent_id": "train2",
            "conllu": """# sent_id = train2
# text = ἐδόκεε
1	ἐδόκεε	δοκέω	VERB	V-	_	0	root	_	_
""",
        },
    ]

    monkeypatch.setattr(
        "llmparser.main.load_database",
        lambda *_args, **_kwargs: DummyDatabase(examples),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "correct",
            str(prediction_path),
            "--db",
            str(tmp_path / "db"),
            "--output",
            str(output_path),
            "--lemma-checker",
            "rule",
            "--n-examples",
            "2",
        ],
    )

    assert result.exit_code == 0
    output_text = output_path.read_text()
    assert "δοκέω" in output_text
    assert "status: patched" in output_text


def test_correct_command_can_run_rule_based_pos_checker(tmp_path: Path, monkeypatch) -> None:
    prediction_path = tmp_path / "predicted.conllu"
    output_path = tmp_path / "corrected.yml"
    prediction_path.write_text(
        """# sent_id = s1
# text = Hello
1	Hello	hello	NOUN	NN	_	0	root	_	_
"""
    )
    examples = [
        {
            "sent_id": "train1",
            "conllu": """# sent_id = train1
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
""",
        },
        {
            "sent_id": "train2",
            "conllu": """# sent_id = train2
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
""",
        },
    ]

    monkeypatch.setattr(
        "llmparser.main.load_database",
        lambda *_args, **_kwargs: DummyDatabase(examples),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "correct",
            str(prediction_path),
            "--db",
            str(tmp_path / "db"),
            "--output",
            str(output_path),
            "--pos-checker",
            "rule",
            "--n-examples",
            "2",
        ],
    )

    assert result.exit_code == 0
    output_text = output_path.read_text()
    assert "INTJ" in output_text
    assert "Uh" in output_text
    assert "status: patched" in output_text


def test_correct_command_can_run_rule_based_feats_checker(tmp_path: Path, monkeypatch) -> None:
    prediction_path = tmp_path / "predicted.conllu"
    output_path = tmp_path / "corrected.yml"
    prediction_path.write_text(
        """# sent_id = s1
# text = λόγον
1	λόγον	λόγος	NOUN	Nb	Case=Nom	0	root	_	_
"""
    )
    examples = [
        {
            "sent_id": "train1",
            "conllu": """# sent_id = train1
# text = λόγον
1	λόγον	λόγος	NOUN	Nb	Case=Acc|Gender=Masc|Number=Sing	0	root	_	_
""",
        },
        {
            "sent_id": "train2",
            "conllu": """# sent_id = train2
# text = λόγον
1	λόγον	λόγος	NOUN	Nb	Case=Acc|Gender=Masc|Number=Sing	0	root	_	_
""",
        },
    ]

    monkeypatch.setattr(
        "llmparser.main.load_database",
        lambda *_args, **_kwargs: DummyDatabase(examples),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "correct",
            str(prediction_path),
            "--db",
            str(tmp_path / "db"),
            "--output",
            str(output_path),
            "--feats-checker",
            "rule",
            "--n-examples",
            "2",
        ],
    )

    assert result.exit_code == 0
    output_text = output_path.read_text()
    assert "Case: Acc" in output_text
    assert "Gender: Masc" in output_text
    assert "status: patched" in output_text


def test_correct_command_can_run_rule_based_syntax_checker(tmp_path: Path, monkeypatch) -> None:
    prediction_path = tmp_path / "predicted.conllu"
    output_path = tmp_path / "corrected.yml"
    prediction_path.write_text(
        """# sent_id = s1
# text = Broken.
1	Broken	broken	ADJ	_	_	2	root	_	_
2	.	.	PUNCT	_	_	1	punct	_	_
"""
    )

    monkeypatch.setattr(
        "llmparser.main.load_database",
        lambda *_args, **_kwargs: DummyDatabase([]),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "correct",
            str(prediction_path),
            "--db",
            str(tmp_path / "db"),
            "--output",
            str(output_path),
            "--syntax-checker",
            "rule",
        ],
    )

    assert result.exit_code == 0
    output_text = output_path.read_text()
    assert "head" in output_text
    assert "value: 0" in output_text
    assert "status: patched" in output_text


def test_correct_command_can_use_correction_config(tmp_path: Path, monkeypatch) -> None:
    prediction_path = tmp_path / "predicted.conllu"
    output_path = tmp_path / "corrected.yml"
    config_path = tmp_path / "correction.yml"
    prediction_path.write_text(
        """# sent_id = s1
# text = Hello
1	Hello	hello	NOUN	NN	_	0	root	_	_
"""
    )
    config_path.write_text(
        """
lemma:
  rule: true
upos_xpos:
  rule: true
"""
    )
    examples = [
        {
            "sent_id": "train1",
            "conllu": """# sent_id = train1
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
""",
        },
        {
            "sent_id": "train2",
            "conllu": """# sent_id = train2
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
""",
        },
    ]

    monkeypatch.setattr(
        "llmparser.main.load_database",
        lambda *_args, **_kwargs: DummyDatabase(examples),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "correct",
            str(prediction_path),
            "--db",
            str(tmp_path / "db"),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--n-examples",
            "2",
        ],
    )

    assert result.exit_code == 0
    output_text = output_path.read_text()
    assert "INTJ" in output_text
    assert "Uh" in output_text
    assert "status: patched" in output_text


def test_serve_mcp_command_reports_missing_sdk(tmp_path: Path, monkeypatch) -> None:
    prediction_path = tmp_path / "predicted.conllu"
    prediction_path.write_text(
        """# sent_id = s1
# text = Hello
1	Hello	hello	INTJ	_	_	0	root	_	_
"""
    )

    monkeypatch.setattr(
        "llmparser.main.load_database",
        lambda *_args, **_kwargs: DummyDatabase([]),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["serve-mcp", str(prediction_path), "--db", str(tmp_path / "db")],
    )

    assert result.exit_code != 0
    assert "MCP SDK is not installed" in result.output
