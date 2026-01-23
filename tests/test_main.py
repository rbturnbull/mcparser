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
