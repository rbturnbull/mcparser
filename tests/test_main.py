from pathlib import Path

from typer.testing import CliRunner

from llmparser.database import build_embedding_docs, load_database
from llmparser.main import app


def test_db_command_builds_database(tmp_path: Path, patch_embeddings_and_chroma) -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"
    db_path = tmp_path / "db"

    runner = CliRunner()
    result = runner.invoke(app, ["db", str(conllu_path), str(db_path)])

    assert result.exit_code == 0
    expected_docs = build_embedding_docs(conllu_path)
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
