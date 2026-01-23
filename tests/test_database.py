from pathlib import Path

from llmparser.database import build_database, load_database
from llmparser.databases.database_chroma import build_embedding_docs
from llmparser.io import conllu_to_dict


def test_build_embedding_docs_reads_conllu_file() -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"

    documents = build_embedding_docs(conllu_path)

    assert documents
    first = documents[0]
    assert first.page_content
    assert first.metadata.get("sent_id") == "5.7"
    assert "sent_id = 5.7" in first.metadata.get("conllu", "")


def test_build_and_load_database(tmp_path: Path, patch_embeddings_and_chroma) -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"
    expected_docs = conllu_to_dict(conllu_path)
    db_path = tmp_path / "db"

    db = build_database(conllu_path, db_path)
    assert db.docs_count == len(expected_docs)

    loaded_db = load_database(db_path)
    assert loaded_db.docs_count == len(expected_docs)
