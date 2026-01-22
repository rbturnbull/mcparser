from pathlib import Path

import llmparser.database as database
from llmparser.database import build_database, build_embedding_docs, load_database


def test_build_embedding_docs_reads_conllu_file() -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"

    documents = build_embedding_docs(conllu_path)

    assert documents
    first = documents[0]
    assert first.page_content
    assert first.metadata.get("sent_id") == "5.7"
    assert "sent_id = 5.7" in first.metadata.get("conllu", "")


def test_build_and_load_database(tmp_path: Path, monkeypatch) -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"
    expected_docs = build_embedding_docs(conllu_path)
    db_path = tmp_path / "db"

    class DummyEmbeddings:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakeChroma:
        def __init__(self, persist_directory: str, embedding_function) -> None:
            self.persist_directory = str(persist_directory)
            self.embedding_function = embedding_function
            self.documents = []
            marker = Path(self.persist_directory) / "docs_count.txt"
            self.docs_count = int(marker.read_text().strip()) if marker.exists() else 0

        @classmethod
        def from_documents(cls, documents, embedding, persist_directory: str):
            instance = cls(persist_directory=persist_directory, embedding_function=embedding)
            instance.documents = list(documents)
            instance.docs_count = len(instance.documents)
            path = Path(persist_directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "docs_count.txt").write_text(str(instance.docs_count))
            return instance

        def add_documents(self, documents) -> None:
            self.documents.extend(documents)
            self.docs_count = len(self.documents)
            path = Path(self.persist_directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "docs_count.txt").write_text(str(self.docs_count))

    monkeypatch.setattr(database, "OpenAIEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(database, "Chroma", FakeChroma)

    db = build_database(conllu_path, db_path)
    assert db.docs_count == len(expected_docs)

    loaded_db = load_database(db_path)
    assert loaded_db.docs_count == len(expected_docs)
