from pathlib import Path

import pytest

import llmparser.database as database


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


@pytest.fixture
def patch_embeddings_and_chroma(monkeypatch):
    monkeypatch.setattr(database, "OpenAIEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(database, "Chroma", FakeChroma)
    return FakeChroma
