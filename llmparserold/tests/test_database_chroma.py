from llmparser.database.database_chroma import DatabaseChroma


class DummyDoc:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content


class DummyDB:
    def __init__(self, docs, docs_count=0):
        self._docs = list(docs)
        self.docs_count = docs_count

    def similarity_search(self, _query: str, k: int = 10):
        return self._docs[:k]


def test_database_chroma_call_requires_loaded_db() -> None:
    db = DatabaseChroma.__new__(DatabaseChroma)
    db.db = None

    try:
        db("query")
        assert False, "Expected RuntimeError when database is not loaded."
    except RuntimeError as exc:
        assert "Database is not loaded" in str(exc)


def test_database_chroma_similarity_search_requires_loaded_db() -> None:
    db = DatabaseChroma.__new__(DatabaseChroma)
    db.db = None

    try:
        db.similarity_search("query")
        assert False, "Expected RuntimeError when database is not loaded."
    except RuntimeError as exc:
        assert "Database is not loaded" in str(exc)


def test_database_chroma_call_and_docs_count() -> None:
    docs = [
        DummyDoc(metadata={"id": "1"}, page_content="text-1"),
        DummyDoc(metadata={"id": "2"}, page_content="text-2"),
    ]
    db = DatabaseChroma.__new__(DatabaseChroma)
    db.db = DummyDB(docs, docs_count=2)

    results = db("query", k=1)

    assert results == [{"id": "1", "text": "text-1"}]
    assert db.docs_count == 2
