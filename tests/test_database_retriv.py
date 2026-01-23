from pathlib import Path

import llmparser.database.database_retriv as database_retriv


class FakeSparseRetriever:
    _stores = {}

    def __init__(self, index_name: str):
        self.index_name = index_name
        base_path = Path(database_retriv.os.environ.get("RETRIV_BASE_PATH", ""))
        self._key = (str(base_path), index_name)
        self._docs = self._stores.setdefault(self._key, [])

    def index(self, docs, show_progress: bool = True):
        self._docs.extend(list(docs))

    def search(self, _query: str, return_docs: bool = True, cutoff: int = 10):
        return list(self._docs)[:cutoff]


def test_build_retriv_collection_reads_conllu() -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"

    collection = database_retriv.build_retriv_collection(conllu_path)

    assert collection
    assert any(item.get("id") == "5.7" for item in collection)
    assert any(item.get("conllu") for item in collection)
    assert all(item.get("text") for item in collection)


def test_database_retriv_build_load_and_search(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(database_retriv, "SparseRetriever", FakeSparseRetriever)
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"
    db_path = tmp_path / "retriv_db"

    db = database_retriv.build_database(conllu_path, db_path, index_name="index")
    assert db.index_name == "index"

    loaded_db = database_retriv.load_database(db_path, index_name="index")
    assert loaded_db.index_name == "index"

    results = db("test query", k=2)
    assert len(results) <= 2
    assert results

    example_docs = db.similarity_search("test query", k=1)
    assert example_docs
    assert hasattr(example_docs[0], "metadata")
