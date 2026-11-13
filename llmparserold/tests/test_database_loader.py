from pathlib import Path

import llmparser.database as database_pkg


class FakeChroma:
    def __init__(self, path: Path | str, embedding_model_id: str | None = None):
        self.path = Path(path)
        self.embedding_model_id = embedding_model_id


class FakeRetriv:
    def __init__(self, path: Path | str):
        self.path = Path(path)


class AlwaysFail:
    def __init__(self, _path):
        raise ValueError("nope")


def test_load_database_prefers_chroma_when_embedding_model_id_set(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(database_pkg, "DatabaseChroma", FakeChroma)
    monkeypatch.setattr(database_pkg, "DatabaseRetriv", FakeRetriv)
    monkeypatch.setattr(database_pkg, "_DATABASE_CLASSES", (FakeChroma, FakeRetriv))

    db = database_pkg.load_database(tmp_path, embedding_model_id="model-x")

    assert isinstance(db, FakeChroma)
    assert db.embedding_model_id == "model-x"


def test_load_database_falls_back_to_next_class(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(database_pkg, "_DATABASE_CLASSES", (AlwaysFail, FakeRetriv))

    db = database_pkg.load_database(tmp_path)

    assert isinstance(db, FakeRetriv)


def test_load_database_raises_with_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(database_pkg, "_DATABASE_CLASSES", (AlwaysFail, AlwaysFail))

    try:
        database_pkg.load_database(tmp_path)
        assert False, "Expected RuntimeError when all loaders fail."
    except RuntimeError as exc:
        message = str(exc)
        assert "Unable to load database" in message
        assert "AlwaysFail" in message
