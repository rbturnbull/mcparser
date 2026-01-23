from __future__ import annotations

from pathlib import Path

from .database_chroma import DatabaseChroma, DEFAULT_EMBEDDING_MODEL_ID, build_embedding_docs
from .database_retriv import DatabaseRetriv


_DATABASE_CLASSES: tuple[type, ...] = (DatabaseChroma, DatabaseRetriv)


def build_database(
    conllu_path: Path,
    output_path: Path,
    embedding_model_id: str | None = DEFAULT_EMBEDDING_MODEL_ID,
):
    model_id = embedding_model_id or DEFAULT_EMBEDDING_MODEL_ID
    return DatabaseChroma.build(conllu_path, output_path, embedding_model_id=model_id)


def load_database(
    path: Path | str,
    embedding_model_id: str | None = None,
):
    errors: list[tuple[str, Exception]] = []
    for db_cls in _DATABASE_CLASSES:
        try:
            if db_cls is DatabaseChroma and embedding_model_id is not None:
                return DatabaseChroma(path, embedding_model_id=embedding_model_id)
            return db_cls(path)
        except Exception as exc:
            errors.append((db_cls.__name__, exc))
            continue
    error_lines = "; ".join(f"{name}: {exc}" for name, exc in errors)
    raise RuntimeError(f"Unable to load database from {path}. Errors: {error_lines}")
