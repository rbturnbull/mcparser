from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .databases.database_chroma import (
    DatabaseChroma,
)
from .databases.database_retriv import DatabaseRetriv


_DATABASE_CLASSES: tuple[type, ...] = (DatabaseChroma, DatabaseRetriv)


def build_database(
    conllu_path: Path,
    output_path: Path,
):
    return DatabaseRetriv.build(conllu_path, output_path)


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
