from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import conllu
from retriv import SparseRetriever
from rich.progress import Progress, track

from .database_base import DEFAULT_NUM_RESULTS, DatabaseBase


def build_retriv_collection(conllu_path: Path) -> list[dict[str, Any]]:
    """
    Convert a .conllu file into a retriv collection:
      - required keys: {"id": ..., "text": ...}
      - we also store the full CoNLL-U serialization + metadata for later use.
    """
    collection: list[dict[str, Any]] = []

    with open(conllu_path, "r", encoding="utf-8") as f:
        data = conllu.parse_incr(f)

        for i, example in enumerate(track(data, description="Reading CoNLL-U...")):
            if 'text' not in example.metadata:
                continue
            metadata = dict(example.metadata)
            metadata["conllu"] = example.serialize()
            if 'id' not in metadata:
                metadata['id'] = metadata.get('sent_id', f"doc_{i}")

            # retriv needs "text"; extra fields are OK (and can be returned if you ask for docs)
            collection.append(metadata)

    return collection


class DatabaseRetriv(DatabaseBase):
    def __init__(self, path: Path | str, index_name: str = "index"):
        self.index_name = index_name
        self.db: SparseRetriever | None = None
        super().__init__(path)

    @classmethod
    def build(cls, conllu_path: Path, output_path: Path, index_name: str = "index") -> "DatabaseRetriv":
        """
        Build a retriv sparse index (BM25) called 'index' and persist it under output_path.

        retriv stores indices under a "base path". We set RETRIV_BASE_PATH so you fully control
        where the index lives on disk.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        os.environ["RETRIV_BASE_PATH"] = str(output_path)

        collection = build_retriv_collection(conllu_path)
        print(f"Indexing {len(collection)} items into retriv base path: {output_path}")

        sr = SparseRetriever(index_name=index_name)

        batch_size = 2_000
        with Progress() as progress:
            task = progress.add_task("[cyan]Indexing documents...", total=len(collection))
            for start in range(0, len(collection), batch_size):
                batch = collection[start : start + batch_size]
                sr.index(batch, show_progress=False)
                progress.update(task, advance=len(batch))

        return cls(output_path, index_name=index_name)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        assert path.exists(), f"Database path {path} does not exist"

        os.environ["RETRIV_BASE_PATH"] = str(path)
        if hasattr(SparseRetriever, "load"):
            self.db = SparseRetriever.load(index_name=self.index_name)
        else:
            self.db = SparseRetriever(index_name=self.index_name)

    def __call__(self, query: str, k: int = DEFAULT_NUM_RESULTS) -> list[dict]:
        if self.db is None:
            raise RuntimeError("Database is not loaded.")
        return self.db.search(query, return_docs=True, cutoff=k)

    def similarity_search(self, query: str, k: int = DEFAULT_NUM_RESULTS):
        if self.db is None:
            raise RuntimeError("Database is not loaded.")
        results = self.db.search(query, return_docs=True, cutoff=k)
        return [SimpleNamespace(metadata=result) for result in results]

    @property
    def docs_count(self) -> int | None:
        if self.db is None:
            return None
        return getattr(self.db, "doc_count", None)


def build_database(conllu_path: Path, output_path: Path, index_name: str = "index") -> DatabaseRetriv:
    return DatabaseRetriv.build(conllu_path, output_path, index_name=index_name)


def load_database(path: Path | str, index_name: str = "index") -> DatabaseRetriv:
    return DatabaseRetriv(path, index_name=index_name)


def search_database(db: DatabaseRetriv, query: str, cutoff: int = 10) -> list[dict[str, Any]]:
    return db(query, k=cutoff)
