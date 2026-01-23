from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import conllu
from retriv import SparseRetriever
from rich.progress import Progress, track


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


def build_database(conllu_path: Path, output_path: Path) -> SparseRetriever:
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

    sr = SparseRetriever(index_name="index")

    # If you want progress feedback, chunk the indexing (works well for ~10k docs)
    batch_size = 2_000
    with Progress() as progress:
        task = progress.add_task("[cyan]Indexing documents...", total=len(collection))
        for start in range(0, len(collection), batch_size):
            batch = collection[start : start + batch_size]
            if start == 0:
                sr.index(batch)
            else:
                # retriv supports incremental indexing by calling index() again
                sr.index(batch, show_progress=False)
            progress.update(task, advance=len(batch))

    return sr


def load_database(path: Path | str) -> SparseRetriever:
    """
    Load an existing retriv index named 'index' from the given base path.
    """
    path = Path(path)
    assert path.exists(), f"Database path {path} does not exist"

    os.environ["RETRIV_BASE_PATH"] = str(path)
    return SparseRetriever(index_name="my-index")


def search_database(db: SparseRetriever, query: str, cutoff: int = 10) -> list[dict[str, Any]]:
    """
    Search and return docs (including stored metadata_json).
    """
    return db.search(query, return_docs=True, cutoff=cutoff)
