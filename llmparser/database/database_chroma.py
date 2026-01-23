from __future__ import annotations

from pathlib import Path

import conllu
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from rich.progress import Progress, track

from .database_base import DEFAULT_NUM_RESULTS, DatabaseBase


DEFAULT_EMBEDDING_MODEL_ID = "text-embedding-3-large"


def build_embedding_docs(conllu_path:Path) -> list[Document]:
    documents = []
    with open(conllu_path, "r", encoding="utf-8") as f:
        data = conllu.parse_incr(f)

        for example in track(data):
            text = example.metadata.get("text", "").strip()
            if not text:
                continue
            metadata = dict(example.metadata)
            metadata["conllu"] = example.serialize()
            document = Document(page_content=text, metadata=metadata)
            documents.append(document)
    return documents


class DatabaseChroma(DatabaseBase):
    def __init__(self, path: Path | str, embedding_model_id: str | None = None):
        self.embedding_model_id = embedding_model_id
        self.embedding_model: OpenAIEmbeddings | None = None
        self.db: Chroma | None = None
        super().__init__(path)

    @classmethod
    def build(
        cls,
        conllu_path: Path,
        output_path: Path,
        embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    ) -> "DatabaseChroma":
        model = OpenAIEmbeddings(model=embedding_model_id)
        docs = build_embedding_docs(conllu_path)

        print(f"Embedding {len(docs)} items to {output_path}")
        batch_size = 100
        num_batches = len(docs) // batch_size + (1 if len(docs) % batch_size != 0 else 0)

        db = None
        with Progress() as progress:
            task = progress.add_task("[cyan]Indexing documents...", total=len(docs))
            for i in range(num_batches):
                batch = docs[i * batch_size:(i + 1) * batch_size]
                if i == 0:
                    db = Chroma.from_documents(
                        documents=batch,
                        embedding=model,
                        persist_directory=str(output_path),
                    )
                else:
                    db.add_documents(batch)
                progress.update(task, advance=len(batch))
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "embedding_model.txt").write_text(embedding_model_id)
        return cls(output_path)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        assert path.exists(), f"Database path {path} does not exist"
        if self.embedding_model_id is None:
            self.embedding_model_id = (path / "embedding_model.txt").read_text().strip()
        self.embedding_model = OpenAIEmbeddings(model=self.embedding_model_id)
        self.db = Chroma(persist_directory=str(path), embedding_function=self.embedding_model)

    def __call__(self, query: str, k: int = DEFAULT_NUM_RESULTS) -> list[dict]:
        if self.db is None:
            raise RuntimeError("Database is not loaded.")
        results = self.db.similarity_search(query, k=k)
        return [
            {**doc.metadata, "text": doc.page_content}
            for doc in results
        ]

    def similarity_search(self, query: str, k: int = DEFAULT_NUM_RESULTS):
        if self.db is None:
            raise RuntimeError("Database is not loaded.")
        return self.db.similarity_search(query, k=k)

    @property
    def docs_count(self) -> int | None:
        if self.db is None:
            return None
        return getattr(self.db, "docs_count", None)


def build_database(
    conllu_path: Path,
    output_path: Path,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
) -> DatabaseChroma:
    return DatabaseChroma.build(conllu_path, output_path, embedding_model_id=embedding_model_id)


def load_database(
    path: Path | str,
    embedding_model_id: str | None = None,
) -> DatabaseChroma:
    return DatabaseChroma(path, embedding_model_id=embedding_model_id)
