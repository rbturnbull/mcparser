from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rich.progress import track, Progress
import conllu


DEFAULT_EMBEDDING_MODEL_ID = "text-embedding-3-large"

def get_db(docs:list[Document], model:OpenAIEmbeddings, path:Path|str) -> Chroma:
    if Path(path).exists():
        db = Chroma(persist_directory=str(path), embedding_function=model)
    else:
        print(f"Embedding {len(docs)} items to {path}")
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
                        persist_directory=str(path),
                    )                        
                else:
                    db.add_documents(batch)
                progress.update(task, advance=len(batch))
    return db


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


def build_database(
    conllu_path:Path,
    db:Path,
    embedding_model_id:str = "text-embedding-3-large",
):
    embeddings_model = OpenAIEmbeddings(model=embedding_model_id)
    
    docs = build_embedding_docs(conllu_path)
    get_db(docs, embeddings_model, db)



