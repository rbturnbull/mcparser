from pathlib import Path

from llmparser.database import build_embedding_docs


def test_build_embedding_docs_reads_conllu_file() -> None:
    conllu_path = Path(__file__).parent / "test_data" / "hit_hittb-ud-test.conllu"

    documents = build_embedding_docs(conllu_path)

    assert documents
    first = documents[0]
    assert first.page_content
    assert first.metadata.get("sent_id") == "5.7"
    assert "sent_id = 5.7" in first.metadata.get("conllu", "")
