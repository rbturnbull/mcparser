from llmparser.evidence import TrainingEvidence


class DummyDatabase:
    def __init__(self):
        self.queries = []

    def __call__(self, query: str, k: int = 10):
        self.queries.append((query, k))
        return [
            {
                "sent_id": "train1",
                "text": "λόγον λέγει",
                "conllu": "1\tλόγον\tλόγος\tNOUN\t_\tCase=Acc\t2\tobj\t_\t_",
            }
        ]


def test_training_evidence_searches_similar_sentences() -> None:
    database = DummyDatabase()
    evidence = TrainingEvidence(database)

    examples = evidence.find_similar_sentences("λόγον λέγει", k=3)

    assert database.queries == [("λόγον λέγει", 3)]
    assert examples[0]["sent_id"] == "train1"


def test_training_evidence_builds_form_and_lemma_queries() -> None:
    database = DummyDatabase()
    evidence = TrainingEvidence(database)

    evidence.find_examples_for_form("λόγον", k=4)
    evidence.find_examples_for_lemma("λόγος", k=5)

    assert database.queries == [
        ("FORM λόγον", 4),
        ("LEMMA λόγος", 5),
    ]
