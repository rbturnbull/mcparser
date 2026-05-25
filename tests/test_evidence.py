from llmparser.conllu_model import ConlluSentence
from llmparser.evidence import SegmentationEvidence, TokenEvidence, TrainingEvidence


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


def test_training_evidence_builds_token_evidence_bundle() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = λόγον λέγει
1	λόγον	λόγος	NOUN	_	Case=Acc	2	obj	_	_
2	λέγει	λέγω	VERB	_	_	0	root	_	_
"""
    )
    token = sentence.tokens[0]
    database = DummyDatabase()
    evidence = TrainingEvidence(database)

    bundle = evidence.find_token_evidence(sentence, token, column="lemma", k=2)

    assert isinstance(bundle, TokenEvidence)
    assert bundle.form_examples[0]["sent_id"] == "train1"
    assert bundle.lemma_examples[0]["sent_id"] == "train1"
    assert bundle.form_upos_examples[0]["sent_id"] == "train1"
    assert bundle.similar_sentence_examples[0]["sent_id"] == "train1"
    assert database.queries == [
        ("FORM λόγον", 2),
        ("LEMMA λόγος", 2),
        ("FORM λόγον UPOS NOUN", 2),
        ("λόγον λέγει", 2),
    ]


def test_token_evidence_all_examples_combines_and_deduplicates_sent_ids() -> None:
    bundle = TokenEvidence(
        form_examples=[{"sent_id": "train1"}, {"sent_id": "train2"}],
        lemma_examples=[{"sent_id": "train1"}],
        form_upos_examples=[{"sent_id": "train3"}],
    )

    assert [example["sent_id"] for example in bundle.all_examples()] == [
        "train1",
        "train2",
        "train3",
    ]


def test_training_evidence_builds_segmentation_evidence_bundle() -> None:
    database = DummyDatabase()
    evidence = TrainingEvidence(database)

    bundle = evidence.find_segmentation_evidence(
        surface="وكتابه",
        pieces=["و", "كتابه"],
        sentence_text="وكتابه جاء",
        k=2,
    )

    assert isinstance(bundle, SegmentationEvidence)
    assert bundle.surface_examples[0]["sent_id"] == "train1"
    assert bundle.subform_examples[0][0]["sent_id"] == "train1"
    assert bundle.sequence_examples[0]["sent_id"] == "train1"
    assert bundle.similar_sentence_examples[0]["sent_id"] == "train1"
    assert database.queries == [
        ("SURFACE وكتابه", 2),
        ("SUBFORM و", 2),
        ("SUBFORM كتابه", 2),
        ("SEQUENCE و كتابه", 2),
        ("وكتابه جاء", 2),
    ]
