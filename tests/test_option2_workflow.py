from pathlib import Path

from llmparser.conllu_model import ConlluSentence
from llmparser.validation import validate_sentence
from llmparser.workflow import check_external_predictions


PREDICTED_CONLLU = """# sent_id = s1
# text = Hello world.
1	Hello	hello	NOUN	_	_	2	nsubj	_	_
2	world	world	NOUN	_	_	0	root	_	_
"""


class DummyDatabase:
    def __init__(self):
        self.queries = []

    def __call__(self, query: str, k: int = 10):
        self.queries.append((query, k))
        return [
            {
                "sent_id": "train1",
                "text": "Hello there.",
                "conllu": "1\tHello\thello\tINTJ\t_\t_\t0\troot\t_\t_",
            }
        ]


def test_conllu_sentence_round_trips_prediction() -> None:
    sentence = ConlluSentence.from_text(PREDICTED_CONLLU)

    assert sentence.sent_id == "s1"
    assert sentence.text == "Hello world."
    assert sentence.tokens[0].form == "Hello"
    assert sentence.tokens[0].upos == "NOUN"

    serialized = sentence.to_conllu()

    assert "# sent_id = s1" in serialized
    assert "1\tHello\thello\tNOUN" in serialized
    assert "2\tworld\tworld\tNOUN" in serialized


def test_validate_sentence_reports_invalid_head() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = bad
# text = Broken.
1	Broken	broken	ADJ	_	_	4	root	_	_
"""
    )

    errors = validate_sentence(sentence)

    assert errors
    assert errors[0].code == "invalid-head"
    assert "4" in errors[0].message


def test_check_external_predictions_applies_checker_patch(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predicted.conllu"
    prediction_path.write_text(PREDICTED_CONLLU)
    database = DummyDatabase()

    def checker(sentence, column, examples):
        if column == "upos" and sentence.tokens[0].form == "Hello":
            return [
                {
                    "token_id": 1,
                    "column": "upos",
                    "value": "INTJ",
                    "reason": "Training examples use INTJ for greeting Hello.",
                    "evidence_sent_ids": ["train1"],
                }
            ]
        return []

    checked = check_external_predictions(
        prediction_path,
        database,
        checker=checker,
        n_examples=1,
    )

    item = checked["s1"]
    assert database.queries == [("Hello world.", 1)]
    assert item["tokens"][0]["upos"] == "INTJ"
    assert "1\tHello\thello\tINTJ" in item["conllu"]
    assert item["audit"]["upos"]["patches"][0]["evidence_sent_ids"] == ["train1"]
