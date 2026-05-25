from pathlib import Path

from llmparser.conllu_model import ConlluSentence
from llmparser.correction import (
    CorrectionPatch,
    TokenCorrectionResult,
    correct_predictions,
    correct_token_column,
)
from llmparser.evidence import TrainingEvidence
from llmparser.segmentation import SegmentationCorrectionResult, TokenSplitPatch
from llmparser.validation import validate_sentence


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


def test_validate_sentence_reports_duplicate_id_invalid_upos_and_root_mismatch() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = bad
# text = Broken root.
1	Broken	broken	BADPOS	_	_	2	root	_	_
1	root	root	NOUN	_	_	0	nsubj	_	_
"""
    )

    errors = validate_sentence(sentence)
    codes = {error.code for error in errors}

    assert "duplicate-id" in codes
    assert "invalid-upos" in codes
    assert "root-deprel-head" in codes
    assert "head-zero-deprel" in codes


def test_correction_patch_normalises_column() -> None:
    patch = CorrectionPatch(token_id=1, column="UPOS", value="INTJ")

    assert patch.column == "upos"


def test_token_correction_result_collects_patch_evidence_and_notes() -> None:
    result = TokenCorrectionResult(
        token_id=1,
        column="LEMMA",
        status="patched",
        patches=[
            CorrectionPatch(
                token_id=1,
                column="lemma",
                value="hello-corrected",
                evidence_sent_ids=["train1"],
            )
        ],
        evidence_sent_ids=["train1"],
        notes="Predicted lemma differed from comparable examples.",
    )

    assert result.column == "lemma"
    assert result.status == "patched"
    assert result.patches[0].value == "hello-corrected"
    assert result.to_dict()["notes"] == "Predicted lemma differed from comparable examples."


def test_correct_predictions_applies_checker_patch(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predicted.conllu"
    prediction_path.write_text(PREDICTED_CONLLU)
    database = DummyDatabase()

    def checker(sentence, column, examples):
        if column == "upos" and sentence.tokens[0].form == "Hello":
            return [
                CorrectionPatch(
                    token_id=1,
                    column="upos",
                    value="INTJ",
                    reason="Training examples use INTJ for greeting Hello.",
                    evidence_sent_ids=["train1"],
                )
            ]
        return []

    corrected = correct_predictions(
        prediction_path,
        database,
        checker=checker,
        n_examples=1,
    )

    item = corrected["s1"]
    assert database.queries == [("Hello world.", 1)]
    assert item["tokens"][0]["upos"] == "INTJ"
    assert "1\tHello\thello\tINTJ" in item["conllu"]
    assert item["audit"]["upos"]["patches"][0]["evidence_sent_ids"] == ["train1"]


def test_correct_token_column_loops_through_tokens_and_applies_lemma_patches() -> None:
    sentence = ConlluSentence.from_text(PREDICTED_CONLLU)
    database = DummyDatabase()
    evidence = TrainingEvidence(database)
    calls = []

    def token_checker(sentence, token, column, examples):
        calls.append((token.id, column, [example["sent_id"] for example in examples.form_examples]))
        if token.form == "Hello":
            return TokenCorrectionResult(
                token_id=token.id,
                column=column,
                status="patched",
                patches=[
                    CorrectionPatch(
                        token_id=token.id,
                        column=column,
                        value="hello-corrected",
                        reason="Comparable training examples prefer this lemma.",
                        evidence_sent_ids=["train1"],
                    )
                ],
                evidence_sent_ids=["train1"],
                notes="Comparable examples prefer this lemma.",
            )
        return TokenCorrectionResult(
            token_id=token.id,
            column=column,
            status="checked",
            evidence_sent_ids=["train1"],
        )

    audit = correct_token_column(
        sentence,
        column="lemma",
        evidence=evidence,
        checker=token_checker,
        n_examples=1,
    )

    assert sentence.tokens[0].lemma == "hello-corrected"
    assert sentence.tokens[1].lemma == "world"
    assert calls == [
        (1, "lemma", ["train1"]),
        (2, "lemma", ["train1"]),
    ]
    assert database.queries == [
        ("FORM Hello", 1),
        ("LEMMA hello", 1),
        ("FORM Hello UPOS NOUN", 1),
        ("Hello world.", 1),
        ("FORM world", 1),
        ("LEMMA world", 1),
        ("FORM world UPOS NOUN", 1),
        ("Hello world.", 1),
    ]
    assert audit.status == "patched"
    assert audit.patches[0].token_id == 1
    assert audit.token_results[0].notes == "Comparable examples prefer this lemma."


def test_correct_predictions_accepts_token_checker_for_lemma(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predicted.conllu"
    prediction_path.write_text(PREDICTED_CONLLU)
    database = DummyDatabase()

    def token_checker(sentence, token, column, examples):
        if column == "lemma" and token.form == "Hello":
            return [CorrectionPatch(token_id=token.id, column=column, value="hello-corrected")]
        return []

    corrected = correct_predictions(
        prediction_path,
        database,
        token_checkers={"lemma": token_checker},
        n_examples=1,
        columns=("lemma",),
    )

    item = corrected["s1"]
    assert item["tokens"][0]["lemma"] == "hello-corrected"
    assert item["audit"]["lemma"]["status"] == "patched"
    assert database.queries == [
        ("Hello world.", 1),
        ("FORM Hello", 1),
        ("LEMMA hello", 1),
        ("FORM Hello UPOS NOUN", 1),
        ("Hello world.", 1),
        ("FORM world", 1),
        ("LEMMA world", 1),
        ("FORM world UPOS NOUN", 1),
        ("Hello world.", 1),
    ]


def test_correct_lemma_skill_file_documents_token_patch_contract() -> None:
    skill_path = Path(__file__).parents[1] / "agents" / "correct_lemma" / "SKILL.md"

    content = skill_path.read_text()

    assert "LEMMA" in content
    assert "CorrectionPatch" in content
    assert "Do not modify" in content


def test_correct_predictions_runs_segmentation_before_token_checkers(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predicted.conllu"
    prediction_path.write_text(
        """# sent_id = ar1
# text = وكتابه جاء
1	وكتابه	وكتابه	NOUN	_	_	2	nsubj	_	_
2	جاء	جاء	VERB	_	_	0	root	_	_
"""
    )
    database = DummyDatabase()
    seen_forms = []

    def segmentation_checker(sentence, examples):
        return SegmentationCorrectionResult(
            status="patched",
            patches=[
                TokenSplitPatch(
                    token_id=1,
                    replacement_tokens=[
                        {
                            "id": 1,
                            "form": "و",
                            "lemma": "و",
                            "upos": "CCONJ",
                            "xpos": "_",
                            "feats": None,
                            "head": None,
                            "deprel": None,
                            "deps": None,
                            "misc": None,
                        },
                        {
                            "id": 2,
                            "form": "كتابه",
                            "lemma": "كتاب",
                            "upos": "NOUN",
                            "xpos": "_",
                            "feats": None,
                            "head": None,
                            "deprel": None,
                            "deps": None,
                            "misc": None,
                        },
                    ],
                )
            ],
            needs_syntax_correction=True,
        )

    def token_checker(sentence, token, column, examples):
        seen_forms.append(token.form)
        return TokenCorrectionResult(token_id=token.id, column=column, status="checked")

    corrected = correct_predictions(
        prediction_path,
        database,
        segmentation_checker=segmentation_checker,
        token_checkers={"lemma": token_checker},
        columns=("lemma",),
    )

    item = corrected["ar1"]
    assert [token["form"] for token in item["tokens"]] == ["و", "كتابه", "جاء"]
    assert seen_forms == ["و", "كتابه", "جاء"]
    assert item["audit"]["segmentation"]["status"] == "patched"
    assert item["audit"]["segmentation"]["needs_syntax_correction"] is True
