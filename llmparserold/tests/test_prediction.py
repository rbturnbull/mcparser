from llmparser.conllu_model import ConlluSentence
from llmparser.correction import CorrectionPatch, TokenCorrectionResult
from llmparser.prediction import predict_sentence_columns


def test_predict_sentence_columns_applies_column_predictor_patches() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Hello
1	Hello	_	_	_	_	_	_	_	_
"""
    )
    calls = []

    def lemma_predictor(sentence, token, column, evidence):
        calls.append((token.id, column))
        return TokenCorrectionResult(
            token_id=token.id,
            column=column,
            status="patched",
            patches=[
                CorrectionPatch(
                    token_id=token.id,
                    column=column,
                    value="hello",
                    reason="Predicted lemma from form.",
                )
            ],
        )

    audit = predict_sentence_columns(
        sentence,
        predictors={"lemma": lemma_predictor},
        columns=("lemma",),
    )

    assert sentence.tokens[0].lemma == "hello"
    assert calls == [(1, "lemma")]
    assert audit["lemma"]["status"] == "patched"


def test_predict_sentence_columns_runs_columns_in_order() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Hello
1	Hello	_	_	_	_	_	_	_	_
"""
    )
    calls = []

    def predictor(sentence, token, column, evidence):
        calls.append(column)
        return []

    predict_sentence_columns(
        sentence,
        predictors={
            "lemma": predictor,
            "upos": predictor,
        },
        columns=("lemma", "upos"),
    )

    assert calls == ["lemma", "upos"]


def test_predict_sentence_columns_records_checked_for_missing_predictor() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Hello
1	Hello	_	_	_	_	_	_	_	_
"""
    )

    audit = predict_sentence_columns(sentence, predictors={}, columns=("lemma",))

    assert audit["lemma"]["status"] == "checked"
    assert audit["lemma"]["patches"] == []
