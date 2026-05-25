from pathlib import Path

from llmparser.conllu_model import ConlluSentence, ConlluToken
from llmparser.segmentation import (
    SegmentationCorrectionResult,
    TokenDeletePatch,
    TokenInsertPatch,
    TokenMergePatch,
    TokenSplitPatch,
    apply_segmentation_patch,
    apply_token_delete,
    apply_token_insert,
    apply_token_merge,
    apply_token_split,
)


def test_apply_token_split_replaces_token_and_renumbers_following_tokens() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = ar1
# text = وكتابه جاء
1	وكتابه	وكتابه	NOUN	_	_	2	nsubj	_	_
2	جاء	جاء	VERB	_	_	0	root	_	_
"""
    )
    patch = TokenSplitPatch(
        token_id=1,
        replacement_tokens=[
            ConlluToken(
                id=1,
                form="و",
                lemma="و",
                upos="CCONJ",
                xpos="_",
                feats=None,
                head=None,
                deprel=None,
                deps=None,
                misc={"Segmentation": "Split"},
            ),
            ConlluToken(
                id=2,
                form="كتابه",
                lemma="كتاب",
                upos="NOUN",
                xpos="_",
                feats=None,
                head=None,
                deprel=None,
                deps=None,
                misc={"Segmentation": "Split"},
            ),
        ],
        reason="Training examples split proclitic conjunction from the noun.",
        evidence_sent_ids=["train1"],
    )

    result = apply_token_split(sentence, patch)

    assert result.status == "patched"
    assert result.needs_syntax_correction is True
    assert [token.id for token in sentence.tokens] == [1, 2, 3]
    assert [token.form for token in sentence.tokens] == ["و", "كتابه", "جاء"]
    assert sentence.tokens[2].head == 0
    assert sentence.sent_id == "ar1"


def test_segmentation_result_serializes_patch() -> None:
    result = SegmentationCorrectionResult(
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
                    }
                ],
            )
        ],
        needs_syntax_correction=True,
    )

    assert result.to_dict()["patches"][0]["replacement_tokens"][0]["form"] == "و"


def test_apply_token_merge_replaces_adjacent_tokens_and_renumbers() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = ar1
# text = و كتابه جاء
1	و	و	CCONJ	_	_	2	cc	_	_
2	كتابه	كتاب	NOUN	_	_	3	nsubj	_	_
3	جاء	جاء	VERB	_	_	0	root	_	_
"""
    )
    patch = TokenMergePatch(
        token_ids=[1, 2],
        replacement_token={
            "id": 1,
            "form": "وكتابه",
            "lemma": "وكتابه",
            "upos": "NOUN",
            "xpos": "_",
            "feats": None,
            "head": None,
            "deprel": None,
            "deps": None,
            "misc": {"Segmentation": "Merged"},
        },
    )

    result = apply_token_merge(sentence, patch)

    assert result.status == "patched"
    assert [token.id for token in sentence.tokens] == [1, 2]
    assert [token.form for token in sentence.tokens] == ["وكتابه", "جاء"]


def test_apply_token_delete_removes_token_and_renumbers() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = ar1
# text = و كتابه جاء
1	و	و	CCONJ	_	_	2	cc	_	_
2	كتابه	كتاب	NOUN	_	_	3	nsubj	_	_
3	جاء	جاء	VERB	_	_	0	root	_	_
"""
    )

    result = apply_token_delete(sentence, TokenDeletePatch(token_id=1))

    assert result.status == "patched"
    assert [token.id for token in sentence.tokens] == [1, 2]
    assert [token.form for token in sentence.tokens] == ["كتابه", "جاء"]


def test_apply_token_insert_adds_tokens_after_target_and_renumbers() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = ar1
# text = كتابه جاء
1	كتابه	كتاب	NOUN	_	_	2	nsubj	_	_
2	جاء	جاء	VERB	_	_	0	root	_	_
"""
    )
    patch = TokenInsertPatch(
        anchor_token_id=1,
        position="after",
        tokens=[
            {
                "id": 2,
                "form": "ه",
                "lemma": "هو",
                "upos": "PRON",
                "xpos": "_",
                "feats": {"Person": "3"},
                "head": None,
                "deprel": None,
                "deps": None,
                "misc": {"Segmentation": "Inserted"},
            }
        ],
    )

    result = apply_token_insert(sentence, patch)

    assert result.status == "patched"
    assert [token.id for token in sentence.tokens] == [1, 2, 3]
    assert [token.form for token in sentence.tokens] == ["كتابه", "ه", "جاء"]


def test_apply_segmentation_patch_dispatches_dict_patch() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = ar1
# text = كتابه جاء
1	كتابه	كتاب	NOUN	_	_	2	nsubj	_	_
2	جاء	جاء	VERB	_	_	0	root	_	_
"""
    )

    result = apply_segmentation_patch(
        sentence,
        {
            "type": "delete",
            "token_id": 1,
        },
    )

    assert result.status == "patched"
    assert sentence.tokens[0].form == "جاء"


def test_correct_segmentation_skill_file_documents_row_patch_contract() -> None:
    skill_path = Path(__file__).parents[1] / "agents" / "correct_segmentation" / "SKILL.md"

    content = skill_path.read_text()

    assert "TokenSplitPatch" in content
    assert "TokenMergePatch" in content
    assert "TokenInsertPatch" in content
    assert "TokenDeletePatch" in content
    assert "row-level" in content
    assert "Do not rewrite" in content
