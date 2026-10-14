from llmparser.conllu_model import ConlluSentence
from llmparser.evidence import TokenEvidence
from llmparser.pos_checker import RuleBasedPosChecker


def test_rule_based_pos_checker_patches_strong_majority() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Hello
1	Hello	hello	NOUN	NN	_	0	root	_	_
"""
    )
    token = sentence.tokens[0]
    checker = RuleBasedPosChecker(min_count=2, min_ratio=0.75)
    evidence = TokenEvidence(
        form_examples=[
            {
                "sent_id": "train1",
                "conllu": """# sent_id = train1
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
""",
            },
            {
                "sent_id": "train2",
                "conllu": """# sent_id = train2
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
""",
            },
        ]
    )

    result = checker(sentence, token, "upos", evidence)

    assert result.status == "patched"
    assert [patch.column for patch in result.patches] == ["upos", "xpos"]
    assert [patch.value for patch in result.patches] == ["INTJ", "Uh"]
    assert result.evidence_sent_ids == ["train1", "train2"]


def test_rule_based_pos_checker_preserves_matching_prediction() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
"""
    )
    token = sentence.tokens[0]
    checker = RuleBasedPosChecker(min_count=1, min_ratio=1.0)
    evidence = [
        {
            "sent_id": "train1",
            "conllu": """# sent_id = train1
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
""",
        },
    ]

    result = checker(sentence, token, "upos", evidence)

    assert result.status == "checked"
    assert result.patches == []
    assert result.evidence_sent_ids == ["train1"]


def test_rule_based_pos_checker_marks_weak_evidence_uncertain() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Hello
1	Hello	hello	NOUN	NN	_	0	root	_	_
"""
    )
    token = sentence.tokens[0]
    checker = RuleBasedPosChecker(min_count=2, min_ratio=0.75)
    evidence = [
        {
            "sent_id": "train1",
            "conllu": """# sent_id = train1
# text = Hello
1	Hello	hello	INTJ	Uh	_	0	root	_	_
""",
        },
    ]

    result = checker(sentence, token, "upos", evidence)

    assert result.status == "uncertain"
    assert result.patches == []


def test_correct_upos_xpos_skill_file_documents_token_patch_contract() -> None:
    from pathlib import Path

    skill_path = Path(__file__).parents[1] / "agents" / "correct_upos_xpos" / "SKILL.md"

    content = skill_path.read_text()

    assert "UPOS" in content
    assert "XPOS" in content
    assert "CorrectionPatch" in content
    assert "Do not modify" in content
