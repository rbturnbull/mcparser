from llmparser.conllu_model import ConlluSentence
from llmparser.evidence import TokenEvidence
from llmparser.feats_checker import RuleBasedFeatsChecker


def test_rule_based_feats_checker_patches_strong_majority() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = λόγον
1	λόγον	λόγος	NOUN	Nb	Case=Nom	0	root	_	_
"""
    )
    token = sentence.tokens[0]
    checker = RuleBasedFeatsChecker(min_count=2, min_ratio=0.75)
    evidence = TokenEvidence(
        form_examples=[
            {
                "sent_id": "train1",
                "conllu": """# sent_id = train1
# text = λόγον
1	λόγον	λόγος	NOUN	Nb	Case=Acc|Gender=Masc|Number=Sing	0	root	_	_
""",
            },
            {
                "sent_id": "train2",
                "conllu": """# sent_id = train2
# text = λόγον
1	λόγον	λόγος	NOUN	Nb	Case=Acc|Gender=Masc|Number=Sing	0	root	_	_
""",
            },
        ]
    )

    result = checker(sentence, token, "feats", evidence)

    assert result.status == "patched"
    assert result.patches[0].column == "feats"
    assert result.patches[0].value == {"Case": "Acc", "Gender": "Masc", "Number": "Sing"}
    assert result.evidence_sent_ids == ["train1", "train2"]


def test_rule_based_feats_checker_preserves_matching_prediction() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = λόγον
1	λόγον	λόγος	NOUN	Nb	Case=Acc|Gender=Masc|Number=Sing	0	root	_	_
"""
    )
    token = sentence.tokens[0]
    checker = RuleBasedFeatsChecker(min_count=1, min_ratio=1.0)
    evidence = [
        {
            "sent_id": "train1",
            "conllu": """# sent_id = train1
# text = λόγον
1	λόγον	λόγος	NOUN	Nb	Case=Acc|Gender=Masc|Number=Sing	0	root	_	_
""",
        },
    ]

    result = checker(sentence, token, "feats", evidence)

    assert result.status == "checked"
    assert result.patches == []
    assert result.evidence_sent_ids == ["train1"]


def test_rule_based_feats_checker_marks_weak_evidence_uncertain() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = λόγον
1	λόγον	λόγος	NOUN	Nb	Case=Nom	0	root	_	_
"""
    )
    token = sentence.tokens[0]
    checker = RuleBasedFeatsChecker(min_count=2, min_ratio=0.75)
    evidence = [
        {
            "sent_id": "train1",
            "conllu": """# sent_id = train1
# text = λόγον
1	λόγον	λόγος	NOUN	Nb	Case=Acc|Gender=Masc|Number=Sing	0	root	_	_
""",
        },
    ]

    result = checker(sentence, token, "feats", evidence)

    assert result.status == "uncertain"
    assert result.patches == []


def test_correct_feats_skill_file_documents_token_patch_contract() -> None:
    from pathlib import Path

    skill_path = Path(__file__).parents[1] / "agents" / "correct_feats" / "SKILL.md"

    content = skill_path.read_text()

    assert "FEATS" in content
    assert "CorrectionPatch" in content
    assert "Do not modify" in content
