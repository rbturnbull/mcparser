from pathlib import Path

from llmparser.conllu_model import ConlluSentence
from llmparser.syntax_checker import RuleBasedSyntaxChecker


def test_rule_based_syntax_checker_patches_root_deprel_wrong_head() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Broken.
1	Broken	broken	ADJ	_	_	2	root	_	_
2	.	.	PUNCT	_	_	1	punct	_	_
"""
    )

    result = RuleBasedSyntaxChecker()(sentence, [])

    assert result.status == "patched"
    assert result.patches[0].column == "head"
    assert result.patches[0].value == 0


def test_rule_based_syntax_checker_patches_head_zero_wrong_deprel() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Broken.
1	Broken	broken	ADJ	_	_	0	nsubj	_	_
"""
    )

    result = RuleBasedSyntaxChecker()(sentence, [])

    assert result.status == "patched"
    assert result.patches[0].column == "deprel"
    assert result.patches[0].value == "root"


def test_rule_based_syntax_checker_marks_multiple_roots_uncertain() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Broken twice.
1	Broken	broken	ADJ	_	_	0	root	_	_
2	twice	twice	ADV	_	_	0	root	_	_
"""
    )

    result = RuleBasedSyntaxChecker()(sentence, [])

    assert result.status == "uncertain"
    assert result.patches == []
    assert "root" in result.notes


def test_rule_based_syntax_checker_marks_missing_head_uncertain() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = Broken.
1	Broken	broken	ADJ	_	_	4	amod	_	_
"""
    )

    result = RuleBasedSyntaxChecker()(sentence, [])

    assert result.status == "uncertain"
    assert result.patches == []
    assert "missing" in result.notes


def test_correct_head_deprel_skill_file_documents_sentence_patch_contract() -> None:
    skill_path = Path(__file__).parents[1] / "agents" / "correct_head_deprel" / "SKILL.md"

    content = skill_path.read_text()

    assert "HEAD" in content
    assert "DEPREL" in content
    assert "SentenceCorrectionResult" in content
    assert "Do not modify" in content
