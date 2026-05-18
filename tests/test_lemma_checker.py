from llmparser.conllu_model import ConlluSentence
from llmparser.evidence import TokenEvidence
from llmparser.lemma_checker import RuleBasedLemmaChecker


def test_rule_based_lemma_checker_patches_strong_majority() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = ἐδόκεε
1	ἐδόκεε	ἐδόκεε	VERB	V-	_	0	root	_	_
"""
    )
    token = sentence.tokens[0]
    checker = RuleBasedLemmaChecker(min_count=2, min_ratio=0.75)
    examples = [
        {
            "sent_id": "train1",
            "conllu": """# sent_id = train1
# text = ἐδόκεε
1	ἐδόκεε	δοκέω	VERB	V-	_	0	root	_	_
""",
        },
        {
            "sent_id": "train2",
            "conllu": """# sent_id = train2
# text = ἐδόκεε
1	ἐδόκεε	δοκέω	VERB	V-	_	0	root	_	_
""",
        },
    ]

    result = checker(sentence, token, "lemma", examples)

    assert result.status == "patched"
    assert result.patches[0].value == "δοκέω"
    assert result.patches[0].evidence_sent_ids == ["train1", "train2"]


def test_rule_based_lemma_checker_preserves_matching_prediction() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = ἐδόκεε
1	ἐδόκεε	δοκέω	VERB	V-	_	0	root	_	_
"""
    )
    token = sentence.tokens[0]
    checker = RuleBasedLemmaChecker(min_count=1, min_ratio=1.0)
    examples = [
        {
            "sent_id": "train1",
            "conllu": """# sent_id = train1
# text = ἐδόκεε
1	ἐδόκεε	δοκέω	VERB	V-	_	0	root	_	_
""",
        },
    ]

    result = checker(sentence, token, "lemma", examples)

    assert result.status == "checked"
    assert result.patches == []
    assert result.evidence_sent_ids == ["train1"]


def test_rule_based_lemma_checker_marks_weak_evidence_uncertain() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = ἐδόκεε
1	ἐδόκεε	ἐδόκεε	VERB	V-	_	0	root	_	_
"""
    )
    token = sentence.tokens[0]
    checker = RuleBasedLemmaChecker(min_count=2, min_ratio=0.75)
    examples = [
        {
            "sent_id": "train1",
            "conllu": """# sent_id = train1
# text = ἐδόκεε
1	ἐδόκεε	δοκέω	VERB	V-	_	0	root	_	_
""",
        },
    ]

    result = checker(sentence, token, "lemma", examples)

    assert result.status == "uncertain"
    assert result.patches == []


def test_rule_based_lemma_checker_accepts_token_evidence_bundle() -> None:
    sentence = ConlluSentence.from_text(
        """# sent_id = pred
# text = ἐδόκεε
1	ἐδόκεε	ἐδόκεε	VERB	V-	_	0	root	_	_
"""
    )
    token = sentence.tokens[0]
    checker = RuleBasedLemmaChecker(min_count=1, min_ratio=1.0)
    bundle = TokenEvidence(
        form_examples=[
            {
                "sent_id": "train1",
                "conllu": """# sent_id = train1
# text = ἐδόκεε
1	ἐδόκεε	δοκέω	VERB	V-	_	0	root	_	_
""",
            }
        ],
        lemma_examples=[
            {
                "sent_id": "noise",
                "conllu": """# sent_id = noise
# text = unrelated
1	unrelated	unrelated	ADJ	V-	_	0	root	_	_
""",
            }
        ],
    )

    result = checker(sentence, token, "lemma", bundle)

    assert result.status == "patched"
    assert result.patches[0].value == "δοκέω"
