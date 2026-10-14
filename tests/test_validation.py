from conftest import sentence
from llmparser.corpus.conllu import Sentence
from llmparser.corpus.validation import validate_sentence


def codes(s, known=None):
    return [(issue.code, issue.token_id) for issue in validate_sentence(s, known)]


def test_valid_sentence_has_no_issues():
    s = sentence(
        """
        1-2 del _ _ _ _ _ _ _ _
        1 de de ADP SPS00 _ 2 case _ _
        2 el el DET DA Definite=Def|PronType=Art 0 root _ _
        """
    )
    assert codes(s) == []
    assert codes(Sentence()) == []


def test_missing_values_are_warnings():
    s = sentence("1 a _ _ _ _ _ _ _ _")
    assert codes(s) == [
        ("missing-lemma", "1"),
        ("missing-upos", "1"),
        ("missing-head", "1"),
        ("missing-deprel", "1"),
    ]


def test_ids_ranges_tags_and_heads():
    s = sentence(
        """
        1-2 ab _ _ _ _ _ _ _ _
        2-3 bc _ _ _ _ _ _ _ _
        3-9 zz _ _ _ _ _ _ _ _
        1 a a NOUNY _ _ 0 nsubj _ _
        2 b b X _ _ x dep _ _
        3 c c X _ _ 3 dep _ _
        5 d d X _ _ 1 dep _ _
        """
    )
    assert codes(s) == [
        ("word-id-sequence", "5"),
        ("overlapping-range", "2-3"),
        ("bad-range", "3-9"),
        ("invalid-upos", "1"),
        ("root-deprel", "1"),
        ("invalid-head", "2"),
        ("self-loop", "3"),
    ]


def test_head_out_of_range_and_root_deprel_with_head():
    s = sentence(
        """
        1 a a X _ _ 9 dep _ _
        2 b b X _ _ 1 root _ _
        """
    )
    assert codes(s) == [("invalid-head", "1"), ("root-deprel", "2")]


def test_feats_checks():
    s = sentence(
        """
        1 a a X _ Number=Sing|Case=Nom 0 root _ _
        2 b b X _ Case=Nom|Case=Gen 1 dep _ _
        3 c c X _ Case 1 dep _ _
        4 d d X _ Case=Nom|Number=Sing 1 dep _ _
        """
    )
    assert codes(s) == [("unsorted-feats", "1"), ("duplicate-feature", "2"), ("bad-feats", "3")]


def test_unseen_values():
    s = sentence("1 a a X Q Case=Abl 0 root _ _")
    known = {"xpos": {"N"}, "deprel": {"root"}, "feats": {"Case=Nom"}}
    assert codes(s, known) == [("unseen-value", "1"), ("unseen-value", "1")]


def test_tree_checks():
    no_root = sentence(
        """
        1 a a X _ _ 2 dep _ _
        2 b b X _ _ 1 dep _ _
        3 c c X _ _ 1 dep _ _
        """
    )
    assert codes(no_root) == [("root-count", None), ("cycle", "1")]
    two_roots = sentence(
        """
        1 a a X _ _ 0 root _ _
        2 b b X _ _ 0 root _ _
        """
    )
    assert codes(two_roots) == [("root-count", None)]
    issue = validate_sentence(no_root)[1]
    assert issue.message == "Words 1 -> 2 form a cycle."
