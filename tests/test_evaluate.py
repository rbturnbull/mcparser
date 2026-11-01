import pytest

from conftest import DATA
from mcparser.corpus import parse_conllu
from mcparser.evaluate import calibration, calibration_table, evaluate

GOLD = """\
# sent_id = a
1\tVino\tvenir\tVERB\tV\tTense=Past\t0\troot\t_\t_
2-3\tdel\t_\t_\t_\t_\t_\t_\t_\t_
2\tde\tde\tADP\tP\t_\t4\tcase\t_\t_
3\tel\tel\tDET\tD\tGender=Masc|Number=Sing\t4\tdet\t_\t_
4\tmercado\tmercado\tNOUN\tN\t_\t1\tobl\t_\t_

# sent_id = b
1\tsí\tsí\tINTJ\tI\t_\t0\troot\t_\t_

"""

# "del" is left unsplit, "mercado" gets the wrong deprel, el's feats are reordered.
PRED = """\
# sent_id = a
1\tVino\tvenir\tVERB\tV\tTense=Past\t0\troot\t_\t_
2\tdel\tdel\tADP\tP\t_\t3\tcase\t_\t_
3\tmercado\tmercado\tNOUN\tN\t_\t1\tnmod\t_\t_

# sent_id = c
1\tno\tno\tADV\tR\t_\t0\troot\t_\t_

"""


def test_predicted_only():
    gold = parse_conllu(GOLD)
    result = evaluate(gold[:1], gold, predicted_only=True)
    assert (result.sentences, result.missing, result.score("LAS").f1) == (1, [], 1.0)


def test_identical_files_score_100():
    gold = parse_conllu((DATA / "train.conllu").read_text(encoding="utf-8"))
    result = evaluate(gold, gold)
    assert result.sentences == 3 and not result.missing and not result.extra
    assert {score.f1 for score in result.scores} == {1.0}


def test_segmentation_errors_and_missing_sentences():
    result = evaluate(parse_conllu(PRED), parse_conllu(GOLD))
    assert (result.sentences, result.missing, result.extra) == (1, ["b"], ["c"])
    words = result.score("Words")
    assert (words.correct, words.gold, words.predicted) == (2, 5, 3)
    assert words.precision == pytest.approx(2 / 3)
    assert words.recall == pytest.approx(2 / 5)
    assert words.f1 == pytest.approx(0.5)
    assert result.score("UAS").correct == 2  # mercado -> Vino is right even though ids differ
    assert result.score("LAS").correct == 1
    assert result.score("AllTags").correct == 2


def test_feature_order_and_wrong_head():
    gold = parse_conllu(GOLD)
    pred = parse_conllu(GOLD.replace("Gender=Masc|Number=Sing\t4", "Number=Sing|Gender=Masc\t1"))
    result = evaluate(pred, gold)
    assert result.score("UFeats").f1 == 1.0
    assert result.score("UAS").correct == 4  # el now points at Vino
    assert result.score("LAS").correct == 4


def test_empty_inputs_and_table():
    result = evaluate([], [])
    assert {score.f1 for score in result.scores} == {0.0}
    table = evaluate(parse_conllu(GOLD), parse_conllu(GOLD)).table()
    assert table.splitlines()[0].split() == ["Metric", "Precision", "Recall", "F1"]
    assert "LAS          100.00%   100.00%   100.00%" in table


def test_multiword_words_align_by_form():
    gold = parse_conllu(GOLD)
    pred = parse_conllu(GOLD.replace("3\tel\tel\tDET", "3\tl\tel\tDET"))
    assert evaluate(pred, gold).score("Words").correct == 4


def test_empty_nodes_are_ignored():
    gold = parse_conllu(GOLD.replace("4\tmercado", "3.1\tx\t_\t_\t_\t_\t_\t_\t_\t_\n4\tmercado"))
    assert evaluate(gold, gold).score("LAS").f1 == 1.0


def test_tokenisation_and_tag_errors():
    gold = parse_conllu(GOLD)
    pred = parse_conllu(GOLD.replace("4\tmercado\tmercado\tNOUN", "4\tmercado\tmercado\tPROPN"))
    result = evaluate(pred, gold)
    assert result.score("UPOS").correct == 4 and result.score("AllTags").correct == 4
    merged = parse_conllu("# sent_id = a\n1\tVinodel\tx\tX\t_\t_\t0\troot\t_\t_\n2\tmercado\tmercado\tNOUN\tN\t_\t1\tobl\t_\t_\n\n")
    assert evaluate(merged, gold).score("Words").correct == 1


def test_annotated_only():
    gold = parse_conllu(GOLD)
    unparsed = parse_conllu(GOLD.replace("\t0\troot", "\t_\troot", 1))
    result = evaluate(unparsed, gold, annotated_only=True)
    assert (result.sentences, result.missing) == (1, [])  # sentence a has no root head any more: left out
    assert result.score("LAS").f1 == 1.0


SCORED_GOLD = """\
# sent_id = a
1\tdog\tdog\tNOUN\t_\tNumber=Sing\t2\tnsubj\t_\t_
2\tbarks\tbark\tVERB\t_\t_\t0\troot\t_\t_
"""

SCORED_PRED = """\
# sent_id = a
1\tdog\tdog\tNOUN\t_\tNumber=Plur\t0\tnsubj\t_\tConf=upos:0.99,lemma:0.3,feats:0.6,head:0.85,deprel:0.97
2\tbarks\tbark\tVERB\t_\t_\t0\tobj\t_\tConf=upos:0.2,head:0.99,deprel:0.1
"""


def test_calibration_buckets_by_confidence():
    buckets = calibration(parse_conllu(SCORED_PRED), parse_conllu(SCORED_GOLD) + parse_conllu("# sent_id = z\n1\tx\tx\tX\t_\t_\t0\troot\t_\t_\n"))
    cells = {b.label: (b.cells, b.correct) for b in buckets}
    assert cells == {
        "0.00-0.50": (3, 2),  # lemma and upos right, deprel wrong
        "0.50-0.70": (1, 0),  # feats wrong
        "0.70-0.80": (0, 0),
        "0.80-0.90": (1, 0),  # head wrong
        "0.90-0.95": (0, 0),
        "0.95-1.00": (3, 3),
    }
    assert buckets[-1].accuracy == 1.0 and buckets[2].accuracy == 0.0
    table = calibration_table(buckets)
    assert table.splitlines()[0].startswith("Confidence") and "0.95-1.00" in table
