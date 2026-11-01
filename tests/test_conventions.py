import pytest

from mcparser.agents.evidence import EvidenceBuilder
from mcparser.agents.roles import ROLES
from mcparser.corpus import CorpusStore, parse_conllu
from mcparser.settings import Settings


def word(number, upos, feats):
    return f"{number}\tx\tx\t{upos}\t_\t{feats}\t0\troot\t_\t_"


@pytest.fixture
def treebank(tmp_path):
    """DET in the genitive singular: 30 Masc and 10 Masc,Neut; in the nominative: 30 Masc; ADJ too few to count."""
    sentences = []
    for number in range(30):
        lines = [word(1, "DET", "Case=Gen|Gender=Masc|Number=Sing"), word(2, "DET", "Case=Nom|Gender=Masc|Number=Sing")]
        if number < 10:
            lines.append(word(3, "DET", "Case=Gen|Gender=Masc,Neut|Number=Sing"))
        lines.append(word(len(lines) + 1, "ADJ", "Case=Gen|Gender=Masc,Neut|Number=Sing") if number == 0 else word(len(lines) + 1, "X", "_"))
        sentences.append(f"# sent_id = t{number}\n" + "\n".join(lines) + "\n\n")
    path = tmp_path / "train.conllu"
    path.write_text("".join(sentences), encoding="utf-8")
    pred = tmp_path / "pred.conllu"
    pred.write_text("# sent_id = p1\n1\tx\t_\tDET\t_\t_\t_\t_\t_\t_\n\n", encoding="utf-8")
    store = CorpusStore()
    store.add_source("train", path)
    store.add_source("pred", pred, mode="rw")
    return store


def test_feature_conventions(treebank):
    found = treebank.feature_conventions("DET")
    assert [(c.case, c.number, c.feature, c.total) for c in found] == [("Gen", "Sing", "Gender", 40)]
    assert [(v.value, v.count) for v in found[0].values] == [("Masc", 30), ("Masc,Neut", 10)]
    assert treebank.feature_conventions("DET", min_share=0.5) == []  # a quarter of the examples is not enough
    assert treebank.feature_conventions("DET", min_count=50) == []
    assert treebank.feature_conventions("ADJ") == []  # one example
    assert treebank.feature_conventions("NOUN") == []


def test_feats_evidence_lists_the_conventions(treebank):
    line = EvidenceBuilder(treebank, "pred").build(ROLES["feats"], [("p1", [])]).splitlines()[-1]
    assert line == "- Combined values the treebank uses with DET (case number: shares): Gender with Gen Sing: Masc 75%, Masc,Neut 25%"
    quiet = EvidenceBuilder(treebank, "pred", conventions=False).build(ROLES["feats"], [("p1", [])])
    assert "Combined values" not in quiet
    assert "Combined values" not in EvidenceBuilder(treebank, "pred").build(ROLES["xpos"], [("p1", [])])


def test_conventions_are_a_setting():
    assert Settings().conventions and not Settings(conventions=False).conventions
    assert parse_conllu("# sent_id = a\n" + word(1, "X", "_") + "\n\n")  # the helper writes valid CoNLL-U
