from llmparser.baseline import evidence_confidence, run_baseline
from llmparser.corpus.confidence import read_confidence


def test_evidence_confidence():
    assert evidence_confidence(1, 1) == 0.33
    assert evidence_confidence(3, 3) == 0.7  # a candidate only: always reviewed by an agent
    assert evidence_confidence(3, 6) == 0.5


def test_baseline_splits_and_analyses(store):
    store.set_cell("pred", "p2", "1", "lemma", "keep")
    report = run_baseline(store, "pred")
    assert report.model_dump() == {"sentences": 2, "splits": 1, "analysed": 6, "unknown": 1}
    p1 = store.source("pred").sentence("p1")
    assert [(t.id, t.form) for t in p1.tokens] == [("1", "Vino"), ("2-3", "al"), ("2", "a"), ("3", "el"), ("4", "mercado"), ("5", ".")]
    assert (p1.token("2").upos, p1.token("3").head) == ("ADP", "2")  # no content tag in "a el", so "a" heads the split
    assert read_confidence(p1.token("3"))["form"] == 0.33
    assert p1.token("4").xpos == "NCMS000"
    assert p1.token("1").lemma == "venir"  # existing values are kept
    p2 = store.source("pred").sentence("p2")
    assert (p2.token("1").lemma, p2.token("1").xpos) == ("keep", "DT")
    assert store.list_flags("pred").total == 0
    assert {edit.agent for edit in store.get_history("pred", limit=100)} == {"baseline", "agent"}


def test_baseline_loose_match_and_unknown_words(store):
    store.split_word("pred", "p2", "2", ["Dogs", "zzz"], flag_head=False)
    report = run_baseline(store, "pred")
    p2 = store.source("pred").sentence("p2")
    assert p2.token("2").lemma == "dog"  # "Dogs" matched "dogs" loosely
    assert report.unknown == 1  # "zzz"


def test_baseline_is_idempotent(store):
    run_baseline(store, "pred")
    before = store.source("pred").sentence("p1").to_conllu()
    report = run_baseline(store, "pred")
    assert (report.splits, report.analysed) == (0, 0)
    assert store.source("pred").sentence("p1").to_conllu() == before
