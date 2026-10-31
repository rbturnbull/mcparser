from collections import Counter

import pytest

from mcparser.corpus import CorpusStore
from mcparser.corpus.confidence import read_confidence
from mcparser.rules import Rules, RuleSettings, dominant

ROW = "{id}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t_\t{misc}"


def conllu(*sentences):
    """Sentences as lists of (form, lemma, upos, xpos, feats, head, deprel) rows; "a-b" rows are ranges."""
    blocks = []
    for number, rows in enumerate(sentences):
        lines = [f"# sent_id = s{number}"]
        word = 0
        for row in rows:
            if row[0].startswith("~"):  # multiword token covering the next two words
                lines.append(ROW.format(id=f"{word + 1}-{word + 2}", form=row[0][1:], lemma="_", upos="_", xpos="_", feats="_", head="_", deprel="_", misc="_"))
                continue
            word += 1
            form, lemma, upos, xpos, feats, head, deprel = row
            lines.append(ROW.format(id=word, form=form, lemma=lemma, upos=upos, xpos=xpos, feats=feats, head=head, deprel=deprel, misc="_"))
        blocks.append("\n".join(lines) + "\n\n")
    return "".join(blocks)


DOG = [("the", "the", "DET", "DT", "_", "2", "det"), ("dog", "dog", "NOUN", "NN", "Number=Sing", "3", "nsubj"), ("barks", "bark", "VERB", "VBZ", "_", "0", "root")]
DEL = [("~del",), ("de", "de", "ADP", "SP", "_", "2", "case"), ("el", "el", "DET", "DA", "_", "3", "det"), ("río", "río", "NOUN", "NC", "_", "0", "root")]
SAW = [("I", "I", "PRON", "PRP", "_", "2", "nsubj"), ("saw", "see", "VERB", "VBD", "_", "0", "root")]
SAW_NOUN = [("a", "a", "DET", "DT", "_", "2", "det"), ("saw", "saw", "NOUN", "NN", "_", "0", "root")]


@pytest.fixture
def make(tmp_path):
    def build(train, pred, **settings):
        (tmp_path / "train.conllu").write_text(conllu(*train), encoding="utf-8")
        (tmp_path / "pred.conllu").write_text(pred, encoding="utf-8")
        store = CorpusStore()
        store.add_source("train", tmp_path / "train.conllu")
        store.add_source("pred", tmp_path / "pred.conllu", mode="rw")
        return store, Rules(store, "pred", RuleSettings(min_count=3, **settings))

    return build


def blank(*forms_per_sentence):
    blocks = []
    for number, forms in enumerate(forms_per_sentence):
        rows = [f"{i}\t{form}\t_\t_\t_\t_\t_\t_\t_\t_" for i, form in enumerate(forms, start=1)]
        blocks.append(f"# sent_id = p{number}\n" + "\n".join(rows) + "\n\n")
    return "".join(blocks)


def test_dominant():
    settings = RuleSettings(min_count=3, min_share=0.75)
    assert dominant(Counter(), settings) is None
    assert dominant(Counter({"a": 2}), settings) is None
    assert dominant(Counter({"a": 3, "b": 2}), settings) is None
    assert dominant(Counter({"a": 3, "b": 1}), settings) == "a"


def test_morphology_fills_and_confirms(make):
    store, rules = make([DOG] * 3, blank(["the", "dog", "barks"]))
    store.set_cell("pred", "p0", "2", "lemma", "dog", confidence=0.5)
    store.set_cell("pred", "p0", "3", "lemma", "barking", confidence=0.5)
    report = rules.apply()
    words = store.source("pred").sentence("p0").words
    assert [(w.lemma, w.upos, w.xpos, w.feats) for w in words] == [
        ("the", "DET", "DT", "_"), ("dog", "NOUN", "NN", "Number=Sing"), ("barking", "VERB", "VBZ", "_"),
    ]  # fmt: skip
    assert read_confidence(words[1])["lemma"] == 0.97  # confirmed
    assert read_confidence(words[2])["lemma"] == 0.5  # a different value is never replaced
    assert report.cells == {"form": 3, "lemma": 2, "upos": 3, "xpos": 3, "feats": 3}
    assert report.total == 14
    assert rules.apply().total == 0  # nothing left to do
    assert {e.agent for e in store.get_history("pred", limit=100) if e.action == "set"} == {"rules", "agent"}


def test_context_decides_ambiguous_forms(make):
    store, rules = make([SAW] * 3 + [SAW_NOUN] * 3, blank(["I", "saw"], ["a", "saw"], ["saw"]))
    rules.apply(columns=["upos"])
    assert store.source("pred").sentence("p0").token("2").upos == "VERB"  # after "I"
    assert store.source("pred").sentence("p1").token("2").upos == "NOUN"  # after "a"
    assert store.source("pred").sentence("p2").token("1").upos == "_"  # no context: 50/50


def test_xpos_from_confident_tags(make):
    train = [[("x", "x", "NOUN", "NN", "Number=Sing", "0", "root")], [("y", "y", "NOUN", "NN", "Number=Sing", "0", "root")]] * 2
    store, rules = make(train, blank(["z", "w"]))
    store.update_cells("pred", "p0", [
        {"token_id": "1", "column": "upos", "value": "NOUN", "confidence": 0.95},
        {"token_id": "1", "column": "feats", "value": "Number=Sing", "confidence": 0.95},
        {"token_id": "2", "column": "upos", "value": "NOUN", "confidence": 0.5},
        {"token_id": "2", "column": "feats", "value": "Number=Sing", "confidence": 0.95},
    ])  # fmt: skip
    rules.apply(columns=["xpos"])
    sentence = store.source("pred").sentence("p0")
    assert (sentence.token("1").xpos, sentence.token("2").xpos) == ("NN", "_")  # word 2's UPOS is only a guess


def test_deprel_rules(make):
    store, rules = make([DOG] * 3, blank(["the", "dog", "barks"], ["the", "dog"]))
    rules.apply(columns=["upos", "lemma"])
    store.update_cells("pred", "p0", [
        {"token_id": "1", "column": "head", "value": "2", "confidence": 0.95},
        {"token_id": "2", "column": "head", "value": "3", "confidence": 0.95},
        {"token_id": "3", "column": "head", "value": "0", "confidence": 0.95},
    ])  # fmt: skip
    store.update_cells("pred", "p1", [
        {"token_id": "1", "column": "head", "value": "2", "confidence": 0.5},
        {"token_id": "2", "column": "head", "value": "0", "confidence": 0.5},
    ])  # fmt: skip
    rules.apply(columns=["deprel"])
    assert [w.deprel for w in store.source("pred").sentence("p0").words] == ["det", "nsubj", "root"]
    assert [w.deprel for w in store.source("pred").sentence("p1").words] == ["_", "_"]  # heads are only guesses
    assert read_confidence(store.source("pred").sentence("p0").token("3"))["deprel"] == 0.99


def test_segmentation_rules(make):
    store, rules = make([DEL] * 3 + [[("dell", "dell", "PROPN", "NP", "_", "0", "root")]] * 3, blank(["del", "río"], ["dell"]))
    report = rules.apply(columns=["form"])
    assert report.splits == 1
    tokens = store.source("pred").sentence("p0").tokens
    assert [(t.id, t.form) for t in tokens] == [("1-2", "del"), ("1", "de"), ("2", "el"), ("3", "río")]
    assert read_confidence(tokens[1])["form"] == 0.97
    assert read_confidence(store.source("pred").sentence("p1").token("1"))["form"] == 0.97  # never split
    assert read_confidence(tokens[3])["form"] == 0.97  # río is never split either


def test_no_multiword_tokens_confirms_every_word(make):
    store, rules = make([DOG], blank(["un", "known"]))
    assert rules.apply(columns=["form"]).cells == {"form": 2}


def test_confident_predictions_are_evidence(make, tmp_path):
    pred = blank(["zorg"], ["zorg"], ["zorg"], ["zorg"], ["blip"])
    store, rules = make([DOG], pred)
    for sent_id in ("p0", "p1", "p2"):
        store.update_cells("pred", sent_id, [{"token_id": "1", "column": "upos", "value": "X", "confidence": 0.98}], agent="upos")
    rules.apply(sent_ids=["p3"], columns=["upos"])
    word = store.source("pred").sentence("p3").token("1")
    assert (word.upos, read_confidence(word)["upos"]) == ("X", 0.95)
    store2, rules2 = make([DOG], pred, use_predictions=False)
    assert rules2.apply(sent_ids=["p3"], columns=["upos"]).total == 0


def test_cross_check_flags_upos_that_does_not_fit_the_relation(make):
    det = [("omnis", "omnis", "DET", "_", "_", "2", "det"), ("homo", "homo", "NOUN", "_", "_", "0", "root")]
    pron = [("omnis", "omnis", "PRON", "_", "_", "2", "nsubj"), ("venit", "venio", "VERB", "_", "_", "0", "root")]
    store, rules = make([det] * 3 + [pron] * 3, blank(["omnis", "venit"], ["omnis", "homo"]))
    for sent_id, deprel in (("p0", "nsubj"), ("p1", "det")):
        store.update_cells("pred", sent_id, [
            {"token_id": "1", "column": "lemma", "value": "omnis", "confidence": 0.97},
            {"token_id": "1", "column": "upos", "value": "DET", "confidence": 0.6},
            {"token_id": "1", "column": "deprel", "value": deprel, "confidence": 0.6},
        ], agent="deprel")  # fmt: skip
    report = rules.apply(columns=["deprel"])
    flags = store.list_flags("pred").flags
    assert report.flags == 1
    assert [(f.sent_id, f.token_id, f.column, f.suggestion, f.raised_by) for f in flags] == [("p0", "1", "upos", "PRON", "rules")]
    assert flags[0].reason == "In the treebank, 'omnis' as nsubj is PRON, not DET."
    assert rules.apply(columns=["deprel"]).flags == 0  # not raised twice
    store.update_cells("pred", "p1", [{"token_id": "1", "column": "deprel", "value": "nsubj", "confidence": 0.3}], agent="deprel")
    assert rules.cross_check(store.source("pred").sentence("p1")) == []  # relation too uncertain to check against


def test_relation_check_flags_labels_that_contradict_the_treebank(make):
    store, rules = make([DOG] * 3, blank(["the", "dog", "barks"]))
    rules.apply(columns=["upos", "lemma"])
    store.update_cells("pred", "p0", [
        {"token_id": "1", "column": "head", "value": "2", "confidence": 0.9},
        {"token_id": "1", "column": "deprel", "value": "amod", "confidence": 0.9},
        {"token_id": "2", "column": "head", "value": "3", "confidence": 0.3},
        {"token_id": "2", "column": "deprel", "value": "obj", "confidence": 0.9},
        {"token_id": "3", "column": "head", "value": "0", "confidence": 0.9},
        {"token_id": "3", "column": "deprel", "value": "root", "confidence": 0.9},
    ], agent="syntax")  # fmt: skip
    sentence = store.source("pred").sentence("p0")
    assert rules.relation_check(sentence) == [("1", "amod", "det", "3 of 3, by lemma+head lemma")]  # word 2: head a guess
    report = rules.apply(columns=["deprel"])
    flag = store.list_flags("pred", column="deprel").flags[0]
    assert report.flags == 1 and (flag.token_id, flag.suggestion, flag.raised_by) == ("1", "det", "rules")
    assert flag.reason.startswith("The treebank labels arcs like this det (3 of 3")
    assert rules.apply(columns=["upos"]).flags == 0  # only checked with DEPREL


def test_voice_check_flags_subjects_that_ignore_the_heads_voice(make):
    active = [("dog", "dog", "NOUN", "_", "_", "2", "nsubj"), ("bites", "bite", "VERB", "_", "Voice=Act", "0", "root")]
    passive = [("dog", "dog", "NOUN", "_", "_", "2", "nsubj:pass"), ("bitten", "bite", "VERB", "_", "Voice=Pass", "0", "root")]
    aux = [("is", "be", "AUX", "_", "_", "2", "aux"), ("bitten", "bite", "VERB", "_", "Voice=Pass", "0", "root")]
    aux_active = [("has", "have", "AUX", "_", "_", "2", "aux"), ("bitten", "bite", "VERB", "_", "Voice=Act", "0", "root")]
    store, rules = make([active] * 3 + [passive] * 3 + [DOG] * 3 + [aux, aux_active] * 3, blank(["dog", "bitten"], ["dog", "bites"], ["dog", "ran"], ["a", "b"]))

    def label(sent_id, deprel, verb_feats, confidence=0.9):
        store.update_cells("pred", sent_id, [
            {"token_id": "1", "column": "deprel", "value": deprel, "confidence": confidence},
            {"token_id": "1", "column": "head", "value": "2", "confidence": 0.9},
            {"token_id": "2", "column": "feats", "value": verb_feats, "confidence": 0.9},
        ], agent="syntax")  # fmt: skip

    label("p0", "nsubj", "Voice=Pass")  # passive head, plain label
    label("p1", "nsubj:pass", "Voice=Act")  # active head, passive label
    label("p2", "nsubj", "Voice=Act")  # fine
    sentences = store.source("pred")
    assert rules.voice_check(sentences.sentence("p0")) == [("1", "nsubj", "nsubj:pass", "3 of 3 relations like this have a passive head")]
    assert rules.voice_check(sentences.sentence("p1")) == [("1", "nsubj:pass", "nsubj", "6 of 6 relations like this have a not passive head")]
    assert rules.voice_check(sentences.sentence("p2")) == []
    label("p3", "aux", "Voice=Pass")
    assert rules.voice_check(sentences.sentence("p3")) == []  # the head's voice makes no difference to aux here
    label("p3", "obl", "Voice=Pass")
    assert rules.voice_check(sentences.sentence("p3")) == []  # not a relation whose subtype follows voice
    label("p0", "nsubj", "_")
    assert rules.voice_check(sentences.sentence("p0")) == []  # head without FEATS
    label("p0", "nsubj", "Voice=Pass", confidence=0.3)
    assert rules.voice_check(sentences.sentence("p0")) == []  # label too uncertain
    label("p0", "nsubj:foo", "Voice=Pass")
    assert rules.voice_check(sentences.sentence("p0")) == [("1", "nsubj:foo", "nsubj:pass", "3 of 3 relations like this have a passive head")]
    label("p0", "nsubj:pass", "Voice=Pass")
    assert rules.voice_check(sentences.sentence("p0")) == []  # already right
    label("p0", "nsubj", "Voice=Pass")
    report = rules.apply(columns=["feats"])
    flag = store.list_flags("pred", column="deprel", sent_id="p0").flags[0]
    assert flag.suggestion == "nsubj:pass" and flag.raised_by == "rules" and report.flags >= 1
    assert flag.reason.startswith("The treebank labels this nsubj:pass (3 of 3")
