import threading

import pytest

from mcparser.corpus import CellUpdate, CorpusError, CorpusStore, EditError, NotFoundError, ReadOnlyError
from mcparser.corpus.confidence import read_confidence
from mcparser.corpus.store import normalise_column


def test_sources(store, train_path):
    assert [info.name for info in store.list_sources()] == ["train", "pred"]
    with pytest.raises(CorpusError, match="already exists"):
        store.add_source("train", train_path)
    with pytest.raises(NotFoundError, match="available: train, pred"):
        store.source("nope")
    with pytest.raises(NotFoundError, match="available: none"):
        CorpusStore().source("x")


def test_normalise_column():
    assert normalise_column(" Segmentation ") == "form"
    assert normalise_column("UPOS") == "upos"


# ---------------------------------------------------------------- browsing


def test_list_sentences_orders_and_filters(store):
    page = store.list_sentences("pred")
    assert [s.sent_id for s in page.sentences] == ["p1", "p2"]
    assert page.sentences[0].text == "Vino al mercado."
    store.set_cell("pred", "p2", "1", "upos", "DET", confidence=1.0)
    assert [s.sent_id for s in store.list_sentences("pred", order="confidence_desc").sentences] == ["p2", "p1"]
    assert [s.sent_id for s in store.list_sentences("pred", order="confidence_asc").sentences] == ["p1", "p2"]
    assert store.list_sentences("pred", max_confidence=0.065).total == 1
    assert store.list_sentences("pred", offset=1, limit=1).sentences[0].sent_id == "p2"


def test_get_sentence(store):
    view = store.get_sentence("pred", "p1")
    assert view.conllu.startswith("# sent_id = p1")
    assert view.tokens is None
    assert {issue.code for issue in view.issues} == {"missing-upos", "missing-lemma"}
    assert len(store.get_sentence("pred", "p1", include_tokens=True).tokens) == 4


def test_get_context(store):
    assert [s.sent_id for s in store.get_context("train", "t2", before=1, after=0)] == ["t1", "t2"]
    assert [s.sent_id for s in store.get_context("train", "t1")] == ["t1", "t2", "t3"]


def test_get_token(store):
    view = store.get_token("train", "t3", "3")
    assert view.head.form == "río"
    assert view.multiword.form == "al"
    assert view.confidence["upos"] == 1.0
    root = store.get_token("train", "t3", "2")
    assert root.head is None
    assert [d.form for d in root.dependents] == ["Ella", "río"]
    assert root.multiword is None
    assert store.get_token("train", "t3", "3-4").multiword is None


# ---------------------------------------------------------------- search


def test_find_tokens(store):
    result = store.find_tokens(upos="NOUN", sources=["train"], group_by=["lemma"])
    assert result.total == 3
    assert [(g.values["lemma"], g.count) for g in result.groups] == [("dog", 1), ("mercado", 1), ("río", 1)]
    hit = store.find_tokens(form="río").hits[0]
    assert (hit.source, hit.sent_id, hit.head_form, hit.context) == ("train", "t3", "fue", "fue a el [[río]]")
    assert store.find_tokens(form="RIO", loose=True).total == 1
    assert store.find_tokens(lemma="DOG", loose=True).total == 2
    assert store.find_tokens(root="dg").total == 1
    assert store.find_tokens(root="DG", loose=True).total == 1
    assert store.find_tokens(upos="NOUN", feats="Number=Plur").total == 1
    assert store.find_tokens(feats="Definite=Def").total == 3
    assert store.find_tokens(deprel="obl").total == 4
    assert store.find_tokens(deprel="obl:arg").total == 1
    assert store.find_tokens(form="Vino", upos="VERB").total == 2
    assert store.find_tokens(form="Vino", upos="NOUN").total == 0
    assert store.find_tokens(form="Vino", min_confidence=0.5).total == 1
    assert store.find_tokens(form="Vino", exclude_sentence="pred:p1").total == 1
    assert store.find_tokens(form="Vino", exclude_sentence="t2").total == 1
    assert store.find_tokens(form="Vino", exclude_sentence="train:p1").total == 2
    root = store.find_tokens(form="Vino", sources=["train"]).hits[0]
    assert root.head_form is None
    with pytest.raises(EditError, match="at least one"):
        store.find_tokens()
    with pytest.raises(EditError, match="Unknown column"):
        store.find_tokens(form="x", group_by=["bogus"])


def test_find_tokens_orders_by_confidence(store):
    hits = store.find_tokens(form="mercado").hits
    assert [hit.source for hit in hits] == ["train", "pred"]
    assert hits[1].confidence == 0.0


def test_analyse_form(store):
    result = store.analyse_form("el")
    assert result.total == 2
    assert result.groups[0].values == {"lemma": "el", "upos": "DET", "xpos": "DA0MS0", "feats": "Definite=Def|Gender=Masc|Number=Sing|PronType=Art"}


def test_find_multiword(store):
    result = store.find_multiword("al")
    assert result.total == 1
    assert result.analyses[0].pieces == ["a", "el"]
    assert result.analyses[0].upos == ["ADP", "DET"]
    assert result.analyses[0].examples == ["train:t3:3-4"]
    assert result.as_single_word == 1  # unsplit in the predictions
    assert store.find_multiword("AL", loose=True, sources=["train"]).as_single_word == 0


def test_find_multiword_caps_examples(tmp_path):
    path = tmp_path / "m.conllu"
    block = "1-2\tal\t_\t_\t_\t_\t_\t_\t_\t_\n1\ta\t_\tADP\t_\t_\t2\tcase\t_\t_\n2\tel\t_\tDET\t_\t_\t0\troot\t_\t_\n\n"
    path.write_text("".join(f"# sent_id = {i}\n{block}" for i in range(5)), encoding="utf-8")
    store = CorpusStore()
    store.add_source("m", path)
    result = store.find_multiword("al")
    assert result.analyses[0].count == 5
    assert len(result.analyses[0].examples) == 3


def test_find_arcs(store):
    result = store.find_arcs(dependent_upos="ADP", sources=["train"])
    assert result.total == 2
    assert result.deprels[0].value == "case"
    assert result.directions[0].value == "right"
    hit = result.hits[0]
    assert (hit.dependent.form, hit.head.form, hit.distance) == ("de", "mercado", 2)
    roots = store.find_arcs(head_upos="ROOT", sources=["train"])
    assert roots.total == 3 and roots.hits[0].head is None and roots.hits[0].direction == "root"
    assert store.find_arcs(head_lemma="bark", deprel="nsubj").hits[0].dependent.form == "dogs"
    assert store.find_arcs(deprel="punct", sources=["train"]).directions[0].value == "left"
    assert store.find_arcs(head_form="VINO", loose=True, deprel="obl").total == 3
    assert store.find_arcs(head_upos="ROOT", dependent_form="Vino", min_confidence=0.5).total == 1
    assert store.find_arcs(dependent_form="Vino", exclude_sentence="pred:p1").total == 1
    assert store.find_arcs(dependent_form="de", deprel="nsubj").total == 0
    assert store.find_arcs(dependent_form="de", dependent_upos="NOUN").total == 0
    assert store.find_arcs(dependent_form="de", head_upos="VERB").total == 0
    with pytest.raises(EditError, match="at least one"):
        store.find_arcs()


def test_find_arcs_skips_unknown_and_invalid_heads(tmp_path):
    path = tmp_path / "bad.conllu"
    path.write_text("# sent_id = b\n1\ta\t_\tX\t_\t_\t_\t_\t_\t_\n2\tb\t_\tX\t_\t_\t9\tdep\t_\t_\n\n", encoding="utf-8")
    store = CorpusStore()
    store.add_source("bad", path)
    assert store.find_arcs(dependent_upos="X").total == 0
    assert store.find_tokens(form="b").hits[0].head_form is None


def test_search_sentences(store):
    hits = store.search_sentences("mercado")
    assert [(hit.source, hit.sent_id) for hit in hits] == [("train", "t2"), ("pred", "p1")]
    assert [hit.sent_id for hit in store.search_sentences("mercado", min_confidence=0.5)] == ["t2"]
    assert [hit.sent_id for hit in store.search_sentences("mercado", exclude_sentence="pred:p1")] == ["t2"]
    assert len(store.search_sentences("mercado", limit=1)) == 1


# ---------------------------------------------------------------- tagsets


def test_allowed_values(store):
    upos = {item.value for item in store.allowed_values("pos")}
    assert "NOUN" in upos and "_" not in upos
    assert [item.value for item in store.allowed_values("xpos", upos="ADP")] == ["SPS00"]
    assert store.allowed_values("deprel", upos="NOPE") == []
    assert {item.value for item in store.allowed_values("upos", sources=["pred"])} == {"VERB", "NOUN", "PUNCT", "DET"}
    with pytest.raises(EditError, match="supports"):
        store.allowed_values("lemma")


def test_feature_inventory(store):
    inventory = store.feature_inventory()
    assert list(inventory) == ["Definite", "Gender", "Mood", "Number", "Person", "PronType", "Tense", "VerbForm"]
    assert {item.value for item in inventory["Number"]} == {"Plur", "Sing"}
    assert set(store.feature_inventory(upos="DET")) == {"Definite", "Gender", "Number", "PronType"}


def test_trusted_defaults_to_all_sources_without_read_only(pred_path):
    store = CorpusStore()
    store.add_source("pred", pred_path, mode="rw")
    assert {item.value for item in store.allowed_values("upos")} == {"VERB", "NOUN", "PUNCT", "DET"}
    assert store.known_values() == {}


# ---------------------------------------------------------------- quality


def test_validate(store):
    report = store.validate("pred")
    assert report.errors == 0 and report.warnings > 0
    assert store.validate("pred", sent_id="p2").total == 0
    assert store.validate("pred", sent_id="p1").total == 2
    assert len(store.validate("pred", limit=2).issues) == 2
    store.set_cell("pred", "p2", "1", "head", "1")
    report = store.validate("pred")
    assert report.issues[0].issue.severity == "error"


def test_low_confidence(store):
    page = store.low_confidence("pred", threshold=0.85, columns=["pos", "lemma"], sent_id="p1")
    assert [(c.token_id, c.column) for c in page.cells][:2] == [("2", "upos"), ("2", "lemma")]
    assert ("1", "lemma") not in [(c.token_id, c.column) for c in page.cells]
    assert ("1", "upos") in [(c.token_id, c.column) for c in page.cells]
    assert page.cells[-1].confidence == 0.8
    assert store.low_confidence("pred").total == 6 * 7 - 2
    assert store.low_confidence("train").total == 0
    with pytest.raises(EditError, match="has no confidence"):
        store.low_confidence("pred", columns=["misc"])


# ---------------------------------------------------------------- editing


def test_set_cell_records_history_and_confidence(store, pred_path):
    store.round = 2
    result = store.set_cell("pred", "p1", "2", "UPOS", "ADP", confidence=0.6, reason="like al", agent="pos")
    edit = result.edits[0]
    assert (edit.old, edit.new, edit.column, edit.agent, edit.round, edit.seq) == ("_", "ADP", "upos", "pos", 2, 1)
    assert read_confidence(store.source("pred").sentence("p1").token("2")) == {"upos": 0.6}
    assert "Conf=upos:0.60" in pred_path.read_text(encoding="utf-8")
    store.set_cell("pred", "p1", "2", "upos", "ADV")
    assert read_confidence(store.source("pred").sentence("p1").token("2")) == {}
    history = store.get_history("pred", sent_id="p1", token_id="2")
    assert [edit.new for edit in history] == ["ADV", "ADP"]
    assert store.get_history("pred", limit=1)[0].seq == 2


def test_update_cells_is_atomic(store):
    with pytest.raises(EditError, match="universal POS"):
        store.update_cells(
            "pred",
            "p2",
            [{"token_id": "1", "column": "lemma", "value": "x"}, {"token_id": "2", "column": "upos", "value": "N"}],
        )
    assert store.source("pred").sentence("p2").token("1").lemma == "the"
    assert store.get_history("pred") == []


def test_update_cells_confidence_only_and_misc(store):
    result = store.update_cells(
        "pred",
        "p2",
        [
            CellUpdate(token_id="1", column="head", confidence=0.9, reason="det before noun"),
            {"token_id": "2", "column": "misc", "value": "Gloss=dog|Conf=head:0.10"},
        ],
        agent="scorer",
    )
    assert [edit.action for edit in result.edits] == ["confidence", "set"]
    sentence = store.source("pred").sentence("p2")
    assert read_confidence(sentence.token("1")) == {"head": 0.9}
    assert sentence.token("2").misc == "Gloss=dog"


@pytest.mark.parametrize(
    ("update", "error", "message"),
    [
        ({"token_id": "1", "column": "id", "value": "3"}, EditError, "cannot be edited"),
        ({"token_id": "1", "column": "upos"}, EditError, "value, a confidence"),
        ({"token_id": "1", "column": "misc", "confidence": 0.5}, EditError, "has no confidence"),
        ({"token_id": "1", "column": "misc", "value": "A=1", "confidence": 0.5}, EditError, "has no confidence"),
        ({"token_id": "1", "column": "head", "value": "7"}, EditError, "HEAD must be"),
        ({"token_id": "1", "column": "head", "value": "x"}, EditError, "HEAD must be"),
        ({"token_id": "9", "column": "upos", "value": "X"}, NotFoundError, "No token"),
    ],
)
def test_update_cells_errors(store, update, error, message):
    with pytest.raises(error, match=message):
        store.update_cells("pred", "p2", [update])


def test_edits_on_read_only_and_multiword(store):
    with pytest.raises(ReadOnlyError):
        store.set_cell("train", "t1", "1", "upos", "X")
    store.split_word("pred", "p1", "2", ["a", "el"], multiword=True, head_index=1)
    with pytest.raises(EditError, match="Only FORM and MISC"):
        store.set_cell("pred", "p1", "2-3", "upos", "X")
    store.set_cell("pred", "p1", "2-3", "misc", "SpaceAfter=No")
    store.set_cell("pred", "p1", "2", "head", "_")


def test_without_autosave(pred_path):
    store = CorpusStore(autosave=False)
    store.add_source("pred", pred_path, mode="rw")
    store.set_cell("pred", "p2", "1", "upos", "PRON")
    assert "\tPRON\t" not in pred_path.read_text(encoding="utf-8")
    assert store.save() == ["pred"]
    assert store.save("pred") == ["pred"]
    assert "\tPRON\t" in pred_path.read_text(encoding="utf-8")
    store.set_cell("pred", "p2", "1", "upos", "X")
    store.reload("pred")
    assert store.source("pred").sentence("p2").token("1").upos == "PRON"


def test_segmentation_edits_update_flags_and_raise_head_flag(store):
    store.flag_issue("pred", "p1", "lemma", "check", token_id="3")
    store.flag_issue("pred", "p1", "upos", "sentence-level")
    result = store.split_word("pred", "p1", "2", ["a", "el"], multiword=True, head_index=1, confidence=0.7, reason="like train", agent="seg")
    assert result.id_map == {"1": "1", "2": "3", "3": "4", "4": "5"}
    assert result.new_ids == ["2", "3"]
    assert result.edits[0].old == "Vino al mercado ."
    assert result.edits[0].new == "Vino a el mercado ."
    sentence = store.source("pred").sentence("p1")
    assert read_confidence(sentence.token("2")) == {"form": 0.7}
    flags = store.list_flags("pred").flags
    assert [(f.column, f.token_id) for f in flags] == [("lemma", "4"), ("upos", None), ("head", None)]
    assert flags[2].raised_by == "seg"

    store.merge_words("pred", "p1", ["2", "3"], confidence=0.9)
    assert store.source("pred").sentence("p1").word_forms() == ["Vino", "al", "mercado", "."]
    store.insert_word("pred", "p1", "4", "!", confidence=0.5)
    store.delete_word("pred", "p1", "5")
    store.add_multiword("pred", "p1", 2, 3, "almercado")
    store.remove_multiword("pred", "p1", "2-3")
    actions = [edit.action for edit in store.get_history("pred")][::-1]
    assert actions == [
        "flag", "flag", "split_word", "flag", "merge_words", "flag", "insert_word", "flag",
        "delete_word", "flag", "add_multiword", "remove_multiword",
    ]  # fmt: skip
    assert store.list_flags("pred", column="head").total == 4


def test_segmentation_flag_on_deleted_word_becomes_sentence_level(store):
    flag = store.flag_issue("pred", "p1", "upos", "odd", token_id="4")
    store.delete_word("pred", "p1", "4")
    assert flag.token_id is None


def test_flags(store):
    flag = store.flag_issue("pred", "p2", "Segmentation", " maybe two words ", token_id="2", suggestion="do g", agent="lemma")
    assert (flag.id, flag.column, flag.reason, flag.raised_by) == (1, "form", "maybe two words", "lemma")
    other = store.flag_issue("pred", "p1", "deprel", "wrong")
    assert store.list_flags("pred", column="segmentation").flags == [flag]
    assert store.list_flags("pred", sent_id="p1").flags == [other]
    assert store.list_flags("pred", limit=1).total == 2
    resolved = store.resolve_flag("pred", 1, note="fine", agent="seg")
    assert (resolved.status, resolved.resolved_by, resolved.note) == ("resolved", "seg", "fine")
    store.resolve_flag("pred", 2, status="dismissed")
    assert store.list_flags("pred").total == 0
    assert store.list_flags("pred", status="dismissed").flags[0].id == 2
    assert store.list_flags("pred", status="all").total == 2
    assert store.list_sources()[1].open_flags == 0
    with pytest.raises(EditError, match="already resolved"):
        store.resolve_flag("pred", 1)
    with pytest.raises(NotFoundError, match="No flag 9"):
        store.resolve_flag("pred", 9)
    with pytest.raises(EditError, match="'resolved' or 'dismissed'"):
        store.resolve_flag("pred", 1, status="open")
    with pytest.raises(EditError, match="needs a reason"):
        store.flag_issue("pred", "p1", "upos", "  ")
    with pytest.raises(EditError, match="has no confidence"):
        store.flag_issue("pred", "p1", "misc", "x")
    with pytest.raises(ReadOnlyError):
        store.flag_issue("train", "t1", "upos", "x")


def test_store_is_thread_safe(store):
    def worker(i):
        store.set_cell("pred", "p2", "1", "misc", f"N={i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(store.get_history("pred", limit=100)) == 10


def test_score_cells(store):
    result = store.score_cells("pred", "p1", {"2:*": 0.7, "1:upos": 0.95, "1:pos": 0.9, "3": 0.5}, agent="scorer")
    al = store.source("pred").sentence("p1").token("2")
    assert read_confidence(al) == {"form": 0.7, "xpos": 0.7, "feats": 0.7, "head": 0.7, "deprel": 0.7}  # empty LEMMA/UPOS skipped
    assert read_confidence(store.source("pred").sentence("p1").token("1"))["upos"] == 0.9  # later key wins
    assert read_confidence(store.source("pred").sentence("p1").token("3"))["lemma"] == 0.5
    assert {edit.action for edit in result.edits} == {"confidence"}
    with pytest.raises(NotFoundError):
        store.score_cells("pred", "p1", {"9:*": 0.5})
    store.set_cell("pred", "p2", "1", "lemma", "_")
    store.set_cell("pred", "p2", "1", "upos", "_")
    store.set_cell("pred", "p2", "1", "head", "_")
    store.set_cell("pred", "p2", "1", "deprel", "_")
    for column in ("form", "xpos", "feats"):
        store.set_cell("pred", "p2", "1", column, "_" if column != "form" else "the")
    with pytest.raises(EditError, match="No cells to score"):
        store.score_cells("pred", "p2", {})


def test_find_tokens_by_suffix_and_context(store):
    assert store.find_tokens(suffix="ADO", sources=["train"]).total == 1  # indexed length (loose)
    assert store.find_tokens(suffix="do", upos="NOUN", sources=["train"]).total == 1  # scanned
    assert store.find_tokens(suffix="o", sources=["train"]).total == 3  # Vino, mercado, río (unindexed length)
    assert store.find_tokens(form="el", previous_form="DE", sources=["train"]).total == 1
    assert store.find_tokens(form="el", next_form="río", sources=["train"]).total == 1
    assert store.find_tokens(form="Vino", previous_form="x", sources=["train"]).total == 0  # first word
    assert store.analyse_form("el", previous_form="a", sources=["train"]).groups[0].count == 1
    assert store.find_tokens(form="el", next_upos="NOUN", sources=["train"]).total == 2
    assert store.find_tokens(form="el", next_upos="VERB", sources=["train"]).total == 0
    assert store.analyse_form("el", previous_upos="ADP", sources=["train"]).total == 2


def test_find_arcs_by_direction(store):
    assert store.find_arcs(deprel="punct", direction="left", sources=["train"]).total == 2
    assert store.find_arcs(deprel="punct", direction="right", sources=["train"]).total == 0
    assert store.find_arcs(head_upos="ROOT", direction="root", sources=["train"]).total == 3


def test_relation_stats_and_xpos_for_tags(store):
    det = store.relation_stats("el", "DET", "mercado", "NOUN", "right")
    assert (det.basis, det.total) == ("UPOS+head UPOS", 3) and det.deprels[0].value == "det"
    assert store.relation_stats("el", "DET", "río", "NOUN", "right", min_total=1).basis == "lemma+head lemma"
    assert store.relation_stats("venir", "VERB", None, None, "root", min_total=1).basis == "lemma+head lemma"
    assert store.relation_stats("zz", "INTJ", "x", "X", "left").basis == "none"
    assert store.xpos_for_tags("NOUN", "Gender=Masc|Number=Sing")[0].value == "NCMS000"
    assert store.xpos_for_tags("NOUN", "Case=Abl") == []


def test_dry_run_changes_nothing(store):
    result = store.update_cells("pred", "p2", [{"token_id": "1", "column": "head", "value": "1"}], dry_run=True)
    assert any(issue.code == "self-loop" for issue in result.issues)
    assert store.source("pred").sentence("p2").token("1").head == "2"
    assert store.get_history("pred") == []
    store.score_cells("pred", "p2", {"1:*": 0.5}, dry_run=True)
    assert read_confidence(store.source("pred").sentence("p2").token("1")) == {}


def test_unchained_relations(store):
    first = store.unchained_relations(min_count=1, max_share=1)
    assert set(first) <= set(store.known_values()["deprel"])
    assert all(chained <= count for count, chained in first.values())
    assert store.unchained_relations(min_count=10**6) == {}
