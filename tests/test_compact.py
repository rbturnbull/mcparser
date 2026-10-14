import json

import pytest

from llmparser.agents.compact import compact_result, counts, misc
from llmparser.agents.evidence import EvidenceBuilder
from llmparser.agents.roles import ROLES
from llmparser.corpus import CorpusStore


def dumped(value):
    """A store result as the agents receive it from MCP (decoded JSON)."""
    if isinstance(value, list):
        return [json.loads(item.model_dump_json()) for item in value]
    if isinstance(value, dict):
        return {key: [json.loads(v.model_dump_json()) for v in items] for key, items in value.items()}
    return json.loads(value.model_dump_json())


def show(store, name, result, role="upos"):
    return compact_result(name, dumped(result), ROLES[role])


def test_helpers():
    assert counts([{"value": "A", "count": 2}, {"value": "B", "count": 1}], limit=1) == "A×2, … (+1 more)"
    assert counts([]) == "none"
    assert misc("SpaceAfter=No|Conf=upos:0.50") == "SpaceAfter=No"


def test_searches(store):
    text = show(store, "analyse_form", store.analyse_form("el"))
    assert text.splitlines()[0] == "2 matches. By lemma upos xpos feats: el DET DA0MS0 Definite=Def|Gender=Masc|Number=Sing|PronType=Art ×2"
    assert "- train:t2:3 el el DET DA0MS0" in text and "head=4(mercado) det | Vino de [[el]] mercado ." in text
    assert show(store, "find_tokens", store.find_tokens(form="nothing")) == "No matches."
    pred_hit = show(store, "find_tokens", store.find_tokens(form="Vino", sources=["pred"]))
    assert "head=0(ROOT)" in pred_hit and "conf=0.00" in pred_hit  # weakest morphology cell (XPOS)
    store.set_cell("pred", "p2", "1", "head", "_")
    assert "head=_" in show(store, "find_tokens", store.find_tokens(form="the", sources=["pred"]))
    assert "Root=dg" in show(store, "find_tokens", store.find_tokens(root="dg"))

    arcs = show(store, "find_arcs", store.find_arcs(dependent_upos="ADP", sources=["train"]))
    assert arcs.splitlines()[0] == "2 arcs. Relations: case×2. Head is to the: right×2."
    assert "de(ADP, 2) -case-> mercado(NOUN, 4)" in arcs
    roots = show(store, "find_arcs", store.find_arcs(dependent_form="Vino"))
    assert "-root-> ROOT conf=0.00" in roots

    mw = show(store, "find_multiword", store.find_multiword("al"))
    assert mw.splitlines() == ["'al': split 1×, unsplit 1×.", "- a + el (ADP DET) ×1, e.g. train:t3:3-4"]
    hits = show(store, "search_sentences", store.search_sentences("mercado"))
    assert hits.startswith("- train:t2 (score")
    assert show(store, "search_sentences", []) == "No similar sentences."
    assert show(store, "get_context", store.get_context("train", "t2", before=0, after=0)).startswith("- t2 (conf 1.00): Vino")
    assert show(store, "get_context", []) == "None."


def test_tagsets_and_quality(store):
    assert show(store, "allowed_values", store.allowed_values("upos")).startswith("DET×3, NOUN×3, VERB×3")
    assert "Definite: Def×3" in show(store, "feature_inventory", store.feature_inventory(upos="DET"))
    assert show(store, "feature_inventory", {}) == "No features."
    report = show(store, "validate", store.validate("pred"), role="lemma")
    assert report.splitlines() == ["0 errors.", "- p1 warning: Word 2 (al) has no LEMMA."]
    assert show(store, "list_flags", store.list_flags("pred")) == "No flags."
    flag = store.flag_issue("pred", "p1", "upos", "odd", token_id="2", suggestion="ADP", agent="lemma")
    store.flag_issue("pred", "p2", "head", "check tree", agent="scorer")
    assert show(store, "flag_issue", flag) == "Flag 1 is open."
    assert show(store, "list_flags", store.list_flags("pred")).splitlines() == [
        "- flag 1 [open] p1 word 2, upos, from lemma: odd (suggests 'ADP')",
        "- flag 2 [open] p2 sentence, head, from scorer: check tree",
    ]
    view = show(store, "get_sentence", store.get_sentence("pred", "p1"))
    assert view.startswith("# sent_id = p1") and "Mean confidence 0.06." in view
    assert "Open flags:\n- flag 1" in view and "Issues:\n- warning: Word 2 (al) has no UPOS." in view
    assert "LEMMA" not in view  # warnings for other agents' columns are dropped
    assert "Issues" not in show(store, "get_sentence", store.get_sentence("train", "t1"))


def test_edit_results(store):
    result = show(store, "update_cells", store.update_cells("pred", "p2", [{"token_id": "1", "column": "head", "value": "1"}]), role="head")
    assert result.splitlines() == [
        "OK: 1 change(s) saved.",
        "Issues:",
        "- error: Word 1 is its own head.",
        "- error: The sentence has 0 roots (none); expected exactly 1.",
    ] or result.startswith("OK: 1 change(s) saved.\nIssues:\n- error:")
    split = show(store, "split_word", store.split_word("pred", "p1", "2", ["a", "el"], multiword=True), role="segmentation")
    assert split.splitlines()[:3] == ["OK: 1 change(s) saved.", "Renumbered: 2->2, 3->4, 4->5".replace("2->2, ", ""), "Sentence now:"]
    merged = show(store, "remove_multiword", store.remove_multiword("pred", "p1", "2-3"), role="segmentation")
    assert "Renumbered" not in merged and "Sentence now:" in merged


def test_passthrough(store):
    assert compact_result("get_token", {"x": 1}, ROLES["upos"]) == {"x": 1}
    assert compact_result("analyse_form", {"unexpected": True}, ROLES["upos"]) == {"unexpected": True}


# ---------------------------------------------------------------- evidence


@pytest.fixture
def builder(store):
    return EvidenceBuilder(store, "pred")


def cells(store, column, sent_id="p1"):
    return store.low_confidence("pred", threshold=1.0, columns=[column], sent_id=sent_id).cells


def test_morphology_evidence(store, builder):
    text = builder.build(ROLES["upos"], [("p1", cells(store, "upos")), ("p2", [])])
    assert text.splitlines() == [
        "Analyses in the trusted data (lemma upos xpos feats ×count; then after the previous word, word-ending analogies):",
        "- Vino: venir VERB VMIS3S0 Mood=Ind|Number=Sing|Person=3|Tense=Past ×1",
        "- al: no trusted example",
        "- mercado: mercado NOUN NCMS000 Gender=Masc|Number=Sing ×1",
        "- .: . PUNCT . _ ×1; . PUNCT Fp _ ×1 | after 'mercado': . PUNCT Fp _ ×1",
        "- the: the DET DT Definite=Def|PronType=Art ×1",
        "- dog: no trusted example",
    ]
    store.set_cell("pred", "p2", "2", "form", "DOGS")
    assert "(ignoring case/diacritics)" in builder.build(ROLES["lemma"], [("p2", [])])
    xpos = builder.build(ROLES["xpos"], [("p1", [])])
    assert "- XPOS used with NOUN: NCMS000×2, NNS×1" in xpos and "- XPOS used with PUNCT: .×1, Fp×1" in xpos
    feats = builder.build(ROLES["feats"], [("p2", [])])
    assert "- Features used with DET: Definite=Def; Gender=Masc; Number=Sing; PronType=Art" in feats


def test_repeated_forms_are_shown_once(store, builder):
    store.set_cell("pred", "p2", "2", "form", "the")
    text = builder.build(ROLES["upos"], [("p2", [])])
    assert text.count("- the:") == 1


def test_context_and_analogy(tmp_path):
    path = tmp_path / "t.conllu"
    rows = [("si", "dixerint", "Fut"), ("si", "fecerint", "Fut"), ("ut", "dixerint", "Past")] * 2
    path.write_text(
        "".join(
            f"# sent_id = {i}\n1\t{a}\t{a}\tSCONJ\t_\t_\t2\tmark\t_\t_\n2\t{b}\t{b[:2]}co\tVERB\t_\tTense={t}\t0\troot\t_\t_\n\n"
            for i, (a, b, t) in enumerate(rows)
        ),
        encoding="utf-8",
    )
    store = CorpusStore()
    store.add_source("t", path)
    builder = EvidenceBuilder(store, "t")
    assert builder.context("dixerint", "si") == "after 'si': dico VERB _ Tense=Fut ×2"
    assert builder.context("fecerint", "si") == ""  # unambiguous overall: no context needed
    assert builder.context("dixerint", "nisi") == ""
    assert builder.analogy("maledixerint") == "words ending '-erint': VERB Tense=Fut ×4; VERB Tense=Past ×2; e.g. dixerint→dico, fecerint→feco"
    assert builder.analogy("dixerint") == ""  # well attested
    assert builder.analogy("zz") == "" and builder.analogy("qqqqqq") == ""


def test_many_analyses_are_truncated(tmp_path):
    path = tmp_path / "t.conllu"
    path.write_text("".join(f"# sent_id = {i}\n1\tx\tx{i}\tX\t_\t_\t0\troot\t_\t_\n\n" for i in range(5)), encoding="utf-8")
    store = CorpusStore()
    store.add_source("t", path)
    assert EvidenceBuilder(store, "t").analyses("x").endswith("; +2 rarer")


def test_syntax_evidence(store, builder):
    text = builder.build(ROLES["head"], [("p1", []), ("p2", [])])
    examples, attach = text.split("How similar words attach")
    assert examples.startswith("Similar sentences from the treebank")
    assert "## train:t2 (like p1)" in examples and "1\tVino\tvenir\tVERB\t0\troot" in examples
    lines = attach.splitlines()
    assert lines[1] == "- Vino (VERB): root→ROOT root ×3 (by UPOS, 3 arcs)"
    assert lines[2] == "- al (_): no UPOS or lemma yet"
    assert len(lines) == 7
    deprel = builder.build(ROLES["deprel"], [("p1", cells(store, "deprel"))])
    assert "- p1 1 Vino (VERB): arc to ROOT: root×3 (by UPOS+head UPOS)" in deprel
    assert "- p1 2 al (_): no head yet" in deprel
    assert "- p1 3 mercado (NOUN): arc to Vino (VERB, left): no similar trusted arcs" in deprel
    only_vino = builder.build(ROLES["deprel"], [("p1", [c for c in cells(store, "deprel") if c.token_id == "1"])])
    assert only_vino.count("\n- ") == 1
    store.set_cell("pred", "p1", "2", "upos", "INTJ")
    assert "no trusted arcs for this UPOS" in builder.build(ROLES["head"], [("p1", [])])
    store.set_cell("pred", "p2", "2", "lemma", "dog")
    assert "(by UPOS" in builder.build(ROLES["head"], [("p2", [])])  # "dog" has only 1 trusted arc


def test_arcs_by_lemma_when_well_attested(tmp_path):
    path = tmp_path / "t.conllu"
    block = "1\tthe\tthe\tDET\t_\t_\t2\tdet\t_\t_\n2\tdog\tdog\tNOUN\t_\t_\t0\troot\t_\t_\n\n"
    path.write_text("".join(f"# sent_id = {i}\n{block}" for i in range(3)), encoding="utf-8")
    store = CorpusStore()
    store.add_source("t", path)
    word = store.source("t").sentence("0").token("1")
    assert EvidenceBuilder(store, "t").arc_patterns(word) == "det→NOUN right ×3 (by lemma, 3 arcs)"


def test_scorer_evidence(store, builder):
    text = builder.build(ROLES["scorer"], [("p1", []), ("p2", [])])
    assert "- Vino: venir VERB VMIS3S0 Mood=Ind|Number=Sing|Person=3|Tense=Past ×1 | arc to ROOT: root×3 (by UPOS+head UPOS)" in text
    assert "- al: no trusted example\n" in text  # no UPOS yet: no arc
    assert "- the: the DET DT Definite=Def|PronType=Art ×1 | arc to dog (NOUN, right): det×3 (by UPOS+head UPOS)" in text


def test_scorer_skips_arcs_with_unknown_heads(tmp_path, store):
    path = tmp_path / "bad.conllu"
    path.write_text("# sent_id = b\n1\tzz\tzz\tX\t_\t_\t7\tdep\t_\t_\n\n", encoding="utf-8")
    store.add_source("bad", path, mode="rw")
    assert EvidenceBuilder(store, "bad").build(ROLES["scorer"], [("b", [])]).endswith("- zz: no trusted example")


def test_segmentation_evidence(store, builder):
    assert builder.build(ROLES["segmentation"], [("p1", []), ("p2", [])]).splitlines() == [
        "Multiword evidence (surface token: splits in the trusted data):",
        "- 'al': split as a+el (ADP DET) ×1; unsplit ×0",
    ]
    store.split_word("pred", "p1", "2", ["a", "l"], multiword=True)
    store.set_cell("pred", "p2", "1", "form", "Vino")
    assert "'al'" in builder.build(ROLES["segmentation"], [("p1", []), ("p2", [])])
    assert builder.build(ROLES["segmentation"], [("p2", [])]).endswith("is ever split in the trusted data.")


def test_fixed_expressions_and_examples(tmp_path, monkeypatch):
    rows = lambda words: "\n".join("\t".join(w) for w in words)  # noqa: E731
    fixed = [("1", "a", "a", "ADP", "_", "_", "4", "case", "_", "_"), ("2", "pesar", "pesar", "NOUN", "_", "_", "1", "fixed", "_", "_"),
             ("3", "de", "de", "ADP", "_", "_", "1", "fixed", "_", "_"), ("4", "todo", "todo", "PRON", "_", "_", "0", "root", "_", "_")]
    loose_use = [("1", "a", "a", "ADP", "_", "_", "2", "case", "_", "_"), ("2", "pesar", "pesar", "NOUN", "_", "_", "0", "root", "_", "_"),
                 ("3", "de", "de", "ADP", "_", "_", "2", "case", "_", "_")]
    gap = [("1", "ni", "ni", "CCONJ", "_", "_", "0", "root", "_", "_"), ("2", "x", "x", "X", "_", "_", "1", "dep", "_", "_"),
           ("3", "siquiera", "siquiera", "ADV", "_", "_", "1", "fixed", "_", "_")]
    train = tmp_path / "train.conllu"
    blocks = [rows(fixed)] * 3 + [rows(loose_use), rows(gap), rows(gap), rows(gap)]
    train.write_text("".join(f"# sent_id = t{i}\n{block}\n\n" for i, block in enumerate(blocks)), encoding="utf-8")
    pred = tmp_path / "pred.conllu"
    pred.write_text("# sent_id = p\n1\ta\t_\t_\t_\t_\t_\t_\t_\t_\n2\tpesar\t_\t_\t_\t_\t_\t_\t_\t_\n"
                    "3\tde\t_\t_\t_\t_\t_\t_\t_\t_\n4\ttodo\t_\t_\t_\t_\t_\t_\t_\t_\n\n", encoding="utf-8")
    store = CorpusStore()
    store.add_source("train", train)
    store.add_source("pred", pred, mode="rw")
    found = store.fixed_expressions("pred", "p")
    assert [(f.start, f.end, f.forms, f.fixed, f.occurrences) for f in found] == [("1", "3", ["a", "pesar", "de"], 3, 4)]
    assert store.fixed_expressions("pred", "p", min_share=0.9) == []  # 3 of 4 is below 90%
    assert store.source("train").index.fixed[("ni", "siquiera")] == 0  # not contiguous: not indexed
    builder = EvidenceBuilder(store, "pred")
    text = builder.build(ROLES["deprel"], [("p", [])])
    assert "- p words 1–3 'a pesar de': fixed expression in 3 of 4 occurrences" in text
    assert text.count("## train:") == 1  # each similar sentence once
    monkeypatch.setattr("llmparser.agents.evidence.EXAMPLE_MAX_WORDS", 2)
    assert "Similar sentences" not in EvidenceBuilder(store, "pred").build(ROLES["deprel"], [("p", [])])
