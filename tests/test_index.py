from conftest import DATA
from mcparser.corpus.conllu import parse_conllu
from mcparser.corpus.index import SourceIndex, loose, merge_counters, terms


def index():
    return SourceIndex(parse_conllu((DATA / "train.conllu").read_text(encoding="utf-8")))


def test_loose_and_terms():
    assert loose("Ῥώμη") == "ρωμη"
    assert loose("Río") == "rio"
    assert terms("Vino, del río!") == ["vino", "del", "rio"]


def test_lookups():
    idx = index()
    assert idx.lookup("form", "dogs") == [(0, 1)]
    assert idx.lookup("form", "rio") == []
    assert idx.lookup("form", "RIO", loose_match=True) == [(2, 5)]
    assert idx.lookup("upos", "ADP", loose_match=True) == [(1, 2), (2, 3)]
    assert idx.lookup("root", "dg") == [(0, 1)]
    assert idx.multiword("al") == [(2, 2)]
    assert idx.multiword("AL", loose_match=True) == [(2, 2)]


def test_inventories():
    idx = index()
    assert idx.values["upos"]["NOUN"] == 3
    assert idx.by_upos["xpos"]["NOUN"]["NCMS000"] == 2
    assert idx.features["DET"]["Definite"]["Def"] == 3
    assert "PronType=Art" in idx.feature_pairs()


def test_search_ranks_by_bm25():
    idx = index()
    results = idx.search("el mercado")
    assert [position for position, _ in sorted(results, key=lambda r: -r[1])][0] == 1
    assert idx.search("nothing matches") == []
    assert SourceIndex([]).search("x") == []


def test_merge_counters():
    idx = index()
    assert merge_counters([idx.values["upos"], idx.values["upos"]])["NOUN"] == 6


def test_empty_nodes_and_valueless_features_are_skipped():
    text = "1\ta\ta\tX\t_\tForeign|Case=Nom\t0\troot\t_\t_\n1.1\te\te\tX\t_\t_\t_\t_\t_\t_\n\n"
    idx = SourceIndex(parse_conllu(text))
    assert idx.lookup("form", "e") == []
    assert dict(idx.features["X"]) == {"Case": {"Nom": 1}}


def test_chained_relations_are_counted():
    text = "\n".join(
        ["1\ta\ta\tNOUN\t_\t_\t0\troot\t_\t_", "2\tb\tb\tNOUN\t_\t_\t1\tconj\t_\t_", "3\tc\tc\tNOUN\t_\t_\t2\tconj\t_\t_", ""]
    )
    idx = SourceIndex(parse_conllu(text))
    assert idx.chained == {"conj": 1}
