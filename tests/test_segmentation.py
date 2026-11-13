import pytest

from conftest import sentence
from mcparser.corpus.errors import EditError, NotFoundError
from mcparser.corpus.segmentation import (
    add_multiword,
    delete_word,
    insert_word,
    merge_words,
    remove_multiword,
    split_word,
)


def rows(s):
    """(id, form, head, deprel, deps, misc) for every token line."""
    return [(t.id, t.form, t.head, t.deprel, t.deps, t.misc) for t in s.tokens]


def test_split_word_retokenises():
    s = sentence(
        """
        1 Vino venir VERB _ _ 0 root 0:root _
        2 al a ADP _ _ 3 case 3:case SpaceAfter=No|Gloss=to|Conf=form:0.20
        3 mercado mercado NOUN _ _ 1 obl 1:obl _
        """
    )
    change = split_word(s, "2", ["a", "l"], head_index=1)
    assert rows(s) == [
        ("1", "Vino", "0", "root", "0:root", "_"),
        ("2", "a", "3", "_", "_", "SpaceAfter=No"),
        ("3", "l", "4", "case", "4:case", "Gloss=to|SpaceAfter=No"),
        ("4", "mercado", "1", "obl", "1:obl", "_"),
    ]
    assert s.token("3").upos == "ADP" and s.token("2").upos == "_"
    assert change.id_map == {"1": "1", "2": "3", "3": "4"}
    assert change.new_ids == ["2", "3"]


def test_split_word_as_multiword_moves_dependents_to_head_piece():
    s = sentence(
        """
        1 Vino venir VERB _ _ 0 root 0:root _
        2 al _ ADP _ _ 3 case 3:case SpaceAfter=No
        3 mercado mercado NOUN _ _ 1 obl 1:obl _
        4 x x X _ _ 2 dep 2:dep _
        """
    )
    split_word(s, "2", ["a", "el"], multiword=True)
    assert rows(s) == [
        ("1", "Vino", "0", "root", "0:root", "_"),
        ("2-3", "al", "_", "_", "_", "SpaceAfter=No"),
        ("2", "a", "4", "case", "4:case", "_"),
        ("3", "el", "2", "_", "_", "_"),
        ("4", "mercado", "1", "obl", "1:obl", "_"),
        ("5", "x", "2", "dep", "2:dep", "_"),
    ]


def test_split_inside_existing_range_extends_it():
    s = sentence(
        """
        1-2 xy _ _ _ _ _ _ _ _
        1 x _ _ _ _ 0 root _ _
        2 y _ _ _ _ 1 dep _ _
        """
    )
    split_word(s, "1", ["p", "q"])
    assert [t.id for t in s.tokens] == ["1-3", "1", "2", "3"]
    s2 = sentence(
        """
        1-2 xy _ _ _ _ _ _ _ _
        1 x _ _ _ _ 0 root _ _
        2 y _ _ _ _ 1 dep _ _
        """
    )
    split_word(s2, "2", ["p", "q"])
    assert s2.tokens[0].id == "1-3"


@pytest.mark.parametrize(
    ("token_id", "forms", "kwargs", "message"),
    [
        ("1", ["a"], {}, "at least two"),
        ("1", ["a", "b"], {"head_index": 2}, "head_index"),
        ("1", ["a", ""], {}, "must not be empty"),
        ("1-2", ["a", "b"], {}, "not a word"),
        ("1", ["a", "b"], {"multiword": True}, "already inside"),
    ],
)
def test_split_word_errors(token_id, forms, kwargs, message):
    s = sentence(
        """
        1-2 xy _ _ _ _ _ _ _ _
        1 x _ _ _ _ 0 root _ _
        2 y _ _ _ _ 1 dep _ _
        """
    )
    with pytest.raises(EditError, match=message):
        split_word(s, token_id, forms, **kwargs)


def test_empty_nodes_and_enhanced_deps_are_renumbered():
    s = sentence(
        """
        1 a _ _ _ _ 2 x 2:x _
        2 b _ _ _ _ 0 root 0:root|2.1:y _
        2.1 e _ _ _ _ _ _ 1:z _
        3 c _ _ _ _ 2 x 2:x|weird _
        """
    )
    change = insert_word(s, "0", "z")
    assert rows(s) == [
        ("1", "z", "_", "_", "_", "_"),
        ("2", "a", "3", "x", "3:x", "_"),
        ("3", "b", "0", "root", "0:root|3.1:y", "_"),
        ("3.1", "e", "_", "_", "2:z", "_"),
        ("4", "c", "3", "x", "2:x|weird", "_"),
    ]
    assert change.id_map["2.1"] == "3.1"
    assert change.new_ids == ["1"]


def test_insert_after_word():
    s = sentence(
        """
        1 a _ _ _ _ 0 root _ _
        2 b _ _ _ _ 1 dep _ _
        """
    )
    insert_word(s, "1", "m")
    assert s.word_forms() == ["a", "m", "b"]
    assert s.token("3").head == "1"
    with pytest.raises(NotFoundError):
        insert_word(s, "9", "m")


def test_merge_takes_form_from_exact_range():
    s = sentence(
        """
        1 Vino venir VERB _ _ 0 root 0:root _
        2-3 del _ _ _ _ _ _ _ _
        2 de de ADP _ _ 4 case 4:case Conf=upos:0.9
        3 el el DET _ _ 4 det 4:det SpaceAfter=No
        4 mercado mercado NOUN _ _ 1 obl 1:obl _
        """
    )
    change = merge_words(s, ["2", "3"])
    assert rows(s) == [
        ("1", "Vino", "0", "root", "0:root", "_"),
        ("2", "del", "3", "case", "3:case", "SpaceAfter=No"),
        ("3", "mercado", "1", "obl", "1:obl", "_"),
    ]
    assert s.token("2").upos == "ADP"
    assert change.id_map == {"1": "1", "2": "2", "3": "2", "4": "3"}
    assert change.new_ids == ["2"]


def test_merge_uses_group_head_and_keeps_partial_range():
    s = sentence(
        """
        1 Vino venir VERB _ _ 0 root 0:root _
        2-3 del _ _ _ _ _ _ _ _
        2 de de ADP _ _ 4 case 4:case _
        3 el el DET _ _ 4 det 4:det _
        4 mercado mercado NOUN _ _ 1 obl 1:obl _
        """
    )
    merge_words(s, ["4", "3"])
    assert rows(s) == [
        ("1", "Vino", "0", "root", "0:root", "_"),
        ("2-3", "del", "_", "_", "_", "_"),
        ("2", "de", "3", "case", "3:case", "_"),
        ("3", "elmercado", "1", "obl", "1:obl", "_"),
    ]


def test_merge_cycle_falls_back_to_first_word():
    s = sentence(
        """
        1 a _ X _ _ 2 dep _ _
        2 b _ Y _ _ 1 dep _ _
        """
    )
    merge_words(s, ["1", "2"], form="ab")
    assert rows(s) == [("1", "ab", "_", "dep", "_", "_")]
    assert s.token("1").upos == "X"


def test_merge_errors():
    s = sentence(
        """
        1 a _ _ _ _ 0 root _ _
        2 b _ _ _ _ 1 dep _ _
        3 c _ _ _ _ 1 dep _ _
        """
    )
    with pytest.raises(EditError, match="at least two"):
        merge_words(s, ["1"])
    with pytest.raises(EditError, match="adjacent"):
        merge_words(s, ["1", "3"])


def test_delete_reattaches_dependents_and_shrinks_ranges():
    s = sentence(
        """
        1-3 abc _ _ _ _ _ _ _ SpaceAfter=No
        1 a _ _ _ _ 2 dep _ _
        1.1 e _ _ _ _ _ _ _ _
        2 b _ _ _ _ 0 root _ _
        3 c _ _ _ _ 1 dep _ _
        4 d _ _ _ _ 2 dep _ _
        """
    )
    change = delete_word(s, "1")
    assert rows(s) == [
        ("0.1", "e", "_", "_", "_", "_"),
        ("1-2", "abc", "_", "_", "_", "SpaceAfter=No"),
        ("1", "b", "0", "root", "_", "_"),
        ("2", "c", "1", "dep", "_", "_"),
        ("3", "d", "1", "dep", "_", "_"),
    ]
    assert change.id_map["1"] is None
    assert change.new_ids == []


def test_delete_last_word_of_range_drops_single_word_range():
    s = sentence(
        """
        1 x _ _ _ _ 0 root _ _
        2-3 ab _ _ _ _ _ _ _ _
        2 a _ _ _ _ 1 dep _ _
        3 b _ _ _ _ 3 dep _ _
        """
    )
    delete_word(s, "3")
    assert rows(s) == [("1", "x", "0", "root", "_", "_"), ("2", "a", "1", "dep", "_", "_")]


def test_delete_errors():
    s = sentence("1 a _ _ _ _ 0 root _ _")
    with pytest.raises(EditError, match="only word"):
        delete_word(s, "1")


def test_add_and_remove_multiword():
    s = sentence(
        """
        1 a _ _ _ _ 0 root _ _
        2 b _ _ _ _ 1 dep _ SpaceAfter=No
        3 . _ _ _ _ 1 punct _ _
        """
    )
    add_multiword(s, 1, 2, "ab")
    assert rows(s)[0] == ("1-2", "ab", "_", "_", "_", "SpaceAfter=No")
    assert s.token("2").misc == "_"
    with pytest.raises(EditError, match="already inside"):
        add_multiword(s, 2, 3, "b.")
    with pytest.raises(EditError, match="at least two"):
        add_multiword(s, 2, 2, "b")
    remove_multiword(s, "1-2")
    assert [t.id for t in s.tokens] == ["1", "2", "3"]
    assert s.token("2").misc == "SpaceAfter=No"
    with pytest.raises(EditError, match="not a multiword"):
        remove_multiword(s, "1")


def test_remove_multiword_without_space_after():
    s = sentence(
        """
        1-2 ab _ _ _ _ _ _ _ _
        1 a _ _ _ _ 0 root _ _
        2 b _ _ _ _ 1 dep _ _
        """
    )
    remove_multiword(s, "1-2")
    assert s.token("2").misc == "_"


def test_malformed_range_is_dropped():
    s = sentence(
        """
        1 a _ _ _ _ 0 root _ _
        5-6 zz _ _ _ _ _ _ _ _
        """
    )
    remove_multiword(s, "5-6")
    assert [t.id for t in s.tokens] == ["1"]


def test_edits_leave_unrelated_ranges_alone():
    text = """
        1-2 ab _ _ _ _ _ _ _ _
        1 a _ _ _ _ 0 root _ _
        2 b _ _ _ _ 1 dep _ _
        3 c _ _ _ _ 1 dep _ _
        4 d _ _ _ _ 1 dep _ _
        """
    s = sentence(text)
    split_word(s, "4", ["d", "e"], multiword=True)
    assert [t.id for t in s.tokens] == ["1-2", "1", "2", "3", "4-5", "4", "5"]
    s = sentence(text)
    merge_words(s, ["3", "4"])
    assert [t.id for t in s.tokens] == ["1-2", "1", "2", "3"]
    s = sentence(text)
    add_multiword(s, 3, 4, "cd")
    assert [(t.id, t.misc) for t in s.tokens if t.is_multiword] == [("1-2", "_"), ("3-4", "_")]
