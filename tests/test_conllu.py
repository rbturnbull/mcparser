import pytest

from conftest import DATA, sentence
from llmparser.corpus.conllu import (
    Sentence,
    Token,
    check_cell,
    format_conllu,
    format_feats,
    format_pairs,
    parse_conllu,
    parse_pairs,
    surface_text,
)
from llmparser.corpus.errors import ConlluFormatError, EditError, NotFoundError


def test_round_trip_is_lossless():
    text = (DATA / "train.conllu").read_text(encoding="utf-8")
    assert format_conllu(parse_conllu(text)) == text


def test_parse_tolerates_repeated_blank_lines():
    text = "1\ta\t_\t_\t_\t_\t0\troot\t_\t_\n\n\n\n1\tb\t_\t_\t_\t_\t0\troot\t_\t_\n"
    assert [s.words[0].form for s in parse_conllu(text)] == ["a", "b"]


def test_parse_handles_crlf_and_missing_final_blank_line():
    sentences = parse_conllu("# sent_id = a\r\n1\tx\t_\t_\t_\t_\t0\troot\t_\t_")
    assert sentences[0].sent_id == "a"
    assert sentences[0].words[0].form == "x"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1\tx\t_", "expected 10"),
        ("1a\tx\t_\t_\t_\t_\t0\troot\t_\t_", "invalid token id"),
        ("1\tx\t_\t_\t_\t_\t0\troot\t_\t_\n# late", "comment after token"),
        ("# only a comment\n\n1\tx\t_\t_\t_\t_\t0\troot\t_\t_", "line 2: sentence has comments"),
        ("1\tx\t_\t_\t_\t_\t0\troot\t_\t_\n\n# dangling", "end of file"),
    ],
)
def test_parse_errors(text, message):
    with pytest.raises(ConlluFormatError, match=message):
        parse_conllu(text)


def test_token_line_without_line_number():
    with pytest.raises(ConlluFormatError, match="^expected 10"):
        Token.from_line("1\tx")


def test_token_kinds_and_span():
    word, multiword, empty = Token(id="3"), Token(id="3-4"), Token(id="3.1")
    assert (word.is_word, word.is_multiword, word.is_empty_node) == (True, False, False)
    assert (multiword.is_word, multiword.is_multiword) == (False, True)
    assert empty.is_empty_node and not empty.is_word
    assert multiword.span() == (3, 4)


def test_token_get_set_and_misc():
    token = Token(id="1", form="x", misc="SpaceAfter=No|Gloss=y")
    assert token.get("form") == "x"
    with pytest.raises(EditError, match="Unknown column"):
        token.get("nope")
    token.set("lemma", "  ")
    assert token.lemma == "_"
    assert token.misc_value("Gloss") == "y"
    token.set_misc_value("Gloss", None)
    token.set_misc_value("Root", "k t b")
    assert token.misc == "SpaceAfter=No|Root=k t b"
    assert token.feats_dict == {}
    token.feats = "Number=Sing|Foreign"
    assert token.feats_dict == {"Number": "Sing", "Foreign": None}


def test_check_cell():
    assert check_cell("form", "New York") == "New York"
    assert check_cell("upos", "NOUN") == "NOUN"
    with pytest.raises(EditError, match="Unknown column"):
        check_cell("bogus", "x")
    with pytest.raises(EditError, match="tabs"):
        check_cell("lemma", "a\tb")
    with pytest.raises(EditError, match="spaces"):
        check_cell("upos", "NO UN")


def test_pairs():
    assert parse_pairs("_") == {}
    assert parse_pairs("A=1|B") == {"A": "1", "B": None}
    assert format_pairs({}) == "_"
    assert format_pairs({"A": "1", "B": None}) == "A=1|B"
    assert format_feats({"number": "Sing", "Case": "Nom"}) == "Case=Nom|number=Sing"


def test_sentence_meta_and_lookup():
    s = sentence(
        """
        # sent_id = s1
        # newdoc
        1 a a X _ _ 0 root _ _
        """
    )
    assert s.meta("newdoc") is None
    assert s.meta("missing") is None
    s.set_meta("sent_id", "s2")
    s.set_meta("text", "a")
    assert s.comments == ["sent_id = s2", "newdoc", "text = a"]
    assert s.token(1).form == "a"
    with pytest.raises(NotFoundError, match="No token"):
        s.token("9")
    assert s.word_forms() == ["a"]
    assert s.to_conllu().splitlines()[0] == "# sent_id = s2"


def test_text_falls_back_to_surface_forms():
    s = sentence(
        """
        1 Vino _ _ _ _ _ _ _ _
        2-3 del _ _ _ _ _ _ _ _
        2 de _ _ _ _ _ _ _ _
        3 el _ _ _ _ _ _ _ _
        4 mercado _ _ _ _ _ _ _ SpaceAfter=No
        4.1 fue _ _ _ _ _ _ _ _
        5 . _ _ _ _ _ _ _ _
        """
    )
    assert s.text == "Vino del mercado."
    assert surface_text([]) == ""
    assert Sentence().text == ""
