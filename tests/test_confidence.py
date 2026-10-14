import pytest

from llmparser.corpus.confidence import (
    check_confidence,
    read_confidence,
    set_confidence,
    strip_confidence,
    write_confidence,
)
from llmparser.corpus.conllu import Token
from llmparser.corpus.errors import EditError


def test_read_ignores_malformed_entries():
    token = Token(id="1", misc="Conf=upos:0.5,lemma:x,bogus:0.3,head:2,deprel:0.25")
    assert read_confidence(token) == {"upos": 0.5, "deprel": 0.25}
    assert read_confidence(Token(id="1")) == {}


def test_write_orders_columns_and_removes_when_empty():
    token = Token(id="1", misc="SpaceAfter=No")
    write_confidence(token, {"head": 0.4, "form": 1.0})
    assert token.misc == "SpaceAfter=No|Conf=form:1.00,head:0.40"
    write_confidence(token, {})
    assert token.misc == "SpaceAfter=No"


def test_set_confidence():
    token = Token(id="1")
    set_confidence(token, "upos", 0.75)
    set_confidence(token, "lemma", 0.5)
    set_confidence(token, "upos", None)
    assert read_confidence(token) == {"lemma": 0.5}
    with pytest.raises(EditError, match="has no confidence"):
        set_confidence(token, "misc", 0.5)
    with pytest.raises(EditError, match="between 0 and 1"):
        set_confidence(token, "upos", 1.5)
    assert check_confidence(0.0) == 0.0


def test_strip_confidence_keeps_original_scores():
    assert strip_confidence("Gloss=x|Conf=upos:0.10", "Conf=upos:0.90") == "Gloss=x|Conf=upos:0.90"
    assert strip_confidence("Conf=upos:0.10", "_") == "_"
