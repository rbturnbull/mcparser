import pytest

from llmparser.corpus import CorpusError, format_conllu
from llmparser.textinput import InputSentence, build_sentences, read_input, tokenise


def test_tokenise_detaches_punctuation():
    assert tokenise("Vino (del) mercado...  ¿Sí?") == [
        ("Vino", True), ("(", False), ("del", False), (")", True), ("mercado", False), ("...", True),
        ("¿", False), ("Sí", False), ("?", True),
    ]  # fmt: skip
    assert tokenise("don't U.S.") == [("don't", True), ("U.S", False), (".", True)]
    assert tokenise("الله: ※") == [("الله", False), (":", True), ("※", True)]
    assert tokenise("ἦν ἕτοιμος.") == [("ἦν", True), ("ἕτοιμος", False), (".", True)]


def test_read_plain_text(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text("first line\n\n  second line  \n", encoding="utf-8")
    assert read_input(path) == [InputSentence(sent_id="1", text="first line"), InputSentence(sent_id="2", text="second line")]


def test_read_yaml_mapping_like_ephesians(tmp_path):
    path = tmp_path / "eph.yaml"
    path.write_text(
        "B10K1V1:\n  text: 'بولس رسول'\n  verse: Ephesians 1:1\n  chapter: 1\n  conllu: |-\n    1\tx\n  notes: [a, b]\n"
        "B10K1V2: just text\n",
        encoding="utf-8",
    )
    entries = read_input(path)
    assert entries[0] == InputSentence(sent_id="B10K1V1", text="بولس رسول", meta={"verse": "Ephesians 1:1", "chapter": "1"})
    assert entries[1].text == "just text"


def test_read_yaml_list(tmp_path):
    path = tmp_path / "list.yml"
    path.write_text("- {sent_id: a, text: one}\n- {id: b, text: two}\n- {text: three}\n- four\n", encoding="utf-8")
    assert [(e.sent_id, e.text) for e in read_input(path)] == [("a", "one"), ("b", "two"), ("3", "three"), ("4", "four")]


@pytest.mark.parametrize(("content", "message"), [("x: {verse: 1}\n", "no 'text'"), ("just a string\n", "mapping or list")])
def test_read_yaml_errors(tmp_path, content, message):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CorpusError, match=message):
        read_input(path)


def test_build_sentences():
    (sentence,) = build_sentences([InputSentence(sent_id="v1", text=" Vino  del mercado. ", meta={"verse": "1:1"})])
    assert sentence.comments == ["sent_id = v1", "text = Vino del mercado.", "verse = 1:1"]
    assert [(w.id, w.form, w.misc) for w in sentence.words] == [
        ("1", "Vino", "_"), ("2", "del", "_"), ("3", "mercado", "SpaceAfter=No"), ("4", ".", "_"),
    ]  # fmt: skip
    assert sentence.text == "Vino del mercado."
    assert format_conllu([sentence]).count("\t_\t_\t_\t_\t_\t_\t_\t") == 4
    with pytest.raises(CorpusError, match="is empty"):
        build_sentences([InputSentence(sent_id="e", text="   ")])
