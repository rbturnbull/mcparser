import os

import pytest

from mcparser.corpus import ConflictError, CorpusError, CorpusSource, NotFoundError, ReadOnlyError
from mcparser.corpus.models import Edit, Flag


def test_read_only_defaults(train_path):
    source = CorpusSource("train", train_path)
    assert not source.writable
    assert source.default_confidence == 1.0
    info = source.info()
    assert (info.sentences, info.words, info.open_flags, info.mode) == (3, 14, 0, "ro")
    with pytest.raises(ReadOnlyError):
        source.save()


def test_writable_defaults_and_confidence(pred_path):
    source = CorpusSource("pred", pred_path, mode="rw")
    assert source.default_confidence == 0.0
    word = source.sentence("p1").words[0]
    assert source.confidence(word, "upos") == 0.8
    assert source.confidence(word, "head") == 0.0
    assert source.sentence_confidence(source.sentence("p1")) == pytest.approx(1.7 / 28)
    assert source.index is source.index
    source.changed()
    assert source._index is None


def test_sentence_confidence_of_empty_sentence(tmp_path):
    path = tmp_path / "x.conllu"
    path.write_text("# sent_id = a\n1-2\tab\t_\t_\t_\t_\t_\t_\t_\t_\n\n", encoding="utf-8")
    source = CorpusSource("x", path, default_confidence=0.5)
    assert source.sentence_confidence(source.sentence("a")) == 0.5


def test_missing_sent_ids_are_numbered(tmp_path):
    path = tmp_path / "raw.conllu"
    path.write_text("# text = a\n1\ta\t_\t_\t_\t_\t0\troot\t_\t_\n\n1\tb\t_\t_\t_\t_\t0\troot\t_\t_\n\n", encoding="utf-8")
    source = CorpusSource("raw", path)
    assert [s.sent_id for s in source.sentences] == ["1", "2"]
    assert source.sentences[0].comments == ["sent_id = 1", "text = a"]


def test_load_errors(tmp_path):
    with pytest.raises(NotFoundError, match="not found"):
        CorpusSource("x", tmp_path / "missing.conllu")
    path = tmp_path / "dup.conllu"
    path.write_text("# sent_id = a\n1\ta\t_\t_\t_\t_\t0\troot\t_\t_\n\n# sent_id = a\n1\tb\t_\t_\t_\t_\t0\troot\t_\t_\n\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="Duplicate"):
        CorpusSource("x", path)


def test_unknown_sentence(pred_path):
    source = CorpusSource("pred", pred_path, mode="rw")
    with pytest.raises(NotFoundError, match="No sentence 'z' in source 'pred'"):
        source.sentence("z")


def test_save_writes_sidecars_and_reload_restores_them(pred_path):
    source = CorpusSource("pred", pred_path, mode="rw")
    source.sentence("p2").words[0].lemma = "THE"
    source.flags.append(Flag(id=source.next_flag_id(), sent_id="p2", column="lemma", reason="r", raised_by="a", created="t"))
    source.record(Edit(seq=0, time="t", sent_id="p2", action="set", agent="a"))
    source.save()
    source.save()  # nothing new to append to the history
    assert "THE" in pred_path.read_text(encoding="utf-8")
    assert len(source.history_path.read_text(encoding="utf-8").splitlines()) == 1

    reloaded = CorpusSource("pred", pred_path, mode="rw")
    assert reloaded.flags[0].reason == "r"
    assert reloaded.history[0].seq == 1
    assert reloaded.next_flag_id() == 2
    assert reloaded.info().open_flags == 1


def test_history_file_with_blank_lines(pred_path):
    source = CorpusSource("pred", pred_path, mode="rw")
    source.history_path.write_text("\n", encoding="utf-8")
    source.load()
    assert source.history == []


def test_save_detects_external_changes(pred_path):
    source = CorpusSource("pred", pred_path, mode="rw")
    stat = pred_path.stat()
    os.utime(pred_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    with pytest.raises(ConflictError, match="modified by another process"):
        source.save()
    source.load()
    source.save()
