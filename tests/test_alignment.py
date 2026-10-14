import json
import os
import shutil

import pytest
from mcp import Client

from conftest import DATA
from llmparser.corpus import ConflictError, CorpusError, CorpusStore, EditError, NotFoundError, ReadOnlyError
from llmparser.corpus.alignment import AlignmentSet, Link, format_ugarit, parse_ugarit
from llmparser.mcp import build_server
from llmparser.mcp.config import AlignmentSpec, claude_desktop_config, load_store


@pytest.fixture
def parallel(tmp_path):
    for name in ("parallel_a", "parallel_b"):
        shutil.copy(DATA / f"{name}.conllu", tmp_path / f"{name}.conllu")
    store = CorpusStore()
    store.add_source("la", tmp_path / "parallel_a.conllu")
    store.add_source("grc", tmp_path / "parallel_b.conllu")
    store.add_alignment("verses", "la", "grc", tmp_path / "out" / "verses.jsonl")
    return store


def test_ugarit():
    links = parse_ugarit("1-2S 3-5P 4-4", confidence=0.8)
    assert [(l.source, l.target, l.sure, l.confidence) for l in links] == [("1", "2", True, 0.8), ("3", "5", False, 0.8), ("4", "4", True, 0.8)]
    assert format_ugarit(links) == "1-2S 3-5P 4-4S"
    assert links[0].key == "1-2"
    with pytest.raises(EditError, match="Invalid link 'x'"):
        parse_ugarit("1-1S x")


def test_set_links_and_parallel_view(parallel, tmp_path):
    view = parallel.set_links("verses", "s1", "1-2S 2-3S", confidence=0.9, reason="obvious", agent="aligner")
    assert (view.links, view.unaligned_a, view.unaligned_b) == ("1-2S 2-3S", ["3"], ["1", "4", "5"])
    assert view.confidence == {"1-2": 0.9, "2-3": 0.9}
    assert view.conllu_b.startswith("# sent_id = s1")
    view = parallel.set_links("verses", "s1", [{"source": "3", "target": "5", "sure": False}], replace=False)
    assert view.links == "1-2S 2-3S 3-5P" and view.confidence["3-5"] == 0.0  # writable default
    view = parallel.set_links("verses", "s1", "3-4P 3-5S", replace=False)
    assert view.links == "1-2S 2-3S 3-4P 3-5S"  # 3-5 updated, 3-4 added
    assert parallel.set_links("verses", "s2", "1-1", dry_run=True).links == "1-1S"
    assert parallel.get_parallel("verses", "s2").links == ""
    lines = (tmp_path / "out" / "verses.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["links"][0] == {"source": "1", "target": "2", "sure": True, "confidence": 0.9, "agent": "aligner"}
    history = (tmp_path / "out" / "verses.jsonl.history.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(history[0])["agent"] == "aligner" and json.loads(history[0])["new"] == "1-2S 2-3S"
    view = parallel.remove_links("verses", "s1", "3-4 3-5S", reason="wrong")
    assert view.links == "1-2S 2-3S"
    assert [i.links for i in parallel.list_alignments()] == [2]


@pytest.mark.parametrize(
    ("links", "error", "message"),
    [
        ("9-1S", EditError, "no word 9 in la:s1"),
        ("1-9S", EditError, "no word 9 in grc:s1"),
        ("1-1S", NotFoundError, "No sentence 's3' in source 'grc'"),
    ],
)
def test_set_links_errors(parallel, links, error, message):
    sent_id = "s3" if "s3" in message else "s1"
    with pytest.raises(error, match=message):
        parallel.set_links("verses", sent_id, links)


def test_alignment_lookup_errors(parallel, tmp_path):
    with pytest.raises(NotFoundError, match="available: verses"):
        parallel.get_parallel("nope", "s1")
    with pytest.raises(NotFoundError, match="available: none"):
        CorpusStore().alignment("x")
    with pytest.raises(CorpusError, match="already exists"):
        parallel.add_alignment("la", "la", "grc", tmp_path / "x.jsonl")
    with pytest.raises(NotFoundError, match="Alignment file not found"):
        parallel.add_alignment("gold", "la", "grc", tmp_path / "missing.jsonl", mode="ro")


def test_read_only_gold_alignments(parallel, tmp_path):
    gold = tmp_path / "gold.jsonl"
    gold.write_text('{"sent_id": "s1", "links": [{"source": "1", "target": "2"}]}\n\n', encoding="utf-8")
    info = parallel.add_alignment("gold", "la", "grc", gold, mode="ro")
    assert (info.mode, info.default_confidence, info.sentences, info.links) == ("ro", 1.0, 1, 1)
    with pytest.raises(ReadOnlyError):
        parallel.set_links("gold", "s1", "2-3S")
    with pytest.raises(ReadOnlyError):
        parallel.remove_links("gold", "s1", "1-2")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"links": []}\n', encoding="utf-8")
    with pytest.raises(EditError, match="line 1"):
        parallel.add_alignment("bad", "la", "grc", bad, mode="ro")


def test_saving_conflicts_and_reloading(parallel, tmp_path):
    parallel.autosave = False
    parallel.set_links("verses", "s2", "1-1S", agent="a")
    parallel.set_links("verses", "s1", "2-3S", agent="a")
    assert parallel.save("verses") == ["verses"]
    path = tmp_path / "out" / "verses.jsonl"
    assert [json.loads(line)["sent_id"] for line in path.read_text(encoding="utf-8").splitlines()] == ["s1", "s2"]  # source order
    assert "verses" in parallel.save()
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    parallel.set_links("verses", "s2", "2-2S")
    with pytest.raises(ConflictError):
        parallel.save("verses")
    parallel.remove_links("verses", "s1", "2-3")  # not saved either
    aset = parallel.alignment("verses")
    aset.load()
    assert aset.links("s2")[0].target == "1" and len(aset.history) == 2
    assert AlignmentSet("copy", aset.source_a, aset.source_b, path, default_confidence=0.5).info().links == 2


def test_translation_candidates(parallel):
    parallel.set_links("verses", "s1", "1-2S 2-3S 3-5S", confidence=0.9)
    parallel.set_links("verses", "s2", "1-1S", confidence=0.4)
    result = parallel.translation_candidates("deus")
    assert (result.occurrences, result.unaligned) == (2, 0)
    assert [(c.value, c.count) for c in result.candidates] == [("θεός", 2)]
    assert result.examples == ["verses:s1 Deus→θεὸς", "verses:s2 Deus→θεὸς"]
    assert parallel.translation_candidates("deus", min_confidence=0.5).unaligned == 1
    assert parallel.translation_candidates("deus", exclude_sentence="verses:s2").occurrences == 1
    assert parallel.translation_candidates("deus", exclude_sentence="s2", alignments=["verses"]).occurrences == 1
    assert parallel.translation_candidates("videt").occurrences == 1  # lemma "_" falls back to the form
    assert [c.value for c in parallel.translation_candidates("ΘΕΟΣ", side="b", loose=True).candidates] == ["deus"]
    assert parallel.translation_candidates("ὁ", side="b").unaligned == 2


def test_translation_candidates_skip_missing_sentences_and_limit_examples(parallel, tmp_path):
    path = tmp_path / "out" / "verses.jsonl"
    path.parent.mkdir()
    rows = [{"sent_id": "s1", "links": [{"source": "1", "target": "2"}]}, {"sent_id": "gone", "links": [{"source": "1", "target": "1"}]}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    parallel.alignment("verses").load()
    assert parallel.translation_candidates("deus").occurrences == 1
    for sent_id in ("s1", "s2"):
        parallel.set_links("verses", sent_id, "1-1S 1-2S" if sent_id == "s1" else "1-1S 1-2S")
    assert len(parallel.translation_candidates("deus").examples) == 4
    parallel.set_links("verses", "s1", "1-1S 1-2S 1-4S 1-5S")
    assert len(parallel.translation_candidates("deus").examples) == 5  # capped


def test_low_confidence_links(parallel):
    parallel.set_links("verses", "s1", "1-2S", confidence=0.9)
    parallel.set_links("verses", "s1", "2-3P", confidence=0.3, replace=False)
    parallel.set_links("verses", "s2", "1-1S", confidence=0.2)
    page = parallel.low_confidence_links("verses", threshold=0.5)
    assert [(l.sent_id, l.link, l.form_a, l.form_b, l.confidence) for l in page.links] == [
        ("s2", "1-1S", "Deus", "θεὸς", 0.2), ("s1", "2-3P", "amat", "ἀγαπᾷ", 0.3),
    ]  # fmt: skip
    assert parallel.low_confidence_links("verses", sent_id="s1", limit=1).total == 1


def test_specs_and_config(tmp_path):
    spec = AlignmentSpec.parse(f"verses=la,grc,{tmp_path}/a.jsonl")
    assert (spec.source_a, spec.source_b, spec.mode.value, spec.default_confidence) == ("la", "grc", "rw", None)
    assert AlignmentSpec.parse(f"g=la,grc,{tmp_path}/g.jsonl:ro:0.9").to_arg() == f"g=la,grc,{tmp_path}/g.jsonl:ro:0.9"
    assert AlignmentSpec.parse(f"g=la,grc,{tmp_path}/g.jsonl:ro").to_arg() == f"g=la,grc,{tmp_path}/g.jsonl:ro"
    for bad in ("verses", "v=la,grc", "=la,grc,x", "v=la,,x"):
        with pytest.raises(CorpusError, match="should be NAME=SOURCE_A,SOURCE_B,PATH"):
            AlignmentSpec.parse(bad)
    from llmparser.mcp.config import SourceSpec

    sources = [SourceSpec.parse(f"la={DATA}/parallel_a.conllu"), SourceSpec.parse(f"grc={DATA}/parallel_b.conllu")]
    store = load_store(sources, alignments=[spec])
    assert store.list_alignments()[0].name == "verses"
    args = claude_desktop_config(sources, alignments=[spec])["mcpServers"]["conllu"]["args"]
    assert args[-2:] == ["--alignment", spec.to_arg()]


@pytest.mark.anyio
async def test_mcp_alignment_tools(parallel):
    async with Client(build_server(parallel)) as client:
        async def call(name, **args):
            result = await client.call_tool(name, args)
            assert not result.is_error, result.content[0].text
            content = result.structured_content
            return content["result"] if set(content) == {"result"} else content

        assert (await call("list_alignments"))[0]["source_b"] == "grc"
        checked = await call("set_links", alignment="verses", sent_id="s1", links="1-2S", dry_run=True, agent="aligner")
        assert checked["links"] == "1-2S" and checked["agents"] == {"1-2": "aligner"}
        assert (await call("get_parallel", alignment="verses", sent_id="s1"))["links"] == ""
        assert (await call("set_links", alignment="verses", sent_id="s1", links="1-2S 2-3P", confidence=0.4))["links"] == "1-2S 2-3P"
        assert (await call("get_parallel", alignment="verses", sent_id="s1"))["unaligned_a"] == ["3"]
        assert (await call("translation_candidates", lemma="amo"))["candidates"][0]["value"] == "ἀγαπάω"
        assert (await call("low_confidence_links", alignment="verses"))["total"] == 2
        assert (await call("remove_links", alignment="verses", sent_id="s1", links="2-3"))["links"] == "1-2S"


def test_cli_alignment_option(monkeypatch, tmp_path):
    from mcp.server import MCPServer
    from typer.testing import CliRunner

    from llmparser.cli import app

    calls = []
    monkeypatch.setattr(MCPServer, "run", lambda self, transport, **kwargs: calls.append(transport))
    sources = ["-s", f"la={DATA}/parallel_a.conllu", "-s", f"grc={DATA}/parallel_b.conllu"]
    result = CliRunner().invoke(app, ["mcp", "serve", *sources, "-a", f"verses=la,grc,{tmp_path}/v.jsonl"])
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(app, ["mcp", "config", *sources, "-a", f"verses=la,grc,{tmp_path}/v.jsonl"])
    assert "--alignment" in result.stdout
    result = CliRunner().invoke(app, ["mcp", "serve", *sources, "-a", "bad"])
    assert result.exit_code == 2 and "Alignment spec 'bad'" in result.stderr


def test_link_agents(parallel):
    view = parallel.set_links("verses", "s1", "1-2S 2-3S", confidence=0.95, agent="rules")
    assert view.agents == {"1-2": "rules", "2-3": "rules"}
    stored = parallel.alignment("verses").links("s1")
    # sent again unchanged: keeps its agent; changed or new: the new agent
    resend = [stored[0], Link(source="2", target="3", confidence=0.5), Link(source="3", target="4", sure=False)]
    view = parallel.set_links("verses", "s1", resend, agent="scorer", dry_run=True)
    assert view.agents == {"1-2": "rules", "2-3": "scorer", "3-4": "scorer"}
    assert stored[0].agent == "rules" and parallel.get_parallel("verses", "s1").agents == {"1-2": "rules", "2-3": "rules"}
    mine = Link(source="1", target="2")
    parallel.set_links("verses", "s1", [mine], confidence=0.7, replace=False, agent="aligner")
    assert mine.confidence is None and mine.agent is None  # the caller's link is not modified
    assert parallel.get_parallel("verses", "s1").agents["1-2"] == "aligner"
