import pytest

from mcparser.corpus import CorpusError, CorpusStore, TextCorpus
from mcparser.corpus import textcorpus
from mcparser.corpus.errors import NotFoundError

TEXT = "Vino al mercado, el mercado grande.\n\nel perro y los mercados\r\nÁrbol alto; el ÁRBOL\n"
MORE = "vinieron al elmercado\nmercadox\n"


@pytest.fixture
def paths(tmp_path):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text(TEXT, encoding="utf-8")
    second.write_text(MORE, encoding="utf-8")
    return [first, second]


def test_info_counts_words_and_skips_blank_lines(paths):
    corpus = TextCorpus("es", paths)
    info = corpus.info()
    assert (info.name, info.lines, info.words) == ("es", 5, 19)
    assert info.files == [str(path) for path in paths]
    assert corpus.info() == info  # the index is built once


def test_count_exact_and_loose(paths):
    corpus = TextCorpus("es", paths)
    assert corpus.count("mercado") == 2
    assert corpus.count("arbol") == 0
    assert corpus.count("arbol", loose=True) == 2  # Árbol and ÁRBOL fold to the same string
    assert corpus.count("nothing", loose=True) == 0


def test_find_neighbours_and_examples(paths):
    corpus = TextCorpus("es", paths)
    found = corpus.find("mercado", examples=5, width=1)
    assert (found.corpus, found.form, found.total, found.sample) == ("es", "mercado", 2, 2)
    assert [(v.value, v.count) for v in found.previous] == [("al", 1), ("el", 1)]
    assert [(v.value, v.count) for v in found.next] == [("el", 1), ("grande", 1)]
    # one example per line, in the file's own numbering (the blank line counts)
    assert [(e.file, e.line, e.before, e.form, e.after) for e in found.examples] == [("a.txt", 1, "al", "mercado", ", el")]
    assert [(v.value, v.count) for v in found.prefixed] == [("elmercado", 1)]
    assert [(v.value, v.count) for v in found.suffixed] == [("mercados", 1), ("mercadox", 1)]


def test_find_reads_the_second_file_and_spreads_examples(paths):
    corpus = TextCorpus("es", paths)
    found = corpus.find("al", examples=1)
    assert [(e.file, e.line) for e in found.examples] == [("a.txt", 1)]
    found = corpus.find("el", examples=2)
    assert [e.line for e in found.examples] == [1, 3]
    assert corpus.find("elmercado").examples[0].file == "b.txt"
    assert corpus.find("el", examples=0).examples == []


def test_find_loose_and_missing(paths):
    corpus = TextCorpus("es", paths)
    found = corpus.find("arbol", loose=True)
    assert found.total == 2 and {v.value for v in found.forms} == {"Árbol", "ÁRBOL"}
    assert corpus.find("arbol").total == 0
    missing = corpus.find("zzz")
    assert (missing.total, missing.examples, missing.previous, missing.prefixed) == (0, [], [], [])
    assert corpus.find("el").prefixed == []  # shorter than AFFIX_MINIMUM


def test_postings_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(textcorpus, "POSTINGS_MAX", 2)
    path = tmp_path / "many.txt"
    path.write_text("the cat\n" * 5, encoding="utf-8")
    found = TextCorpus("x", [path]).find("cat", examples=5)
    assert found.total == 5 and found.sample == 2 and len(found.examples) == 2


def test_bad_corpus_files(tmp_path):
    with pytest.raises(CorpusError, match="no files"):
        TextCorpus("x", [])
    with pytest.raises(CorpusError, match="does not exist"):
        TextCorpus("x", [tmp_path / "missing.txt"])


def test_store_corpora(paths):
    store = CorpusStore()
    with pytest.raises(NotFoundError, match="Name a corpus; available: none"):
        store.text_corpus()
    store.add_text_corpus("es", paths[:1])
    assert store.text_corpus() is store.text_corpus("es")
    assert store.corpus_find("mercado").total == 2
    assert store.corpus_find("mercado", corpus="es", examples=0).examples == []
    assert [info.name for info in store.list_corpora()] == ["es"]
    with pytest.raises(CorpusError, match="already exists"):
        store.add_text_corpus("es", paths)
    store.add_text_corpus("more", paths[1:])
    with pytest.raises(NotFoundError, match="Name a corpus; available: es, more"):
        store.text_corpus()
    with pytest.raises(NotFoundError, match="No corpus named 'zz'"):
        store.text_corpus("zz")


# ---------------------------------------------------------------- MCP, agents, CLI


@pytest.fixture
def corpus_store(store, paths):
    store.add_text_corpus("es", paths)
    return store


@pytest.mark.anyio
async def test_mcp_corpus_tools(corpus_store, store):
    from mcp import Client

    from mcparser.mcp import build_server

    async with Client(build_server(corpus_store)) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert {"list_corpora", "corpus_find"} <= names
        assert "Text corpora" in client.instructions
        listed = await client.call_tool("list_corpora", {})
        assert listed.structured_content["result"][0]["name"] == "es"
        found = await client.call_tool("corpus_find", {"form": "mercado", "examples": 1, "width": 2})
        assert found.structured_content["total"] == 2
        assert found.structured_content["examples"][0]["after"] == ", el mercado"
        unknown = await client.call_tool("corpus_find", {"form": "x", "corpus": "zz"})
        assert unknown.is_error
    plain = CorpusStore()
    async with Client(build_server(plain)) as client:
        assert not {"list_corpora", "corpus_find"} & {tool.name for tool in (await client.list_tools()).tools}
        assert "Text corpora" not in client.instructions


def test_corpus_spec_and_config(paths, train_path):
    from mcparser.mcp.config import CorpusSpec, SourceSpec, claude_desktop_config, load_store

    spec = CorpusSpec.parse(f"es={paths[0]},{paths[1]}")
    assert (spec.name, spec.paths) == ("es", [paths[0].resolve(), paths[1].resolve()])
    assert CorpusSpec.parse(str(paths[0])).name == "a"
    assert CorpusSpec.parse(spec.to_arg()) == spec
    with pytest.raises(CorpusError, match="no path"):
        CorpusSpec.parse("es=")
    store = load_store([SourceSpec.parse(str(train_path))], corpora=[spec])
    assert list(store.text_corpora) == ["es"]
    args = claude_desktop_config([SourceSpec.parse(str(train_path))], corpora=[spec])["mcpServers"]["conllu"]["args"]
    assert args[args.index("--corpus") + 1] == spec.to_arg()


def test_compact_corpus_results(corpus_store):
    from mcparser.agents.compact import compact_result
    from mcparser.agents.roles import ROLES

    found = corpus_store.corpus_find("mercado", examples=1, width=2).model_dump(mode="json")
    text = compact_result("corpus_find", found, ROLES["upos"])
    assert text.splitlines()[0] == "'mercado': 2× in corpus 'es' (no analyses, only use)."
    assert "preceded by (in 2 sampled): al×1, el×1" in text and "with a prefix: elmercado×1" in text
    assert "- a.txt:1 Vino al [mercado] , el mercado" in text
    alone = corpus_store.corpus_find("perro", examples=0).model_dump(mode="json")
    assert "with a prefix" not in compact_result("corpus_find", alone, ROLES["upos"])
    missing = corpus_store.corpus_find("zzz").model_dump(mode="json")
    assert compact_result("corpus_find", missing, ROLES["upos"]) == "'zzz': not in the unannotated corpus 'es'."
    infos = [info.model_dump(mode="json") for info in corpus_store.list_corpora()]
    assert compact_result("list_corpora", infos, ROLES["upos"]) == "- es: 5 lines, 19 words"
    assert compact_result("list_corpora", [], ROLES["upos"]) == "None."


def test_prompts_mention_the_corpus_only_when_there_is_one():
    from mcparser.agents.prompts import instructions
    from mcparser.agents.roles import ROLES

    assert "Unannotated text (arabic, latin) is also available" in instructions(ROLES["upos"], "pred", [], corpora=["arabic", "latin"])
    assert "Unannotated text" not in instructions(ROLES["upos"], "pred", [])


def test_evidence_from_the_corpus(corpus_store, monkeypatch):
    from mcparser.agents import evidence

    from mcparser.agents.evidence import EvidenceBuilder
    from mcparser.agents.roles import ROLES

    builder = EvidenceBuilder(corpus_store, "pred")
    assert builder.corpus("mercados") == "text ×1; after los×1"
    assert builder.corpus("mercado") == "text ×2; after al×1, el×1; before el×1, grande×1; prefixed elmercado×1; suffixed mercados×1, mercadox×1"
    assert builder.corpus("zzz") == "text ×0"
    monkeypatch.setattr(evidence, "RARE_COUNT", 1)  # now the treebank's single example of "mercado" is enough
    assert EvidenceBuilder(corpus_store, "pred").corpus("mercado") == ""
    monkeypatch.undo()
    assert EvidenceBuilder(corpus_store, "pred").build(ROLES["lemma"], [("p2", [])]).splitlines()[0].endswith(
        "word-ending analogies and unannotated-text counts for rare forms):"
    )
    store = CorpusStore()
    store.add_source("pred", corpus_store.source("pred").path, mode="rw")
    assert EvidenceBuilder(store, "pred").corpus("zzz") == ""
    assert EvidenceBuilder(store, "pred").stems("zzz") == ""


def test_segmentation_evidence_counts_stems(corpus_store):
    from mcparser.agents.evidence import EvidenceBuilder
    from mcparser.agents.roles import ROLES

    corpus_store.set_cell("pred", "p2", "1", "form", "elmercado")
    corpus_store.set_cell("pred", "p2", "2", "form", "x")
    lines = EvidenceBuilder(corpus_store, "pred").build(ROLES["segmentation"], [("p2", [])]).splitlines()
    assert lines[1] == "- No token of these sentences is ever split in the trusted data."
    assert "a clitic" in lines[2]
    assert lines[3:] == ["- elmercado: elmercado×1, lmercado×0, mercado×2", "- x: x×0"]


def test_corpus_tool_is_a_setting(corpus_store):
    from pydantic_ai.exceptions import ModelRetry

    from mcparser.agents.prompts import instructions
    from mcparser.agents.roles import CORPUS_TOOLS, READ_TOOLS, ROLES
    from mcparser.agents.team import AgentTeam
    from mcparser.agents.toolset import role_toolset
    from mcparser.settings import Settings

    assert "corpus_find" not in READ_TOOLS
    with pytest.raises(ModelRetry, match="not available"):
        role_toolset(None, ROLES["upos"], "pred").check("corpus_find", {})
    role_toolset(None, ROLES["upos"], "pred", CORPUS_TOOLS).check("corpus_find", {})
    assert "corpus_find gives more" in instructions(ROLES["upos"], "pred", [], corpora=["es"])
    assert "corpus_find" not in instructions(ROLES["upos"], "pred", [], corpora=["es"], corpus_tool=False)
    assert AgentTeam(corpus_store, "pred", Settings()).corpus_tool
    assert not AgentTeam(corpus_store, "pred", Settings(corpus_tool=False)).corpus_tool
    assert not AgentTeam(CorpusStore(), "pred", Settings()).corpus_tool
