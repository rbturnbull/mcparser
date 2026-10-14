import json

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from llmparser.mcp import build_server

READ_TOOLS = {
    "list_sources", "list_sentences", "get_sentence", "get_context", "get_token", "search_sentences",
    "find_tokens", "analyse_form", "find_multiword", "find_arcs", "allowed_values", "feature_inventory",
    "validate", "low_confidence", "list_flags", "get_history",
}  # fmt: skip
EDIT_TOOLS = {
    "set_cell", "update_cells", "split_word", "merge_words", "delete_word", "insert_word",
    "add_multiword", "remove_multiword", "flag_issue", "resolve_flag", "save", "reload",
}  # fmt: skip


async def call(client, name, **arguments):
    result = await client.call_tool(name, arguments)
    assert not result.is_error, result.content[0].text
    content = result.structured_content
    return content["result"] if set(content) == {"result"} else content


@pytest.mark.anyio
async def test_lists_all_tools_with_annotations(store):
    async with Client(build_server(store)) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert set(tools) == READ_TOOLS | EDIT_TOOLS
        assert tools["find_tokens"].annotations.read_only_hint is True
        assert tools["delete_word"].annotations.destructive_hint is True
        assert "Read-only sources" in client.instructions


@pytest.mark.anyio
async def test_read_only_server_has_no_edit_tools(store):
    async with Client(build_server(store, allow_edits=False)) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        assert tools == READ_TOOLS


@pytest.mark.anyio
async def test_read_tools(store):
    async with Client(build_server(store)) as client:
        assert [s["name"] for s in await call(client, "list_sources")] == ["train", "pred"]
        assert (await call(client, "list_sentences", source="pred", order="confidence_asc"))["total"] == 2
        assert (await call(client, "get_sentence", source="pred", sent_id="p1"))["tokens"] is None
        assert len(await call(client, "get_context", source="train", sent_id="t2", before=0, after=0)) == 1
        assert (await call(client, "get_token", source="train", sent_id="t3", token_id="5"))["head"]["form"] == "fue"
        assert (await call(client, "search_sentences", query="mercado", sources=["train"]))[0]["sent_id"] == "t2"
        assert (await call(client, "find_tokens", upos="DET", feats="Gender=Masc"))["total"] == 2
        assert (await call(client, "analyse_form", form="el"))["groups"][0]["values"]["upos"] == "DET"
        assert (await call(client, "find_multiword", surface="del"))["analyses"][0]["pieces"] == ["de", "el"]
        assert (await call(client, "find_arcs", deprel="det"))["total"] == 4
        assert (await call(client, "allowed_values", column="deprel"))[0]["value"] in {"punct", "det", "root"}
        assert "Definite" in await call(client, "feature_inventory", upos="DET")
        assert (await call(client, "validate", source="pred"))["errors"] == 0
        assert (await call(client, "low_confidence", source="pred", columns=["upos"], limit=1))["total"] == 5
        assert (await call(client, "list_flags", source="pred"))["total"] == 0
        assert await call(client, "get_history", source="pred") == []


@pytest.mark.anyio
async def test_edit_tools(store, pred_path):
    async with Client(build_server(store)) as client:
        result = await call(
            client, "set_cell", source="pred", sent_id="p2", token_id="1", column="upos", value="DET",
            confidence=0.95, reason="article", agent="pos",
        )  # fmt: skip
        assert result["edits"][0]["agent"] == "pos"
        updates = [{"token_id": "2", "column": "lemma", "confidence": 0.9}]
        assert (await call(client, "update_cells", source="pred", sent_id="p2", updates=updates))["edits"][0]["action"] == "confidence"
        split = await call(client, "split_word", source="pred", sent_id="p1", token_id="2", forms=["a", "el"], multiword=True, head_index=1)
        assert split["id_map"]["3"] == "4"
        await call(client, "remove_multiword", source="pred", sent_id="p1", range_id="2-3")
        await call(client, "merge_words", source="pred", sent_id="p1", token_ids=["2", "3"], form="al")
        await call(client, "insert_word", source="pred", sent_id="p1", after="4", form="!")
        await call(client, "delete_word", source="pred", sent_id="p1", token_id="5")
        await call(client, "add_multiword", source="pred", sent_id="p1", start=2, end=3, form="almercado")
        flag = await call(client, "flag_issue", source="pred", sent_id="p2", column="head", reason="?", token_id="1", suggestion="2")
        assert (await call(client, "resolve_flag", source="pred", flag_id=flag["id"], status="dismissed", note="ok"))["status"] == "dismissed"
        assert await call(client, "save") == ["pred"]
        assert (await call(client, "reload", source="pred"))["open_flags"] == 4
        assert "Conf=upos:0.95" in pred_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_corpus_errors_become_tool_errors(store):
    async with Client(build_server(store)) as client:
        result = await client.call_tool("set_cell", {"source": "train", "sent_id": "t1", "token_id": "1", "column": "upos", "value": "X"})
        assert result.is_error
        assert "read-only" in result.content[0].text


@pytest.mark.anyio
async def test_resources(store):
    async with Client(build_server(store)) as client:
        sources = await client.read_resource("conllu://sources")
        assert [s["name"] for s in json.loads(sources.contents[0].text)] == ["train", "pred"]
        sentence = await client.read_resource("conllu://pred/p2")
        assert sentence.contents[0].text.startswith("# sent_id = p2")
        with pytest.raises(MCPError, match="No sentence"):
            await client.read_resource("conllu://pred/missing")
