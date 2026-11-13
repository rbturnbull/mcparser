from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

from llmparser.prompt import build_chain, build_prompt


def test_build_prompt_includes_text_and_examples() -> None:
    text = "Hello world."
    examples = [SimpleNamespace(metadata={"conllu": "1\tHello\t_\tINTJ\t_\t_\t0\troot\t_\t_"})]

    prompt = build_prompt(text, examples)
    messages = prompt.format_messages()

    assert len(messages) == 2
    assert "computational linguist" in messages[0].content
    assert text in messages[1].content
    assert examples[0].metadata["conllu"] in messages[1].content


def test_build_chain_invokes_llm_and_parses_output() -> None:
    llm = RunnableLambda(lambda _: "ok")
    chain = build_chain(llm, "Test.", [])

    assert chain.invoke({}) == "ok"
