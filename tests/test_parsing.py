from types import SimpleNamespace

import llmloader

from llmparser.parsing import parse_items


class DummyDatabase:
    def __init__(self, examples):
        self._examples = list(examples)

    def similarity_search(self, _sentence: str, k: int = 100):
        return self._examples[:k]


def test_parse_items_uses_examples_and_text() -> None:
    llm = llmloader.load("dummy")
    data = {"item1": {"text": "Hello world."}}
    examples = [SimpleNamespace(metadata={"conllu": "1\tHello\t_\tINTJ\t_\t_\t0\troot\t_\t_"})]
    database = DummyDatabase(examples)

    parsed = parse_items(data, database, llm, n_examples=1)

    assert "conllu" in parsed["item1"]
    assert "Hello world." in parsed["item1"]["conllu"]
    assert examples[0].metadata["conllu"] in parsed["item1"]["conllu"]
