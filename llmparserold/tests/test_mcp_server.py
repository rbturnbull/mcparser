from llmparser.correction import CorrectionPatch
from llmparser.mcp_server import CorrectionSession, CorrectionToolRegistry


PREDICTED_CONLLU = """# sent_id = s1
# text = Hello world.
1	Hello	hello	NOUN	_	_	2	nsubj	_	_
2	world	world	NOUN	_	_	0	root	_	_
"""


class DummyDatabase:
    def __init__(self):
        self.queries = []

    def __call__(self, query: str, k: int = 10):
        self.queries.append((query, k))
        return [
            {
                "sent_id": "train1",
                "text": "Hello there.",
                "conllu": "1\tHello\thello\tINTJ\t_\t_\t0\troot\t_\t_",
            }
        ]


def test_correction_session_loads_sentences_by_sent_id() -> None:
    session = CorrectionSession.from_conllu_text(PREDICTED_CONLLU, DummyDatabase())

    sentence = session.get_sentence("s1")

    assert sentence["sent_id"] == "s1"
    assert sentence["tokens"][0]["form"] == "Hello"


def test_correction_session_gets_token() -> None:
    session = CorrectionSession.from_conllu_text(PREDICTED_CONLLU, DummyDatabase())

    token = session.get_token("s1", 1)

    assert token["id"] == 1
    assert token["form"] == "Hello"


def test_correction_session_gets_token_evidence() -> None:
    database = DummyDatabase()
    session = CorrectionSession.from_conllu_text(PREDICTED_CONLLU, database)

    evidence = session.get_token_evidence("s1", 1, "lemma", k=1)

    assert evidence["form_examples"][0]["sent_id"] == "train1"
    assert database.queries[:2] == [
        ("FORM Hello", 1),
        ("LEMMA hello", 1),
    ]


def test_correction_session_gets_segmentation_evidence() -> None:
    database = DummyDatabase()
    session = CorrectionSession.from_conllu_text(PREDICTED_CONLLU, database)

    evidence = session.get_segmentation_evidence("s1", "Hello", ["He", "llo"], k=1)

    assert evidence["surface_examples"][0]["sent_id"] == "train1"
    assert database.queries[:3] == [
        ("SURFACE Hello", 1),
        ("SUBFORM He", 1),
        ("SUBFORM llo", 1),
    ]


def test_correction_session_applies_patch_and_serializes_sentence() -> None:
    session = CorrectionSession.from_conllu_text(PREDICTED_CONLLU, DummyDatabase())

    patch = CorrectionPatch(token_id=1, column="upos", value="INTJ")
    updated = session.apply_patch("s1", patch)
    serialized = session.serialize_sentence("s1")

    assert updated["tokens"][0]["upos"] == "INTJ"
    assert "1\tHello\thello\tINTJ" in serialized


def test_correction_session_validates_prediction() -> None:
    invalid = """# sent_id = bad
# text = Broken.
1	Broken	broken	ADJ	_	_	4	root	_	_
"""
    session = CorrectionSession.from_conllu_text(invalid, DummyDatabase())

    errors = session.validate_prediction("bad")

    assert errors[0]["code"] == "invalid-head"


def test_correction_tool_registry_lists_and_calls_tools() -> None:
    session = CorrectionSession.from_conllu_text(PREDICTED_CONLLU, DummyDatabase())
    registry = CorrectionToolRegistry(session)

    assert registry.tool_names() == [
        "apply_patch",
        "apply_segmentation_patch",
        "apply_token_split",
        "get_segmentation_evidence",
        "get_sentence",
        "get_token",
        "get_token_evidence",
        "serialize_sentence",
        "validate_prediction",
    ]

    token = registry.call_tool("get_token", item_id="s1", token_id=1)

    assert token["form"] == "Hello"


def test_correction_tool_registry_applies_patch_from_dict() -> None:
    session = CorrectionSession.from_conllu_text(PREDICTED_CONLLU, DummyDatabase())
    registry = CorrectionToolRegistry(session)

    updated = registry.call_tool(
        "apply_patch",
        item_id="s1",
        patch={"token_id": 1, "column": "upos", "value": "INTJ"},
    )

    assert updated["tokens"][0]["upos"] == "INTJ"


def test_correction_session_applies_token_split() -> None:
    session = CorrectionSession.from_conllu_text(PREDICTED_CONLLU, DummyDatabase())
    result = session.apply_token_split(
        "s1",
        {
            "token_id": 1,
            "replacement_tokens": [
                {
                    "id": 1,
                    "form": "He",
                    "lemma": "he",
                    "upos": "PRON",
                    "xpos": "_",
                    "feats": None,
                    "head": None,
                    "deprel": None,
                    "deps": None,
                    "misc": None,
                },
                {
                    "id": 2,
                    "form": "llo",
                    "lemma": "llo",
                    "upos": "X",
                    "xpos": "_",
                    "feats": None,
                    "head": None,
                    "deprel": None,
                    "deps": None,
                    "misc": None,
                },
            ],
        },
    )

    assert result["status"] == "patched"
    assert session.get_sentence("s1")["tokens"][0]["form"] == "He"
    assert "apply_token_split" in CorrectionToolRegistry(session).tool_names()


def test_correction_session_applies_generic_segmentation_patch() -> None:
    session = CorrectionSession.from_conllu_text(PREDICTED_CONLLU, DummyDatabase())
    result = session.apply_segmentation_patch(
        "s1",
        {
            "type": "delete",
            "token_id": 1,
        },
    )

    assert result["status"] == "patched"
    assert session.get_sentence("s1")["tokens"][0]["form"] == "world"
    assert "apply_segmentation_patch" in CorrectionToolRegistry(session).tool_names()
