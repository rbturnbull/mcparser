from __future__ import annotations

from typing import Any

import conllu
from pydantic import BaseModel, ConfigDict


CONLLU_COLUMNS = (
    "id",
    "form",
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc",
)


class ConlluToken(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Any
    form: str
    lemma: str | None
    upos: str | None
    xpos: str | None
    feats: Any
    head: Any
    deprel: str | None
    deps: Any
    misc: Any

    @classmethod
    def from_mapping(cls, token: dict[str, Any]) -> "ConlluToken":
        return cls(**{column: token.get(column) for column in CONLLU_COLUMNS})

    def to_mapping(self) -> dict[str, Any]:
        return {column: getattr(self, column) for column in CONLLU_COLUMNS}

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def set_column(self, column: str, value: Any) -> None:
        column = column.lower()
        if column not in CONLLU_COLUMNS:
            raise ValueError(f"Unsupported CoNLL-U column: {column}")
        setattr(self, column, value)


class ConlluSentence(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: list[ConlluToken]
    metadata: dict[str, Any]

    @classmethod
    def from_text(cls, text: str) -> "ConlluSentence":
        sentences = conllu.parse(text)
        if len(sentences) != 1:
            raise ValueError(f"Expected exactly one CoNLL-U sentence, found {len(sentences)}")
        return cls.from_tokenlist(sentences[0])

    @classmethod
    def from_tokenlist(cls, tokenlist) -> "ConlluSentence":
        return cls(
            tokens=[ConlluToken.from_mapping(dict(token)) for token in tokenlist],
            metadata=dict(tokenlist.metadata),
        )

    @classmethod
    def iter_from_text(cls, text: str) -> list["ConlluSentence"]:
        return [cls.from_tokenlist(sentence) for sentence in conllu.parse(text)]

    @property
    def sent_id(self) -> str | None:
        sent_id = self.metadata.get("sent_id")
        return str(sent_id) if sent_id is not None else None

    @property
    def text(self) -> str:
        return str(self.metadata.get("text", "")).strip()

    def token_by_id(self, token_id: Any) -> ConlluToken:
        for token in self.tokens:
            if token.id == token_id:
                return token
        raise KeyError(f"No token with id {token_id!r}")

    def apply_patch(self, patch: dict[str, Any]) -> None:
        token_id = patch["token_id"]
        column = patch["column"]
        value = patch["value"]
        self.token_by_id(token_id).set_column(column, value)

    def to_tokenlist(self):
        return conllu.TokenList(
            [token.to_mapping() for token in self.tokens],
            metadata=self.metadata,
        )

    def to_conllu(self) -> str:
        return self.to_tokenlist().serialize().strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sent_id": self.sent_id,
            "text": self.text,
            "tokens": [token.to_dict() for token in self.tokens],
            "conllu": self.to_conllu(),
        }
