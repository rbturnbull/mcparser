from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .conllu_model import ConlluSentence, ConlluToken


def _example_key(example: dict[str, Any]) -> str:
    sent_id = example.get("sent_id")
    if sent_id is not None:
        return f"sent_id:{sent_id}"
    conllu = example.get("conllu")
    if conllu is not None:
        return f"conllu:{conllu}"
    return repr(sorted(example.items()))


class TokenEvidence(BaseModel):
    form_examples: list[dict[str, Any]] = Field(default_factory=list)
    lemma_examples: list[dict[str, Any]] = Field(default_factory=list)
    form_upos_examples: list[dict[str, Any]] = Field(default_factory=list)
    similar_sentence_examples: list[dict[str, Any]] = Field(default_factory=list)

    def all_examples(self) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in (
            self.form_examples,
            self.lemma_examples,
            self.form_upos_examples,
            self.similar_sentence_examples,
        ):
            for example in group:
                key = _example_key(example)
                if key in seen:
                    continue
                seen.add(key)
                examples.append(example)
        return examples


class SegmentationEvidence(BaseModel):
    surface_examples: list[dict[str, Any]] = Field(default_factory=list)
    subform_examples: list[list[dict[str, Any]]] = Field(default_factory=list)
    sequence_examples: list[dict[str, Any]] = Field(default_factory=list)
    similar_sentence_examples: list[dict[str, Any]] = Field(default_factory=list)

    def all_examples(self) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        seen: set[str] = set()
        groups = [
            self.surface_examples,
            self.sequence_examples,
            self.similar_sentence_examples,
        ]
        groups.extend(self.subform_examples)
        for group in groups:
            for example in group:
                key = _example_key(example)
                if key in seen:
                    continue
                seen.add(key)
                examples.append(example)
        return examples


class TrainingEvidence:
    def __init__(self, database):
        self.database = database

    def find_similar_sentences(self, text: str, k: int = 10) -> list[dict[str, Any]]:
        return list(self.database(text, k=k))

    def find_examples_for_form(self, form: str, k: int = 10) -> list[dict[str, Any]]:
        return list(self.database(f"FORM {form}", k=k))

    def find_examples_for_lemma(self, lemma: str, k: int = 10) -> list[dict[str, Any]]:
        return list(self.database(f"LEMMA {lemma}", k=k))

    def find_examples_for_form_upos(self, form: str, upos: str, k: int = 10) -> list[dict[str, Any]]:
        return list(self.database(f"FORM {form} UPOS {upos}", k=k))

    def find_examples_for_surface(self, surface: str, k: int = 10) -> list[dict[str, Any]]:
        return list(self.database(f"SURFACE {surface}", k=k))

    def find_examples_for_subform(self, subform: str, k: int = 10) -> list[dict[str, Any]]:
        return list(self.database(f"SUBFORM {subform}", k=k))

    def find_examples_for_sequence(self, pieces: list[str], k: int = 10) -> list[dict[str, Any]]:
        return list(self.database(f"SEQUENCE {' '.join(pieces)}", k=k))

    def find_segmentation_evidence(
        self,
        surface: str,
        pieces: list[str],
        sentence_text: str,
        k: int = 10,
    ) -> SegmentationEvidence:
        return SegmentationEvidence(
            surface_examples=self.find_examples_for_surface(surface, k=k),
            subform_examples=[
                self.find_examples_for_subform(piece, k=k)
                for piece in pieces
            ],
            sequence_examples=self.find_examples_for_sequence(pieces, k=k) if pieces else [],
            similar_sentence_examples=self.find_similar_sentences(sentence_text, k=k),
        )

    def find_token_evidence(
        self,
        sentence: ConlluSentence,
        token: ConlluToken,
        column: str,
        k: int = 10,
    ) -> TokenEvidence:
        form_examples = self.find_examples_for_form(token.form, k=k)

        lemma_examples: list[dict[str, Any]] = []
        if token.lemma not in (None, "_", ""):
            lemma_examples = self.find_examples_for_lemma(str(token.lemma), k=k)

        form_upos_examples: list[dict[str, Any]] = []
        if token.upos not in (None, "_", ""):
            form_upos_examples = self.find_examples_for_form_upos(token.form, str(token.upos), k=k)

        return TokenEvidence(
            form_examples=form_examples,
            lemma_examples=lemma_examples,
            form_upos_examples=form_upos_examples,
            similar_sentence_examples=self.find_similar_sentences(sentence.text, k=k),
        )

    def find_examples_for_token_column(
        self,
        sentence: ConlluSentence,
        token: ConlluToken,
        column: str,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        if column == "lemma":
            return self.find_examples_for_form(token.form, k=k)
        if column == "form":
            return self.find_examples_for_form(token.form, k=k)

        query_parts = [
            f"FORM {token.form}",
            f"COLUMN {column}",
        ]
        if token.lemma not in (None, "_", ""):
            query_parts.append(f"LEMMA {token.lemma}")
        if sentence.text:
            query_parts.append(sentence.text)
        return list(self.database(" ".join(query_parts), k=k))
