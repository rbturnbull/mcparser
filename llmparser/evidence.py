from __future__ import annotations

from typing import Any

from .conllu_model import ConlluSentence, ConlluToken


class TrainingEvidence:
    def __init__(self, database):
        self.database = database

    def find_similar_sentences(self, text: str, k: int = 10) -> list[dict[str, Any]]:
        return list(self.database(text, k=k))

    def find_examples_for_form(self, form: str, k: int = 10) -> list[dict[str, Any]]:
        return list(self.database(f"FORM {form}", k=k))

    def find_examples_for_lemma(self, lemma: str, k: int = 10) -> list[dict[str, Any]]:
        return list(self.database(f"LEMMA {lemma}", k=k))

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
