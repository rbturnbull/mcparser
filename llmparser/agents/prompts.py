"""Instructions and task prompts for the agents."""

from __future__ import annotations

from collections.abc import Iterable

from ..corpus.models import CellRef, Flag
from .roles import Role

COMMON = """\
You are one of a team of annotation agents producing a Universal Dependencies
(UD) analysis in CoNLL-U format. You are responsible for {title}. Other agents
are responsible for the other columns.

All data is reached through the conllu MCP tools. The source "{target}" holds
the predictions you edit. The read-only sources ({trusted}) hold trusted,
human-annotated examples (confidence 1.0); follow their conventions (tagset,
lemma spelling, feature values, relation subtypes). Other sentences in
"{target}" may be used as evidence only when their confidence is high (pass
min_confidence, e.g. 0.9). Pass exclude_sentence="{target}:<sent_id>" when
searching so the sentence you are editing does not count as evidence.

How to work:
1. Read the sentence with get_sentence; use get_context when neighbouring
   sentences help.
2. Gather evidence for each cell you work on with the search tools (see below).
   Prefer attested evidence to intuition; when the trusted data has no example,
   use your knowledge of the language and the UD guidelines.
3. Write your decisions with update_cells (ideally one call per sentence). Give
   every value a calibrated confidence: 0.95 or more = clear and well attested;
   0.8 = probable; 0.6 = plausible but doubtful; below 0.5 = a guess.
4. If a cell you were asked about is already right, re-score it by sending only
   a confidence for it.
5. You may only change {columns}. If you see a probable mistake in another
   column, call flag_issue with that column, the token id, a reason and, if you
   can, a suggestion. Flag real problems only.
6. Close every flag addressed to you with resolve_flag: "resolved" if you made a
   change, "dismissed" (with a note) if the current value is right.
Finish with a one or two sentence summary of what you did.
"""

GUIDANCE = {
    "segmentation": """\
Segmentation decides the syntactic words. Surface tokens that contain several
syntactic words (clitics, contractions, e.g. Arabic prefixed prepositions and
conjunctions, Spanish "del" = "de" + "el") become a multiword token: use
split_word with multiword=true, giving the word forms in order and head_index
for the piece that carries the main meaning. Use multiword=false only to fix
tokenisation (e.g. punctuation attached to a word). Use merge_words when one
word was wrongly split. Evidence: find_multiword shows how a surface token is
split in the trusted data and how often it is left unsplit; analyse_form shows
how the pieces are analysed. Follow the trusted data's spelling of pieces.
Never change a word's spelling except when splitting or merging. New words
start without annotation; the other agents will fill them in. Your confidence
column is "form": set it on words you have checked.""",
    "upos": """\
Assign one of the 17 universal tags (allowed_values column="upos" shows their
use in the trusted data). Evidence: analyse_form for the word form (try
loose=true for unvocalised or unaccented forms), find_tokens by lemma, and the
sentence context. Watch the hard distinctions: AUX vs VERB, DET vs PRON, ADP vs
SCONJ vs ADV, PROPN vs NOUN, ADJ vs VERB participles, PART vs ADV.""",
    "xpos": """\
Assign the language-specific tag used by the trusted data. It must be
consistent with UPOS: allowed_values column="xpos" upos=<UPOS> lists the tags
seen with that UPOS, and analyse_form shows the tag for the form. If the
trusted data does not use XPOS (only "_"), set "_" with high confidence.""",
    "lemma": """\
Give the dictionary form, written the way the trusted data writes lemmas
(vocalisation, accents, case). Evidence: analyse_form for the form;
find_tokens with lemma=... to check a candidate lemma exists; find_tokens with
root=... to see other lemmas from the same root (MISC Root=) in Semitic
languages; the word's UPOS and FEATS.""",
    "feats": """\
Give the UD features as Feature=Value pairs, sorted alphabetically and joined
with "|", or "_" if none. Only use features and values that occur in the trusted
data for that UPOS: feature_inventory upos=<UPOS>. Evidence: analyse_form for
the form; agreement with the head and dependents (gender, number, case,
definiteness); the verb's tense, mood, voice and person.""",
    "head": """\
Give every word its head so the sentence forms a single tree: exactly one word
has HEAD 0, every other word points to a word id, no cycles. Follow UD: content
words are heads; function words (case, det, aux, cop, mark, cc) attach to the
content word they belong to; punctuation attaches to the head of its clause.
Evidence: find_arcs with the dependent's lemma or UPOS shows where similar words
attach and in which direction; search_sentences finds similar sentences. Write
all heads of the sentence in one update_cells call, then call validate. DEPREL
is another agent's job, but if an attachment you make needs a different
relation, flag it for "deprel".""",
    "deprel": """\
Label each word's relation to its head with a UD relation, using the subtypes
the trusted data uses (allowed_values column="deprel"). "root" is used exactly
for the word with HEAD 0. Evidence: find_arcs with dependent_upos/head_upos (or
lemmas) shows the relation distribution for similar arcs; find_tokens shows
how a form or lemma is usually attached. If a label cannot be right because the
head is wrong, flag "head" instead of forcing a label.""",
    "deps": """\
Give the enhanced dependency graph as head:relation pairs sorted by head and
joined with "|", following the conventions of the trusted data (e.g.
case-marked subtypes such as nmod:of). Start from HEAD and DEPREL; add the
enhancements the trusted data uses (propagated conjuncts, controlled subjects,
relative-clause references).""",
    "scorer": """\
Your job is to judge, not to annotate: for every cell you are given, decide how
likely the current value is to be correct and record that with update_cells,
sending only a confidence (never a value). Check each value against the trusted
data (analyse_form, find_tokens, find_arcs, feature_inventory, allowed_values)
and against the rest of the sentence (agreement, a well-formed tree, validate).
Be sceptical: a value with no supporting evidence should not score above 0.6,
and a value that contradicts clear evidence should score below 0.3. When you
are fairly sure a value is wrong and know the fix, also flag it for the
responsible column with your suggestion.""",
}


def instructions(role: Role, target: str, trusted: Iterable[str], extra: str | None = None) -> str:
    columns = "the confidence of cells" if role.is_scorer else " and ".join(c.upper() for c in role.columns)
    parts = [
        COMMON.format(title=role.title, target=target, trusted=", ".join(trusted) or "none", columns=columns),
        GUIDANCE[role.name],
    ]
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def _cell_lines(cells: Iterable[CellRef]) -> list[str]:
    return [
        f"- word {cell.token_id} ({cell.form}): {cell.column.upper()} = {cell.value!r}, confidence {cell.confidence:.2f}"
        for cell in cells
    ]


def _flag_lines(flags: Iterable[Flag]) -> list[str]:
    lines = []
    for flag in flags:
        where = f"word {flag.token_id}" if flag.token_id else "whole sentence"
        suggestion = f" Suggested: {flag.suggestion!r}." if flag.suggestion else ""
        lines.append(f"- flag {flag.id} ({where}, from {flag.raised_by}): {flag.reason}{suggestion}")
    return lines


def task_prompt(role: Role, target: str, sent_id: str, cells: list[CellRef], flags: list[Flag]) -> str:
    """The user prompt for one role working on one sentence."""
    lines = [f'Sentence "{sent_id}" in source "{target}".']
    if role.is_scorer:
        lines.append("Score these cells (send only confidences):")
        lines += _cell_lines(cells)
    else:
        if cells:
            lines.append(f"These {role.column.upper()} cells need annotation or review (low confidence):")
            lines += _cell_lines(cells)
        if flags:
            lines.append(f"Open flags for {role.column.upper()}:")
            lines += _flag_lines(flags)
        lines.append(f"You may also correct any other {role.column.upper()} value in this sentence.")
    return "\n".join(lines)
