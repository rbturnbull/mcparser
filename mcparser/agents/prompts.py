"""Instructions and task prompts for the agents."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..corpus.models import CellRef, Flag
from .compact import misc as misc_without_conf
from .roles import Role


@dataclass
class TaskItem:
    """One sentence of a task: its CoNLL-U, the cells to work on and open flags."""

    sent_id: str
    cells: list[CellRef]
    flags: list[Flag] = field(default_factory=list)
    conllu: str = ""

COMMON = """\
You are one of a team of annotation agents producing a Universal Dependencies
(UD) analysis in CoNLL-U format. You are responsible for {title}. Other agents
and code rules are responsible for the other columns.

Your goal is to reproduce the annotation conventions of the trusted treebank
({trusted}), not your own preferred analysis. When the evidence shows how the
treebank analyses a form, an ending or a construction, follow it unless the
context clearly rules it out; when it has no example, use your knowledge of the
language and the UD guidelines. The source "{target}" holds the predictions;
cells already settled by code are not shown to you.

How to work:
1. The task gives you the sentences (CoNLL-U; empty columns at the end of a
   line are left out), the cells to work on (word ids, "3..7" meaning words 3
   to 7, with their confidence; the values are in the table), open flags, and
   the main evidence from the treebank.
   Search tools are available (analyse_form with previous_form, find_tokens
   with suffix, relation_stats, find_arcs, ...) but call them only when the
   evidence given is not enough. Make independent calls in parallel.
2. {answer}
3. Give every value a calibrated confidence: 0.95 or more = clear and well
   attested; 0.8 = probable; 0.6 = plausible but doubtful; below 0.5 = a guess.
4. If you see a probable mistake in another column, report it as a flag (with
   the column, the word, a reason and, if you can, a suggestion). Flag real
   problems only.
"""

ANSWERS = {
    "annotation": (
        "Answer with ONE final result. Put one line per cell in `values`: "
        '"SENTENCE TOKEN_ID VALUE CONFIDENCE" (e.g. "s1 3 {example} 0.95", where s1 is the sentence label), covering every cell listed '
        "and any other {columns} value you correct. Close each open flag addressed to you in `resolved`: "
        '"resolved" if you changed something, "dismissed" (with a note) if not. Keep `summary` under 15 words.'
    ),
    "scores": (
        "Answer with ONE final result: lines \"SENTENCE TOKEN_ID COLUMN CONFIDENCE\" in `scores`, where SENTENCE "
        'is the label (s1, s2, ...). Be brief: start each sentence with "s1 * * 0.95" (all its listed cells), '
        'then add a line only for the cells that deserve a different score, e.g. "s1 3 upos 0.4" or "s1 3 * 0.6" '
        "(all listed cells of word 3). Later lines override earlier ones. Keep `summary` under 15 words."
    ),
    "syntax": (
        "Answer with ONE final result. Put one line per word in `values`: "
        '"SENTENCE TOKEN_ID HEAD DEPREL CONFIDENCE" (e.g. "s1 3 4 nsubj 0.95"; s1 is the sentence label), or with '
        'separate confidences for HEAD and DEPREL, "s1 3 4 nsubj 0.95 0.7". Give both values for every word listed, '
        "even if only one of them is listed (repeat the current value if it is right), and for any other word whose "
        "head or relation you correct. Close each open flag addressed to you in `resolved`: "
        '"resolved" if you changed something, "dismissed" (with a note) if not. Keep `summary` under 15 words.'
    ),
    "tools": (
        "Make your changes with the tools (split_word, merge_words, ...; update_cells to set the confidence of "
        "column FORM for words you checked), close flags addressed to you with resolve_flag, then finish with a "
        "summary of at most 15 words."
    ),
}

EXAMPLES = {"syntax": "4 nsubj", "lemma": "dico", "upos": "NOUN", "xpos": "Nb", "feats": "Case=Nom|Number=Sing", "head": "4", "deprel": "nsubj", "deps": "4:nsubj"}

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
definiteness); the verb's tense, mood, voice and person.
Where the treebank leaves an ambiguous form unresolved with a combined value
(e.g. Gender=Masc,Neut for an article form), use the combined value exactly as
the treebank does rather than choosing one.""",
    "head": """\
Give every word its head so the sentence forms a single tree: exactly one word
has HEAD 0, every other word points to a word id, no cycles. Follow UD: content
words are heads; function words (case, det, aux, cop, mark, cc) attach to the
content word they belong to; punctuation attaches to the head of its clause.
Give the heads of all words of a sentence together: an answer that leaves a
cycle or the wrong number of roots is sent back. The evidence shows where
similar words attach and in which direction; find_arcs and search_sentences
give more. In coordination, follow how the treebank attaches shared
dependents. DEPREL is another agent's job, but if an attachment you make needs
a different relation, flag it for "deprel".""",
    "deprel": """\
Label each word's relation to its head with a UD relation, using the subtypes
the trusted data uses (allowed_values column="deprel"). "root" is used exactly
for the word with HEAD 0. The evidence gives the relations the treebank uses
for arcs like each word's current one (same lemmas, or same lemma and head
UPOS and direction, or same UPOS pair); prefer them, and use relation_stats or
find_arcs for more. If a label cannot be right because the head is wrong, flag
"head" instead of forcing a label.""",
    "syntax": """\
Build the dependency tree and label it: give every word its head and its
relation, deciding both together. Exactly one word has HEAD 0 and DEPREL root;
every other word points to a word id; no cycles. Follow UD: content words are
heads; function words (case, det, aux, cop, mark, cc) attach to the content
word they belong to; punctuation attaches to the head of its clause. Use the
relations and subtypes of the trusted data. Answer for all listed words of a
sentence together: an answer that leaves a cycle or the wrong number of roots
is sent back. The evidence shows where similar words attach and with which
relation, similar treebank sentences and fixed expressions; once a word has a
head, it also shows the relations the treebank uses for arcs like its current
one. Prefer those relations, and when they do not fit, ask whether the head is
wrong. Watch the treebank's conventions for coordination (where the conjunction
and shared dependents attach), dislocated and fronted phrases, and clauses
(mark vs aux, advcl vs acl). UD rules that models often break: every conjunct
attaches to the FIRST conjunct (never to the previous conjunct), and each cc to
the conjunct that follows it; in a copular clause the predicate (not the cop) is
the head and the copula attaches to it; the final punctuation mark attaches to
the root of its sentence, other punctuation to the head of the phrase or clause
it closes or opens (look at how the treebank treats the same mark); with a
passive verb, use the passive subject relation (nsubj:pass) if the treebank has
it. Open flags may come from code that compared your
labels with the treebank: take them seriously, but the context decides.""",
    "deps": """\
Give the enhanced dependency graph as head:relation pairs sorted by head and
joined with "|", following the conventions of the trusted data (e.g.
case-marked subtypes such as nmod:of). Start from HEAD and DEPREL; add the
enhancements the trusted data uses (propagated conjuncts, controlled subjects,
relative-clause references).""",
    "scorer": """\
Your job is to judge, not to annotate: for every cell you are given, decide how
likely the current value is to be correct. Check each value
against the evidence given (the form's analyses, how often its arc is attested)
and the rest of the sentence (agreement, a well-formed tree); use the search
tools only when that is not enough.
Be sceptical: a value with no supporting evidence should not score above 0.6,
and a value that contradicts clear evidence should score below 0.3. When you
are fairly sure a value is wrong and know the fix, also flag it for the
responsible column with your suggestion.""",
}


SYNTAX_ROLES = frozenset({"syntax", "head", "deprel", "deps", "scorer"})


def instructions(
    role: Role, target: str, trusted: Iterable[str], extra: str | None = None, relations: str | None = None
) -> str:
    columns = "the confidence of cells" if role.is_scorer else " and ".join(c.upper() for c in role.columns)
    kind = "scores" if role.is_scorer else "tools" if role.edits_with_tools else "syntax" if role.name == "syntax" else "annotation"
    answer = ANSWERS[kind].format(example=EXAMPLES.get(role.name, ""), columns=columns)
    parts = [
        COMMON.format(title=role.title, target=target, trusted=", ".join(trusted) or "none", answer=answer),
        GUIDANCE[role.name],
    ]
    if relations and role.name in SYNTAX_ROLES:
        parts.append(f"Relations used in the treebank (with counts; use its subtypes):\n{relations}")
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def _ranges(ids: list[str]) -> list[str]:
    """``["1", "2", "3", "5", "6.1"]`` → ``["1..3", "5", "6.1"]`` (runs of consecutive word numbers)."""
    runs: list[list[str]] = []
    for token_id in ids:
        last = runs[-1][-1] if runs else ""
        if token_id.isdigit() and last.isdigit() and int(token_id) == int(last) + 1:
            runs[-1].append(token_id)
        else:
            runs.append([token_id])
    return [run[0] if len(run) == 1 else f"{run[0]}..{run[-1]}" for run in runs]


def _cell_line(cells: list[CellRef]) -> str:
    """The cells of one column as ranges of word ids sharing a confidence: ``1..5 9 at 0.00, 7 at 0.60``."""
    groups: list[tuple[str, list[str]]] = []
    for cell in cells:
        confidence = f"{cell.confidence:.2f}"
        if groups and groups[-1][0] == confidence:
            groups[-1][1].append(cell.token_id)
        else:
            groups.append((confidence, [cell.token_id]))
    return ", ".join(f"{' '.join(_ranges(ids))} at {confidence}" for confidence, ids in groups)


def _score_line(cells: list[CellRef]) -> str:
    """The cells to score, per column: ``UPOS 1..5 7; HEAD 1..12``."""
    columns: dict[str, list[str]] = {}
    for cell in cells:
        columns.setdefault(cell.column, []).append(cell.token_id)
    return "; ".join(f"{column.upper()} {' '.join(_ranges(ids))}" for column, ids in columns.items())


def _flag_lines(flags: Iterable[Flag]) -> list[str]:
    lines = []
    for flag in flags:
        where = f"word {flag.token_id}" if flag.token_id else "whole sentence"
        suggestion = f" Suggested: {flag.suggestion!r}." if flag.suggestion else ""
        lines.append(f"- flag {flag.id} ({where}, from {flag.raised_by}): {flag.reason}{suggestion}")
    return lines


def sentence_block(conllu: str) -> str:
    """CoNLL-U without the Conf attributes (the cells to work on list confidences) and trailing empty columns."""
    lines = []
    for line in conllu.splitlines():
        if line.startswith("#"):
            lines.append(line)
            continue
        cells = line.split("\t")
        cells[9] = misc_without_conf(cells[9])
        while len(cells) > 2 and cells[-1] == "_":
            cells.pop()
        lines.append("\t".join(cells))
    return "\n".join(lines)


def task_prompt(role: Role, target: str, items: list[TaskItem], evidence: str = "") -> str:
    """The user prompt for one role working on a batch of sentences."""
    lines = [f'Source "{target}". {len(items)} sentence(s).']
    for number, item in enumerate(items, start=1):
        lines.append(f"\n## s{number} = sentence {item.sent_id}")
        if item.conllu:
            lines.append(sentence_block(item.conllu))
        if role.is_scorer:
            lines.append(f"Cells to score: {_score_line(item.cells)}")
            continue
        for column in role.columns:
            cells = [cell for cell in item.cells if cell.column == column]
            if cells:
                lines.append(f"{column.upper()} cells to annotate or review: {_cell_line(cells)}")
        if item.flags:
            lines.append(f"Open flags for {'/'.join(column.upper() for column in role.columns)}:")
            lines += _flag_lines(item.flags)
    if evidence:
        lines += ["", evidence]
    return "\n".join(lines)
