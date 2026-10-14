"""Compact text renderings of MCP tool results for the agents.

Every tool result stays in the conversation and is re-read on each later model
request, so its size is paid many times over. JSON repeats every key for every
hit; these renderings keep the same values in short lines. They also drop what
an agent cannot use: echoes of the edits it just made, and "missing value"
warnings for columns other agents are responsible for.

The MCP server itself still returns structured JSON (for Claude Desktop and
other clients); only the agents' view is compacted, in
:class:`~mcparser.agents.toolset.RoleToolset`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..corpus.confidence import CONF_KEY
from ..corpus.conllu import Token
from .roles import Role

Result = Any
"""A tool result as decoded from MCP: dicts and lists of JSON values."""


def counts(items: list[dict[str, Any]], limit: int = 15) -> str:
    """``[{"value": "NOUN", "count": 3}, ...]`` as ``NOUN×3, ...``."""
    text = ", ".join(f"{item['value']}×{item['count']}" for item in items[:limit])
    if len(items) > limit:
        text += f", … (+{len(items) - limit} more)"
    return text or "none"


def misc(value: str) -> str:
    """MISC without the ``Conf`` attribute (confidence is shown separately)."""
    token = Token(id="1", misc=value)
    token.set_misc_value(CONF_KEY, None)
    return token.misc


def issues(items: list[dict[str, Any]], role: Role) -> list[str]:
    """Errors, plus warnings about the role's own columns."""
    lines = []
    for issue in items:
        if issue["severity"] == "warning" and issue.get("column") not in role.columns:
            continue
        lines.append(f"- {issue['severity']}: {issue['message']}")
    return ["Issues:", *lines] if lines else []


def flag_line(flag: dict[str, Any]) -> str:
    where = f"word {flag['token_id']}" if flag.get("token_id") else "sentence"
    suggestion = f" (suggests {flag['suggestion']!r})" if flag.get("suggestion") else ""
    return (
        f"- flag {flag['id']} [{flag['status']}] {flag['sent_id']} {where}, {flag['column']}, "
        f"from {flag['raised_by']}: {flag['reason']}{suggestion}"
    )


def groups(result: dict[str, Any]) -> str:
    if not result["groups"]:
        return "No matches."
    header = " ".join(result["groups"][0]["values"])
    shown = "; ".join(f"{' '.join(g['values'].values())} ×{g['count']}" for g in result["groups"])
    return f"{result['total']} matches. By {header}: {shown}"


def hit_line(hit: dict[str, Any]) -> str:
    head = f"{hit['head']}({hit['head_form'] or 'ROOT'})" if hit["head"] != "_" else "_"
    extra = misc(hit["misc"])
    parts = [
        f"- {hit['source']}:{hit['sent_id']}:{hit['token_id']}",
        hit["form"], hit["lemma"], hit["upos"], hit["xpos"], hit["feats"], f"head={head}", hit["deprel"],
    ]  # fmt: skip
    if extra != "_":
        parts.append(extra)
    if hit["confidence"] < 1:
        parts.append(f"conf={hit['confidence']:.2f}")
    return " ".join(parts) + f" | {hit['context']}"


def token_search(result: dict[str, Any], role: Role) -> str:
    lines = [groups(result)]
    if result["hits"]:
        lines.append("Examples (form lemma upos xpos feats head deprel misc | context):")
        lines += [hit_line(hit) for hit in result["hits"]]
    return "\n".join(lines)


def arc_search(result: dict[str, Any], role: Role) -> str:
    lines = [
        f"{result['total']} arcs. Relations: {counts(result['deprels'])}. "
        f"Head is to the: {counts(result['directions'])}."
    ]
    for hit in result["hits"]:
        dep, head = hit["dependent"], hit["head"]
        head_text = f"{head['form']}({head['upos']}, {head['id']})" if head else "ROOT"
        conf = f" conf={hit['confidence']:.2f}" if hit["confidence"] < 1 else ""
        lines.append(
            f"- {hit['source']}:{hit['sent_id']} {dep['form']}({dep['upos']}, {dep['id']}) "
            f"-{hit['deprel']}-> {head_text}{conf} | {hit['context']}"
        )
    return "\n".join(lines)


def multiword(result: dict[str, Any], role: Role) -> str:
    lines = [f"{result['surface']!r}: split {result['total']}×, unsplit {result['as_single_word']}×."]
    lines += [
        f"- {' + '.join(a['pieces'])} ({' '.join(a['upos'])}) ×{a['count']}, e.g. {', '.join(a['examples'])}"
        for a in result["analyses"]
    ]
    return "\n".join(lines)


def sentence_view(result: dict[str, Any], role: Role) -> str:
    lines = [result["conllu"], f"Mean confidence {result['confidence']:.2f}."]
    if result["open_flags"]:
        lines += ["Open flags:", *(flag_line(flag) for flag in result["open_flags"])]
    lines += issues(result["issues"], role)
    return "\n".join(lines)


def summaries(result: list[dict[str, Any]], role: Role) -> str:
    return "\n".join(f"- {s['sent_id']} (conf {s['confidence']:.2f}): {s['text']}" for s in result) or "None."


def sentence_hits(result: list[dict[str, Any]], role: Role) -> str:
    lines = [f"- {h['source']}:{h['sent_id']} (score {h['score']:.1f}, conf {h['confidence']:.2f}): {h['text']}" for h in result]
    return "\n".join(lines) or "No similar sentences."


def value_counts(result: list[dict[str, Any]], role: Role) -> str:
    return counts(result, limit=100)


def inventory(result: dict[str, list[dict[str, Any]]], role: Role) -> str:
    return "\n".join(f"{feature}: {counts(values, limit=30)}" for feature, values in result.items()) or "No features."


def validation(result: dict[str, Any], role: Role) -> str:
    lines = []
    for item in result["issues"]:
        issue = item["issue"]
        if issue["severity"] == "warning" and issue.get("column") not in role.columns:
            continue
        lines.append(f"- {item['sent_id']} {issue['severity']}: {issue['message']}")
    return "\n".join([f"{result['errors']} errors.", *lines])


def flags(result: dict[str, Any], role: Role) -> str:
    return "\n".join(flag_line(flag) for flag in result["flags"]) or "No flags."


def flag(result: dict[str, Any], role: Role) -> str:
    return f"Flag {result['id']} is {result['status']}."


def edit(result: dict[str, Any], role: Role) -> str:
    lines = [f"OK: {len(result['edits'])} change(s) saved."]
    if result.get("id_map") is not None:
        moved = {old: new for old, new in result["id_map"].items() if old != new}
        if moved:
            lines.append("Renumbered: " + ", ".join(f"{old}->{new or 'deleted'}" for old, new in moved.items()))
        lines += ["Sentence now:", result["conllu"]]
    lines += issues(result["issues"], role)
    return "\n".join(lines)


FORMATTERS: dict[str, Callable[[Any, Role], str]] = {
    "get_sentence": sentence_view,
    "get_context": summaries,
    "search_sentences": sentence_hits,
    "find_tokens": token_search,
    "analyse_form": token_search,
    "find_multiword": multiword,
    "find_arcs": arc_search,
    "allowed_values": value_counts,
    "feature_inventory": inventory,
    "validate": validation,
    "list_flags": flags,
    "flag_issue": flag,
    "resolve_flag": flag,
    **{name: edit for name in (
        "set_cell", "update_cells", "score_cells", "split_word", "merge_words",
        "delete_word", "insert_word", "add_multiword", "remove_multiword",
    )},
}  # fmt: skip


def compact_result(name: str, result: Result, role: Role) -> Result:
    """The compact text for a tool result, or the result unchanged if it has no formatter."""
    formatter = FORMATTERS.get(name)
    if formatter is None:
        return result
    try:
        return formatter(result, role)
    except (KeyError, TypeError, AttributeError, IndexError):
        return result  # an unexpected shape is better shown in full than lost
