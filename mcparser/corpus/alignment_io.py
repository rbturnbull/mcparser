"""Exchange formats for word alignments: plain text (Pharaoh/Ugarit) and YAML.

The stored format of an alignment set is JSON Lines (see :mod:`.alignment`).
These formats are for exchange with other tools and for reading by people.

* ``text``: one line per sentence pair, in the order of ``sent_ids``, holding
  space-separated links ``i-j`` (Pharaoh) that may end in ``S`` or ``P``
  (Ugarit; no suffix means sure). A blank line is a pair without links. The
  word numbers of such files usually start at 0 (``base=0``); the word ids of
  CoNLL-U start at 1, so ``base`` says which number the first word has.
* ``yaml``: a mapping from ``sent_id`` to the links as Ugarit text, using
  CoNLL-U word ids (``base`` does not apply)::

      B10K1V1: 1-1S 2-3S 3-4P
"""

from __future__ import annotations

from enum import StrEnum

import yaml

from .alignment import LINK_PATTERN, Link, format_ugarit, parse_ugarit
from .errors import EditError


class AlignmentFormat(StrEnum):
    TEXT = "text"
    YAML = "yaml"


def _shift(number: str, offset: int, what: str, minimum: int = 1) -> str:
    if not number.isdigit():
        raise EditError(f"{what}: word id {number!r} is not a plain word number.")
    shifted = int(number) + offset
    if shifted < minimum:
        raise EditError(f"{what}: word number {number} is below the first word (check the base).")
    return str(shifted)


def read_links(
    text: str, sent_ids: list[str], fmt: AlignmentFormat | str = AlignmentFormat.TEXT, base: int = 0
) -> dict[str, list[Link]]:
    """Parse ``text`` into ``{sent_id: links}`` with CoNLL-U word ids; pairs without links are left out."""
    fmt = AlignmentFormat(fmt)
    pairs: dict[str, list[Link]] = {}
    if fmt is AlignmentFormat.YAML:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise EditError("A YAML alignment should map sent_id to links, e.g. B10K1V1: 1-1S 2-3P.")
        for sent_id, value in data.items():
            links = parse_ugarit(str(value or ""))
            if links:
                pairs[str(sent_id)] = links
        return pairs
    lines = text.splitlines()
    if len(lines) != len(sent_ids):
        raise EditError(f"The file has {len(lines)} lines but there are {len(sent_ids)} sentence pairs.")
    for number, (sent_id, line) in enumerate(zip(sent_ids, lines), start=1):
        links = []
        for item in line.split():
            match = LINK_PATTERN.match(item)
            if match is None:
                raise EditError(f"Line {number}: invalid link {item!r}; expected I-J with an optional S or P.")
            source, target, kind = match.groups()
            where = f"Line {number}"
            links.append(
                Link(source=_shift(source, 1 - base, where), target=_shift(target, 1 - base, where), sure=kind != "P")
            )
        if links:
            pairs[sent_id] = links
    return pairs


def format_links(
    pairs: dict[str, list[Link]],
    sent_ids: list[str],
    fmt: AlignmentFormat | str = AlignmentFormat.TEXT,
    base: int = 0,
    suffix: bool = True,
) -> str:
    """The inverse of :func:`read_links`; ``suffix=False`` writes plain Pharaoh ``i-j`` (S/P dropped)."""
    fmt = AlignmentFormat(fmt)
    if fmt is AlignmentFormat.YAML:
        data = {sent_id: format_ugarit(pairs[sent_id]) for sent_id in sent_ids if pairs.get(sent_id)}
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    lines = []
    for sent_id in sent_ids:
        where = f"Pair {sent_id}"
        items = [
            f"{_shift(link.source, base - 1, where, base)}-{_shift(link.target, base - 1, where, base)}"
            + (("S" if link.sure else "P") if suffix else "")
            for link in pairs.get(sent_id, [])
        ]
        lines.append(" ".join(items))
    return "".join(f"{line}\n" for line in lines)
