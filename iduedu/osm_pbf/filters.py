"""Overpass way filters evaluated on OSM tags.

The street-network builders describe the ways they want with an Overpass QL filter such as
``["highway"]["area"!~"yes"]``. Reading a ``.pbf`` file, the same filter has to be applied to the
tags of every way, so this module turns the filter string into a predicate with the semantics
Overpass gives it:

- ``["k"]`` -- the key is present; ``[!"k"]`` -- the key is absent;
- ``["k"="v"]`` / ``["k"~"re"]`` -- the key is present and its value equals ``v`` / matches ``re``;
- ``["k"!="v"]`` / ``["k"!~"re"]`` -- true when the key is absent, or when its value differs from ``v``
  / does not match ``re``;
- regular expressions are unanchored (``re.search``) and case-sensitive unless the clause ends with
  ``,i`` as in ``["k"~"re",i]``.

Anything else -- key regexes (``[~"k"~"v"]``), ``(if: ...)`` evaluators, spatial or id filters -- is
rejected with :class:`UnsupportedFilterError` rather than silently ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping

__all__ = ["UnsupportedFilterError", "TagClause", "WayFilter", "parse_way_filter"]

_UNQUOTED = re.compile(r"[A-Za-z0-9_:.\-]+")
_ESCAPES = {'"': '"', "'": "'", "\\": "\\", "n": "\n", "t": "\t"}


class UnsupportedFilterError(ValueError):
    """Raised for Overpass filter syntax the ``.pbf`` reader cannot evaluate."""


@dataclass(frozen=True)
class TagClause:
    """One bracketed clause of an Overpass tag filter.

    ``op`` is one of ``"exists"``, ``"absent"``, ``"="``, ``"!="``, ``"~"``, ``"!~"``.
    """

    key: str
    op: str
    value: str | None = None
    ignore_case: bool = False

    def compile(self) -> Callable[[Mapping[str, str]], bool]:
        """Return a predicate on a tag mapping implementing this clause."""
        key, value = self.key, self.value
        if self.op == "exists":
            return lambda tags: key in tags
        if self.op == "absent":
            return lambda tags: key not in tags
        if self.op == "=":
            return lambda tags: tags.get(key) == value
        if self.op == "!=":
            return lambda tags: tags.get(key) != value
        try:
            pattern = re.compile(value, re.IGNORECASE if self.ignore_case else 0)
        except re.error as exc:
            raise UnsupportedFilterError(f"Invalid regular expression {value!r} for key {key!r}: {exc}") from exc
        search = pattern.search
        if self.op == "~":
            return lambda tags: key in tags and search(tags[key]) is not None
        if self.op == "!~":
            return lambda tags: key not in tags or search(tags[key]) is None
        raise UnsupportedFilterError(f"Unknown operator {self.op!r}")  # pragma: no cover

    @property
    def requires_key(self) -> bool:
        """True when the clause can only match a way that carries ``key``."""
        return self.op in {"exists", "=", "~"}


class WayFilter:
    """A parsed Overpass tag filter; calling it with a tag mapping tells whether the tags match."""

    def __init__(self, text: str, clauses: tuple[TagClause, ...]):
        self.text = text
        self.clauses = clauses
        self._predicates = tuple(c.compile() for c in clauses)

    def __call__(self, tags: Mapping[str, str]) -> bool:
        return all(p(tags) for p in self._predicates)

    @property
    def required_keys(self) -> tuple[str, ...]:
        """Keys every matching way must carry; useful to pre-filter ways before the full check."""
        seen = []
        for clause in self.clauses:
            if clause.requires_key and clause.key not in seen:
                seen.append(clause.key)
        return tuple(seen)

    def __repr__(self) -> str:
        return f"WayFilter({self.text!r})"


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def error(self, message: str) -> UnsupportedFilterError:
        return UnsupportedFilterError(f"{message} at position {self.pos} in Overpass filter {self.text!r}")

    def skip_ws(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def peek(self, token: str) -> bool:
        return self.text.startswith(token, self.pos)

    def expect(self, token: str):
        self.skip_ws()
        if not self.peek(token):
            raise self.error(f"Expected {token!r}")
        self.pos += len(token)

    def string(self) -> str:
        self.skip_ws()
        if self.pos >= len(self.text):
            raise self.error("Unexpected end of filter")
        quote = self.text[self.pos]
        if quote not in {'"', "'"}:
            match = _UNQUOTED.match(self.text, self.pos)
            if not match:
                raise self.error("Expected a quoted or plain key/value")
            self.pos = match.end()
            return match.group(0)
        self.pos += 1
        out = []
        while True:
            if self.pos >= len(self.text):
                raise self.error("Unterminated string")
            char = self.text[self.pos]
            if char == quote:
                self.pos += 1
                return "".join(out)
            if char == "\\" and self.pos + 1 < len(self.text):
                nxt = self.text[self.pos + 1]
                if nxt in _ESCAPES:
                    out.append(_ESCAPES[nxt])
                    self.pos += 2
                    continue
                if nxt == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", self.text[self.pos + 2 : self.pos + 6]):
                    out.append(chr(int(self.text[self.pos + 2 : self.pos + 6], 16)))
                    self.pos += 6
                    continue
                # Any other escape is kept as written, so that regular expressions such as "\d" survive.
                out.append(char + nxt)
                self.pos += 2
                continue
            out.append(char)
            self.pos += 1

    def clause(self) -> TagClause:
        self.expect("[")
        self.skip_ws()
        if self.peek("~"):
            raise self.error('Key regular expressions ([~"k"~"v"]) are not supported')
        if self.peek("!"):
            self.pos += 1
            key = self.string()
            self.expect("]")
            return TagClause(key, "absent")
        key = self.string()
        self.skip_ws()
        if self.peek("]"):
            self.pos += 1
            return TagClause(key, "exists")
        for op in ("!=", "!~", "=", "~"):
            if self.peek(op):
                self.pos += len(op)
                break
        else:
            raise self.error("Expected one of =, !=, ~, !~ or ]")
        value = self.string()
        self.skip_ws()
        ignore_case = False
        if self.peek(","):
            self.pos += 1
            self.skip_ws()
            if not self.peek("i") or op in {"=", "!="}:
                raise self.error("Only the ',i' flag of a regular expression clause is supported")
            self.pos += 1
            ignore_case = True
        self.expect("]")
        return TagClause(key, op, value, ignore_case)

    def parse(self) -> tuple[TagClause, ...]:
        clauses = []
        self.skip_ws()
        while self.pos < len(self.text):
            if not self.peek("["):
                raise self.error("Only tag clauses in square brackets are supported")
            clauses.append(self.clause())
            self.skip_ws()
        return tuple(clauses)


def parse_way_filter(text: str) -> WayFilter:
    """Parse an Overpass tag filter such as ``["highway"]["access"!~"private"]``.

    Raises:
        UnsupportedFilterError: For syntax outside the supported subset, or an invalid regex.
    """
    if not isinstance(text, str):
        raise UnsupportedFilterError(f"Overpass filter must be a string, got {type(text).__name__}")
    return WayFilter(text, _Parser(text).parse())
