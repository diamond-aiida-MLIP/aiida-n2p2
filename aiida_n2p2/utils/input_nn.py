"""Parse and render n2p2 ``input.nn`` configuration files."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

KEYWORD_PATTERN = re.compile(
    r'^(?P<commented>#)?(?P<body>[A-Za-z_][A-Za-z0-9_]*)'
    r'(?:\s+(?P<value>.*?))?'
    r'(?:\s+#(?P<inline>.*))?$'
)


@dataclass
class InputNnLine:
    """One logical line of an ``input.nn`` file."""

    kind: str  # 'keyword' or 'raw'
    raw: str
    commented: bool = False
    keyword: str | None = None
    value: str | None = None
    inline_comment: str | None = None

    def to_dict(self) -> dict:
        return {
            'kind': self.kind,
            'raw': self.raw,
            'commented': self.commented,
            'keyword': self.keyword,
            'value': self.value,
            'inline_comment': self.inline_comment,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InputNnLine:
        return cls(**data)


def parse_input_nn(source: str | Path | TextIO | BinaryIO) -> list[InputNnLine]:
    """Parse an ``input.nn`` file into structured lines."""
    if isinstance(source, Path):
        text = source.read_text(encoding='utf-8')
    elif isinstance(source, str):
        text = source
    elif hasattr(source, 'read'):
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode('utf-8')
    else:
        raise TypeError(f'Unsupported source type: {type(source)}')

    lines: list[InputNnLine] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith('#') and not _looks_like_keyword(raw_line):
            lines.append(InputNnLine(kind='raw', raw=raw_line))
            continue

        match = KEYWORD_PATTERN.match(stripped.lstrip('#').strip() if stripped.startswith('#') else stripped)
        if match and _looks_like_keyword(raw_line):
            commented = raw_line.lstrip().startswith('#')
            body = match.group('body')
            value = match.group('value')
            if value is not None:
                value = value.strip()
                if '#' in value and match.group('inline') is None:
                    value_part, _, inline = value.partition('#')
                    value = value_part.strip() or None
                    inline_comment = inline.strip() or None
                else:
                    inline_comment = (
                        match.group('inline').strip() if match.group('inline') else None
                    )
            else:
                inline_comment = (
                    match.group('inline').strip() if match.group('inline') else None
                )
            lines.append(
                InputNnLine(
                    kind='keyword',
                    raw=raw_line,
                    commented=commented,
                    keyword=body,
                    value=value,
                    inline_comment=inline_comment,
                )
            )
        else:
            lines.append(InputNnLine(kind='raw', raw=raw_line))
    return lines


def _looks_like_keyword(raw_line: str) -> bool:
    candidate = raw_line.strip().lstrip('#').strip()
    match = KEYWORD_PATTERN.match(candidate)
    if not match:
        return False
    name = match.group('body')
    value = (match.group('value') or '').strip()
    if value:
        return True
    return '_' in name


def lines_to_dicts(lines: list[InputNnLine]) -> list[dict]:
    return [line.to_dict() for line in lines]


def dicts_to_lines(data: list[dict]) -> list[InputNnLine]:
    return [InputNnLine.from_dict(item) for item in data]


def keyword_states(lines: list[InputNnLine]) -> dict[str, dict]:
    """Return keyword states for provenance snapshots."""
    states: dict[str, dict] = {}
    for line in lines:
        if line.kind != 'keyword' or not line.keyword:
            continue
        if line.commented and line.value is None:
            states[line.keyword] = {'state': 'disabled'}
        elif line.value is None:
            states[line.keyword] = {'state': 'active_flag'}
        elif line.commented:
            states[line.keyword] = {'state': 'disabled', 'value': line.value}
        else:
            states[line.keyword] = {'state': 'active', 'value': line.value}
    return states


def set_keyword(lines: list[InputNnLine], keyword: str, value: str) -> list[InputNnLine]:
    updated = deepcopy(lines)
    for line in updated:
        if line.kind == 'keyword' and line.keyword == keyword:
            line.commented = False
            line.value = value
            return updated
    raise KeyError(f"Keyword '{keyword}' not found in input.nn template.")
    return updated  # pragma: no cover


def enable_keyword(lines: list[InputNnLine], keyword: str) -> list[InputNnLine]:
    updated = deepcopy(lines)
    for line in updated:
        if line.kind == 'keyword' and line.keyword == keyword:
            line.commented = False
            return updated
    raise KeyError(f"Keyword '{keyword}' not found in input.nn template.")
    return updated  # pragma: no cover


def disable_keyword(lines: list[InputNnLine], keyword: str) -> list[InputNnLine]:
    updated = deepcopy(lines)
    for line in updated:
        if line.kind == 'keyword' and line.keyword == keyword:
            line.commented = True
            return updated
    raise KeyError(f"Keyword '{keyword}' not found in input.nn template.")
    return updated  # pragma: no cover


def render_input_nn(lines: list[InputNnLine]) -> str:
    """Render structured lines back to an ``input.nn`` file."""
    rendered: list[str] = []
    for line in lines:
        if line.kind == 'raw':
            rendered.append(line.raw)
            continue
        if not line.keyword:
            rendered.append(line.raw)
            continue

        body = line.keyword
        if line.value is not None:
            body = f'{body} {line.value}'
        if line.inline_comment:
            body = f'{body} # {line.inline_comment}'
        if line.commented:
            body = f'#{body}'
        rendered.append(body)
    return '\n'.join(rendered) + '\n'
