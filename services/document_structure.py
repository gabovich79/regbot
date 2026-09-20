"""Source-preserving boundaries for explicit contents lists and DOCX tables."""
import re

TABLE = re.compile(r'(?m)^\[(תחילת|סוף) טבלה במקור: (\d+)\]$')
CONTENTS = re.compile(r'(?mi)^[ \t]*(?:תוכן(?:\s+עניינים)?|table of contents)\s*:?\s*$')


def boundaries(text, heading_pattern):
    """Return offsets and navigation labels without deleting any source text.

    A first table row is a locator, not an inferred legal title. Numbered
    clauses inside a table stay in the complete table for context expansion.
    Only explicit contents markers followed by at least three heading entries
    suppress those entries as legal section headings.
    """
    headings = {m.start(): m.group().strip().lstrip("'׳") for m in heading_pattern.finditer(text)}
    spans, additions, stack = [], {}, []
    for match in TABLE.finditer(text):
        direction, ident = match.groups()
        if direction == 'תחילת':
            stack.append((ident, match))
        elif stack and stack[-1][0] == ident:
            _, begin = stack.pop()
            if not stack:
                spans.append((begin.start(), match.end()))
                first = next((line.strip() for line in text[begin.end():match.start()].splitlines() if line.strip()), '')
                first = re.sub(r'\[(?:פריסת תא|מיזוג אנכי) במקור:[^\]]*\]', '', first).strip()
                label = f'טבלה {ident} במקור'
                if first and len(first) <= 200:
                    label += f' — שורה ראשונה: {first}'
                additions[begin.start()] = label
                additions[match.end()] = 'מבוא / המשך'
    for marker in CONTENTS.finditer(text):
        position, entries, end = marker.end(), 0, marker.end()
        for line in text[position:].splitlines(keepends=True):
            stripped = line.strip()
            if not stripped:
                position += len(line)
                continue
            if heading_pattern.fullmatch(stripped) is None:
                break
            entries += 1
            position += len(line)
            end = position
        if entries >= 3:
            spans.append((marker.start(), end))
            additions[marker.start()] = 'תוכן עניינים (ניווט בלבד)'
            additions[end] = 'מבוא / המשך'
    headings = {offset: label for offset, label in headings.items()
                if not any(start <= offset < end for start, end in spans)}
    headings.update(additions)
    return headings
