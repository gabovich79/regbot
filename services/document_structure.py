"""Source-preserving boundaries for explicit contents lists and DOCX tables."""
import re

TABLE = re.compile(r'(?m)^\[(תחילת|סוף) טבלה במקור: (\d+)\]$')
CONTENTS = re.compile(r'(?mi)^[ \t]*(?:תוכן(?:\s+עניינים)?|table of contents)\s*:?\s*$')


def caption_words(line):
    without_years = re.sub(r'תש[א-ת]{0,2}["\'״׳][א-ת]{1,2}|תש[א-ת]{1,3}(?=\d)', '', line)
    without_years = re.sub(r'ת["״]ט', '', without_years)
    return [word for word in re.findall(r'[א-ת]{2,}', without_years)
            if word not in {'מס', 'תיקון', 'הוראת', 'שעה'}]


def statutory_caption(line):
    """Recognize a source caption alongside amendment metadata, not metadata alone.

    Caller must independently verify bold typography and exact text alignment.
    This detects navigation boundaries, not section numbers or legal scope.
    """
    if '(תיקון' not in line:
        return False
    # PDF visual ordering can place amendment metadata before the caption.
    # Remove complete Hebrew years before tokenizing: splitting תשס"ה would
    # otherwise leave a spurious word and turn metadata into a heading.
    return bool(caption_words(line))


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
    # Explicit annex part labels are safe navigation boundaries. Do not split
    # a table at text which happens to name a part inside a cell.
    for match in re.finditer(r'(?m)^[ \t]*חלק\s+[0-9א-ת]+[\'׳]?[ \t]*$',text):
        if not any(a<=match.start()<b for a,b in spans):
            headings[match.start()]=match.group().strip()
    return headings
