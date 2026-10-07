"""Conservative Word list labels; unsupported numbering remains visible."""
import re

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
UNRESOLVED = '[מספור במקור לא שוחזר]'


def val(node, name, default=None):
    child = node.find(W + name) if node is not None else None
    return child.get(W + 'val', default) if child is not None else default


class Numbering:
    def __init__(self, doc):
        try:
            root = doc.part.numbering_part.element
        except KeyError:
            root = []
        self.nums = {n.get(W+'numId'): n for n in root if n.tag == W+'num'}
        self.abstracts = {n.get(W+'abstractNumId'): n for n in root if n.tag == W+'abstractNum'}
        self.styles = {s.style_id: s.element for s in doc.styles}
        self.counts = {}

    def properties(self, paragraph):
        props = paragraph.find(W+'pPr')
        result, seen = {}, set()
        while props is not None:
            num = props.find(W+'numPr')
            for key in ('numId', 'ilvl'):
                value = val(num, key)
                if value is not None:
                    result.setdefault(key, value)
            style = val(props, 'pStyle')
            if not style or style in seen:
                break
            seen.add(style)
            element = self.styles.get(style)
            if element is None:
                break
            # Merge nearest style first, then its basedOn ancestors.
            while element is not None:
                num = element.find(W+'pPr/'+W+'numPr')
                for key in ('numId', 'ilvl'):
                    value = val(num, key)
                    if value is not None:
                        result.setdefault(key, value)
                parent = val(element, 'basedOn')
                if not parent or parent in seen:
                    break
                seen.add(parent)
                element = self.styles.get(parent)
            break
        return result

    def level(self, num, level):
        abstract = self.abstracts.get(val(num, 'abstractNumId'))
        if abstract is None or abstract.find(W+'numStyleLink') is not None:
            raise ValueError('Unsupported numbering link')
        definition = abstract.find(W+f'lvl[@{W}ilvl="{level}"]')
        override = num.find(W+f'lvlOverride[@{W}ilvl="{level}"]')
        if override is not None and override.find(W+'lvl') is not None:
            definition = override.find(W+'lvl')
        if definition is None:
            raise ValueError('Missing level')
        start = int(val(override, 'startOverride', val(definition, 'start', '1')))
        return definition, start

    @staticmethod
    def format(number, kind):
        if kind == 'decimal':
            return str(number)
        # These values are shared by Hebrew alphabetical and numeric schemes.
        # Higher values require a separately verified formatter.
        if kind in ('hebrew1', 'hebrew2') and 1 <= number <= 10:
            return 'אבגדהוזחטי'[number-1]
        if kind in ('lowerLetter', 'upperLetter') and 1 <= number <= 26:
            return chr((97 if kind == 'lowerLetter' else 65) + number-1)
        raise ValueError('Unsupported number format')

    def prefix(self, paragraph):
        props = self.properties(paragraph)
        ident = props.get('numId')
        if ident in (None, '0'):
            return ''
        try:
            num = self.nums[ident]
            level = int(props.get('ilvl', '0'))
            definition, start = self.level(num, level)
            if definition.find(W+'isLgl') is not None or definition.find(W+'lvlPicBulletId') is not None:
                raise ValueError('Unsupported legal or picture numbering')
            key = (ident, level)
            self.counts[key] = self.counts.get(key, start-1) + 1
            for other in list(self.counts):
                if other[0] == ident and other[1] > level:
                    child, _ = self.level(num, other[1])
                    restart = int(val(child, 'lvlRestart', str(other[1])))
                    if restart and level < restart:
                        del self.counts[other]
            pattern = val(definition, 'lvlText')
            if pattern is None:
                raise ValueError('Missing label')
            if val(definition, 'numFmt') == 'bullet':
                if any(0xE000 <= ord(c) <= 0xF8FF for c in pattern):
                    raise ValueError('Font-dependent bullet')
                return pattern + ' '

            def replace(match):
                index = int(match.group(1))-1
                target, initial = self.level(num, index)
                return self.format(self.counts.get((ident,index), initial), val(target, 'numFmt'))
            return re.sub(r'%([1-9])', replace, pattern) + ' '
        except (KeyError, ValueError, TypeError):
            return UNRESOLVED + ' '
