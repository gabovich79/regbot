"""Pack complete contiguous source sections, with explicit dependency gaps.

No benchmark questions, answers or inferred legal rules. A section is the
contiguous parent recorded at ingestion; a repeated heading is not identity.
"""
import json
import re

from services.knowledge import ENC

GOVERNING = re.compile(r'(?:^|[\s./])(?:תחולה|תחילה|הגדרות|הוראות מעבר)(?:\s|$|[.:])')
NUMBER = re.compile(r'^\s*(?:סעיף\s+)?(\d{1,3}[א-ת]?)\s*[.)]?(?:\s|$)')
REFERENCE = re.compile(r'(?<![א-ת])סעיף\s+(\d{1,3}[א-ת]?)(?![\dא-ת])')


def pack_sections(ranked, chunks, public_evidence, budget=24000):
    groups, membership, by_version = {}, {}, {}
    ordered = sorted(chunks, key=lambda c: (c['version_id'], c['ordinal']))
    previous, group = None, None
    for chunk in ordered:
        identity = (chunk['version_id'], chunk['section'], chunk['section_text'])
        if previous is None or identity != previous[0] or chunk['ordinal'] != previous[1]+1:
            group = chunk['id']
            groups[group] = []
            by_version.setdefault(chunk['version_id'], []).append(group)
        groups[group].append(chunk)
        membership[chunk['id']] = group
        previous = (identity, chunk['ordinal'])
    seeds = list(dict.fromkeys(membership[c['id']] for c in ranked))
    numbered = {}
    for key, members in groups.items():
        match = NUMBER.match(members[0]['section'])
        if match:
            numbered.setdefault((members[0]['version_id'], match[1]), []).append(key)
    selected, selected_groups, decisions, unresolved, partial = [], set(), [], [], []
    for seed in seeds:
        members = groups[seed]
        version = members[0]['version_id']
        dependencies = [key for key in by_version[version]
                        if GOVERNING.search(groups[key][0]['section'])]
        # Explicit local references only. Cross-document references and ambiguous
        # targets are reported, never silently assigned to a same-numbered rule.
        for match in REFERENCE.finditer(members[0]['section_text']):
            following = members[0]['section_text'][match.end():match.end()+90]
            if re.search(r'\b(?:לחוק|לתקנות|לפקודת|לחוזר(?!\s+זה\b))\b', following[:40]):
                unresolved.append({'from_id':seed,'reference':match.group(0),'reason':'external_reference_requires_resolution'})
                continue
            targets = numbered.get((version, match[1]), [])
            if len(targets) == 1:
                dependencies.extend(targets)
            else:
                unresolved.append({'from_id':seed,'reference':match.group(0),'reason':'missing_or_ambiguous_section'})
        bundle = list(dict.fromkeys([seed]+dependencies))
        additions = [key for key in bundle if key not in selected_groups]
        if not additions:
            continue
        items = [public_evidence(c) for key in additions for c in groups[key]]
        if len(ENC.encode(json.dumps(items, ensure_ascii=False))) > budget:
            # A large dependency set does not mean the selected source section
            # itself is oversized. Keep that section whole, and report missing
            # dependencies explicitly instead of truncating its qualifications.
            existing = {e['id'] for e in selected}
            complete_seed = [public_evidence(c) for c in members if c['id'] not in existing]
            if len(ENC.encode(json.dumps(selected+complete_seed, ensure_ascii=False))) <= budget:
                selected.extend(complete_seed)
                selected_groups.add(seed)
                missing_dependencies = [key for key in additions if key != seed]
                unresolved.extend({'from_id':seed, 'reference':key,
                                   'reason':'dependency_bundle_exceeds_context_budget'}
                                  for key in missing_dependencies)
                decisions.append({'seed':seed, 'included':True,
                                  'reason':'complete_section_with_unresolved_dependencies',
                                  'chunk_ids':[e['id'] for e in complete_seed],
                                  'omitted_dependencies':missing_dependencies})
                continue
            # Legacy indexes can label a whole chapter as one parent. Preserve
            # useful evidence but NEVER describe this fallback as a full section.
            hits = [c for c in ranked if membership[c['id']] == seed]
            for hit in hits:
                positions = {hit['ordinal']+offset for offset in (-1,0,1)}
                fallback = [public_evidence(c) for c in members if c['ordinal'] in positions]
                existing = {e['id'] for e in selected}
                fallback = [e for e in fallback if e['id'] not in existing]
                if fallback and len(ENC.encode(json.dumps(selected+fallback,ensure_ascii=False))) <= budget:
                    selected.extend(fallback)
                    if seed not in partial:
                        partial.append(seed)
                    decisions.append({'seed':seed,'included':True,'reason':'oversized_legacy_parent_partial_context',
                                      'chunk_ids':[e['id'] for e in fallback]})
            continue
        # Include serialized metadata and list separators in the actual budget.
        size = len(ENC.encode(json.dumps(selected+items, ensure_ascii=False)))
        included = size <= budget
        decisions.append({'seed':seed,'section':members[0]['section'],
                          'groups':additions,'chunk_ids':[e['id'] for e in items],
                          'included':included,'resulting_tokens':size,
                          'reason':'complete_section_bundle' if included else 'complete_bundle_exceeds_budget'})
        if included:
            selected.extend(items)
            selected_groups.update(additions)
    return selected, {'strategy':'complete_sections_v1','context_tokens':len(ENC.encode(json.dumps(selected,ensure_ascii=False))),
                      'context_selection':decisions,'unresolved_section_references':unresolved,
                      'omitted_section_seeds':[s for s in seeds if s not in selected_groups],
                      'partial_section_seeds':partial,
                      'complete_section_count':len(selected_groups)}
