import copy
import hashlib
import json

import pytest

from scripts.bounded_diagnostic import memory_preflight, validate_review
from scripts.bounded_diagnostic import select_diagnostic_versions,IDS


def fixture():
    spec={'cases':[{'id':'a','question':'q'}]}
    receipt={'annotation_sha256':hashlib.sha256(json.dumps(spec,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
             'release_approval':False,'decisions':[{'id':'a','question':'q','decision':'source_scoped_correct'}]}
    return spec,receipt


def test_review_binds_content_and_exact_case_decisions():
    spec,receipt=fixture()
    assert validate_review(spec,receipt)==receipt['annotation_sha256']
    changed=copy.deepcopy(spec)
    changed['cases'][0]['question']='different'
    with pytest.raises(ValueError):
        validate_review(changed,receipt)
    receipt['decisions']*=2
    with pytest.raises(ValueError):
        validate_review(spec,receipt)


@pytest.mark.parametrize('decision',['pending','needs_correction','out_of_scope'])
def test_unapproved_reference_cannot_run(decision):
    spec,receipt=fixture()
    receipt['decisions'][0]['decision']=decision
    with pytest.raises(ValueError):
        validate_review(spec,receipt)


def test_shared_512mb_server_requires_reserved_headroom():
    assert memory_preflight(272470016,536870912)[0] is False
    assert memory_preflight(220*1024**2,512*1024**2)[0] is True
    assert memory_preflight(225*1024**2,512*1024**2)[0] is False


def test_explicit_manifest_reads_one_version_per_document_without_discarding_old_versions():
    versions=[{'id':f'v{i}','document_id':i} for i in IDS]+[{'id':'old','document_id':IDS[0]}]
    manifest={str(i):f'v{i}' for i in IDS}
    assert set(select_diagnostic_versions(versions,manifest))==set(manifest.values())
    assert len(versions)==len(IDS)+1
    with pytest.raises(ValueError):select_diagnostic_versions(versions)
    manifest[str(IDS[0])]=manifest[str(IDS[1])]
    with pytest.raises(ValueError):select_diagnostic_versions(versions,manifest)


def test_manifest_cannot_attribute_a_version_to_another_document():
    versions=[{'id':f'v{i}','document_id':i} for i in IDS]
    manifest={str(i):f'v{i}' for i in IDS}
    a,b=map(str,IDS[:2]);manifest[a],manifest[b]=manifest[b],manifest[a]
    with pytest.raises(ValueError,match='mismatch'):select_diagnostic_versions(versions,manifest)
