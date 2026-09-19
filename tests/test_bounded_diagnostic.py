import copy
import hashlib
import json

import pytest

from scripts.bounded_diagnostic import memory_preflight, validate_review


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
