import gzip
import json

from campaign.extension_downstream_analysis import gsm_rows, serializable_accuracy


def test_gsm_filters_remain_separate(tmp_path):
    path = tmp_path / "x.jsonl.gz"
    rows = [{"doc_id": 1, "doc_hash": "a", "filter": "strict-match", "exact_match": 0},
            {"doc_id": 1, "doc_hash": "a", "filter": "flexible-extract", "exact_match": 1}]
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    assert list(gsm_rows(path, "strict-match").values()) == [0.0]
    assert list(gsm_rows(path, "flexible-extract").values()) == [1.0]


def test_bootstrap_arrays_are_not_serialized():
    assert serializable_accuracy({"diff": 1, "boot": object()}) == {"diff": 1}
