from campaign.extension_p30_breadth_plans import document_hashes


def test_document_hashes_covers_both_domains():
    meta = {"math": {"documents": [{"document_sha256": "a"}, {"document_sha256": "b"}]},
            "code": {"documents": [{"document_sha256": "c"}]}}
    assert document_hashes(meta) == {"a", "b", "c"}
