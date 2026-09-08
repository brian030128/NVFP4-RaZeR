"""Check held-out sampling excludes calibration and keeps paired fixed windows."""
import hashlib
from unittest.mock import patch
from types import SimpleNamespace
import torch
from run_c4_frozen import heldout_data


def main():
    class Tokenizer:
        def __call__(self, text, return_tensors):
            length = 511 if text == 'short' else 600
            return SimpleNamespace(input_ids=torch.arange(length).reshape(1, -1))

    excluded = {hashlib.sha256(b'calibration').hexdigest()}
    rows = [{'text': t} for t in ['calibration', 'short', 'document0', 'document0'] +
            [f'document{i}' for i in range(1, 256)]]
    with patch('run_c4_frozen.load_dataset', side_effect=lambda *a, **kw: iter(rows)):
        a, meta = heldout_data(Tokenizer(), excluded)
        b, repeated = heldout_data(Tokenizer(), excluded)
    assert meta == repeated
    assert len(a) == 256 and all(x.shape == (1, 512) for x in a)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    hashes = {d['document_sha256'] for d in meta['documents']}
    assert len(hashes) == 256 and not hashes & excluded
    assert hashlib.sha256(b'short').hexdigest() not in hashes
    with patch('run_c4_frozen.load_dataset', return_value=iter(rows[:2])):
        try:
            heldout_data(Tokenizer(), excluded)
        except AssertionError as exc:
            assert 'Insufficient' in str(exc)
        else:
            raise AssertionError('An incomplete evaluation set must fail')
    print('C4 exclusion, duplicate handling, length filtering, seeded replay and incomplete-set checks passed')


if __name__ == '__main__':
    main()
