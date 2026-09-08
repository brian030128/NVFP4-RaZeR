"""Check fixed WikiText concatenation, window coverage and overlap rejection."""
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
import torch
from run_wiki_frozen import wiki_data


def main():
    class Tokenizer:
        def __call__(self, text, return_tensors):
            assert text == 'first\n\n\n\nsecond'
            return SimpleNamespace(input_ids=torch.arange(1100).reshape(1, -1))
    with patch('run_wiki_frozen.load_dataset', return_value={'text': ['first', '', 'second']}):
        batches, meta = wiki_data(Tokenizer(), set())
        assert len(batches) == 2 and meta['omitted_tail_tokens'] == 76
        assert meta['offsets'] == [0, 512] and meta['total_tokens'] == 1100
        assert torch.equal(torch.cat(batches, dim=1), torch.arange(1024).reshape(1, -1))
        try:
            wiki_data(Tokenizer(), {hashlib.sha256(b'first').hexdigest()})
        except AssertionError as exc:
            assert 'calibration' in str(exc)
        else:
            raise AssertionError('Overlap must reject evaluation instead of dropping rows')
    print('WikiText concatenation, full-window coverage, tail accounting and exact-overlap checks passed')


if __name__ == '__main__': main()
