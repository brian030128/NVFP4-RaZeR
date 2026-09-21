from campaign.evaluate_lmeval import SUITES


def test_gsm8k_decoding_is_frozen_greedy():
    spec = SUITES['gsm8k'][0]
    assert spec['num_fewshot'] == 5
    assert spec['gen_kwargs'] == {
        'do_sample': False,
        'until': ['Question:', '</s>', '<|im_end|>'],
        'max_gen_toks': 256,
    }


def test_pg19_dispatch_uses_frozen_book_builder(monkeypatch):
    from campaign import evaluate_ppl

    seen = {}
    def fake(tok, length, minimum_books):
        seen.update(length=length, minimum_books=minimum_books)
        return ['window'], {'source_books': minimum_books}
    monkeypatch.setattr(evaluate_ppl.D, 'pg19_frozen_books', fake)
    windows, meta = evaluate_ppl.windows_for(object(), 'pg19_8k', None)
    assert windows == ['window'] and meta['source_books'] == 20
    assert seen == {'length': 8192, 'minimum_books': 20}
