"""Verification 3 (PROTOCOL.md): NativeLinear with the RTX PRO 6000 table and with the fallback rule give bitwise-identical
logits, on Llama-3.1-8B's TM-OPT+TC 16x64 artifact ('mixed' set) and FourOverSix ('stock' set).

Inputs: the first WINDOWS WikiText-2 protocol windows (sm120/eval/common.windows), each at prefix lengths chosen to
cover buckets where the table departs from the fallback (Llama's shapes depart at T = 32..1024) and where it does not.
Per forward the logits' bf16 bytes are hashed. The per-width GEMM call counts of both runs are recorded, to show that
the table run used other widths.

    python results/tile_table/check_logits.py ART_DIR   -> check_logits.json
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / 'sm120'))
from mixfp4_sm120 import model as NM  # noqa: E402
from mixfp4_sm120.linear import clear_quant_cache  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

WINDOWS = 3
LENGTHS = (1, 32, 64, 128, 256, 512, 1024, 2048)


def load_common():
    spec = importlib.util.spec_from_file_location('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def run(model, wins):
    out = {}
    for i, w in enumerate(wins):
        for n in LENGTHS:
            logits = model(input_ids=w[:, :n].cuda(), use_cache=False).logits
            out[f'{i}:{n}'] = hashlib.sha256(logits.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()
    return out


def main():
    art = Path(sys.argv[1])
    C = load_common()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = C.load_model('llama8b')
    tok = C.tokenizer('llama8b') if hasattr(C, 'tokenizer') else None
    if tok is None:
        from transformers import AutoTokenizer
        spec = C.MODELS['llama8b']
        tok = AutoTokenizer.from_pretrained(spec['model_id'], revision=spec['revision'])
    wins = C.windows(tok, 'llama8b', 'wiki')[0][:WINDOWS]
    result = dict(windows=WINDOWS, lengths=list(LENGTHS), policies={})
    for label, artifact, family in (('tc-16x64', art / 'llama8b_tc_16x64', 'mixed'), ('FourOverSix', art / 'llama8b_fo6', 'stock')):
        entry = result['policies'][label] = dict(artifact=str(artifact), family=family)
        hashes = {}
        for mode in ('table', 'fallback'):
            ks = KernelSet(family) if mode == 'table' else KernelSet(family, table={})
            assert (ks.table_source is not None and ks.table_source != 'dict') if mode == 'table' else not ks.table
            clear_quant_cache()
            NM.install(model, str(artifact), kernel=ks, loader=C.MODELS['llama8b']['loader'])
            hashes[mode] = run(model, wins)
            entry[mode] = dict(table_source=ks.table_source, calls_by_width=dict(sorted(ks.stats.items())))
        entry['forwards'] = len(hashes['table'])
        entry['identical'] = sum(hashes['table'][k] == hashes['fallback'][k] for k in hashes['table'])
        entry['all_identical'] = entry['identical'] == entry['forwards']
        print(label, json.dumps({k: v for k, v in entry.items() if k not in ('artifact',)}), flush=True)
    result['all_identical'] = all(e['all_identical'] for e in result['policies'].values())
    (HERE / 'check_logits.json').write_text(json.dumps(result, indent=1) + '\n')
    sys.exit(0 if result['all_identical'] else 1)


if __name__ == '__main__':
    main()
