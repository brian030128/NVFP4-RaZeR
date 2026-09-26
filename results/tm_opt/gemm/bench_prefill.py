"""#3 (PROTOCOL_ITEMS.md): full-model prefill latency on the SM120 deployment kernel, one policy per process.

A policy is an exported artifact (sm120/eval/export_artifact.py) and a kernel configuration; every scoped Linear runs
as NativeLinear (per-token activation quantization + GEMM with the fused epilogue). Prefill is sm120/bench/model.py's:
one forward over [batch, prompt] tokens with the KV cache written, median of 5 after 2 warm-ups, CUDA events. Idle
GPU required.

python results/tm_opt/gemm/bench_prefill.py --model llama8b --artifact DIR --kernel n8k64_wB --label NAME --out JSON
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

SM120 = Path('/home/dev/n16k64_campaign/sm120_bench/sm120')
sys.path.insert(0, str(SM120))
sys.path.insert(0, str(SM120 / 'bench'))
import common as B  # noqa: E402  (sm120/bench/common.py)
from mixfp4_sm120 import model as NM  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--kernel', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--prefill', default='1x512,1x2048,4x2048')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    B.require_idle()
    C = load('sm120_eval_common', SM120 / 'eval' / 'common.py')
    BM = load('sm120_bench_model', SM120 / 'bench' / 'model.py')
    res = dict(gpu=B.gpu_info(), model=args.model, label=args.label, artifact=args.artifact, kernel=args.kernel, prefill={})
    model = C.load_model(args.model)
    rep = NM.install(model, args.artifact, kernel=args.kernel, loader=C.MODELS[args.model]['loader'])
    res['install'] = rep.as_dict()
    res['coverage'] = {k: v for k, v in NM.coverage(model).items() if k != 'remaining_bf16_linears'}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    res['weights_gib'] = torch.cuda.memory_allocated() / 2 ** 30
    for spec in args.prefill.split(','):
        b, p = (int(v) for v in spec.split('x'))
        res['prefill'][spec] = BM.prefill(model, b, p)
        print(args.label, spec, json.dumps(res['prefill'][spec]), flush=True)
    res['peak_prefill_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    B.write(args.out, res)


if __name__ == '__main__':
    main()
