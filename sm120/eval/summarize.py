#!/usr/bin/env python3
"""Markdown tables from sm120/results/*.json (used to write sm120/RESULTS.md).

    python sm120/eval/summarize.py ppl sm120/results/ppl/qwen4b.json [...]
    python sm120/eval/summarize.py lmeval sm120/results/lmeval/qwen4b.json
    python sm120/eval/summarize.py kernel sm120/results/bench/kernel_rtx5090.json
    python sm120/eval/summarize.py linear sm120/results/bench/linear_rtx5090.json
    python sm120/eval/summarize.py model sm120/results/bench/model_*.json
"""
import json
import math
import statistics
import sys
from pathlib import Path

REF = json.loads((Path(__file__).resolve().parent / 'reference' / 'repro_ppl.json').read_text())
POL_REF = {'bf16': 'bf16', 'fake_nvfp4': 'nvfp4', 'fake_four_over_six': 'four_over_six', 'fake_n16_k3': 'n16_k3',
           'fake_n8_k3': 'n8_k3', 'native_nvfp4': 'nvfp4', 'native_four_over_six': 'four_over_six',
           'native_n16_k3': 'n16_k3', 'native_n8_k3': 'n8_k3'}


def paired(a, b, tokens_per_window):
    d = [(x - y) / tokens_per_window for x, y in zip(a, b)]
    m = statistics.fmean(d)
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float('nan')
    return m, 2 * se, sum(x > 0 for x in d), sum(x < 0 for x in d)


def ppl(files):
    print('| Model | Policy | Wiki (this run) | Wiki (repro, same path) | C4 (this run) | C4 (repro, same path) |')
    print('|---|---|---:|---:|---:|---:|')
    comps = []
    for f in files:
        r = json.loads(Path(f).read_text())
        m = r['model']
        for pol, v in r['results'].items():
            kind = 'native' if pol.startswith('native') else 'fake'
            ref = REF[kind if pol != 'bf16' else 'fake'].get(m, {}).get(POL_REF.get(pol, pol), {})
            cells = []
            for d in ('wiki', 'c4'):
                cells.append(f'{v[d]["ppl"]:.4f}' if d in v else '')
                cells.append(f'{ref[d]["ppl"]:.4f}' if d in ref else '')
            print(f'| {m} | {pol} | ' + ' | '.join(cells) + ' |')
        for pol, c in r.get('native_vs_fake', {}).items():
            for d, x in c.items():
                comps.append((m, pol, x['partner'], d, x))
    print()
    print('| Model | Native | vs fake | Corpus | ΔNLL/token (native−fake) | 2 SE | +/− windows | PPL ratio |')
    print('|---|---|---|---|---:|---:|---:|---:|')
    for m, pol, partner, d, x in comps:
        print(f'| {m} | {pol} | {partner} | {d} | {x["delta_mean_nll"]:+.5f} | {x["two_se"]:.5f} | '
              f'{x["positive"]}/{x["negative"]} | {x["ppl_ratio"]:.5f} |')


def lmeval(files):
    for f in files:
        r = json.loads(Path(f).read_text())
        tasks = r['tasks']
        print(f'**{r["model"]}** (lm-eval {r["lm_eval"]}, 0-shot, primary metric per task)\n')
        print('| Policy | ' + ' | '.join(tasks) + ' | mean |')
        print('|---|' + '---:|' * (len(tasks) + 1))
        for pol, v in r['results'].items():
            vals = [v['primary'][t]['value'] for t in tasks]
            print(f'| {pol} | ' + ' | '.join(f'{x * 100:.2f}' for x in vals) + f' | {statistics.fmean(vals) * 100:.2f} |')
        print()
        for pol, c in r.get('native_vs_fake', {}).items():
            print(f'{pol} − {next(iter(c.values()))["partner"]} (paired, per example): ' + ', '.join(
                f'{t} {x["delta"] * 100:+.2f} ± {x["two_se"] * 100:.2f} pp ({x["disagreements"]} of {x["n"]} differ)' for t, x in c.items()))
        print()


def kernel(files):
    for f in files:
        r = json.loads(Path(f).read_text())
        print(f'GPU: {r["gpu"].get("device")}, power limit {r["gpu"].get("power.limit")}, driver {r["gpu"].get("driver_version")}\n')
        rows = r['rows']
        key = lambda x: (x['model'], x['proj'], x['tokens'])  # noqa: E731
        groups = {}
        for x in rows:
            groups.setdefault(key(x), []).append(x)
        print('| model | proj (out×in) | T | stock NVFP4 µs | no-dispatch | N16K64 E2M1 tags | N16K64 selector map | N16K64 random50 | 8×1 selector | N8K64 selector | BF16 cuBLAS µs |')
        print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for k, g in groups.items():
            def get(cfg, pat=None):
                for x in g:
                    if x['config'] == cfg and (pat is None or x['pattern'].startswith(pat)):
                        return x
                return None
            st = get('stock_wA', 'e2m1')
            base = st['kernel_us'] if st else float('nan')

            def cell(x, time_key='kernel_us'):
                if x is None:
                    return ''
                t = x[time_key] if time_key == 'kernel_us' else x['event_ms'] * 1e3
                return f'{t:.1f} ({(t / base - 1) * 100:+.1f}%)'
            bf = get('bf16_cublas')
            print(f'| {k[0]} | {k[1]} ({g[0]["out"]}×{g[0]["inp"]}) | {k[2]} | {base:.1f} | '
                  f'{cell(get("n16k64_wA_nodisp"))} | {cell(get("n16k64_wA", "e2m1"))} | {cell(get("n16k64_wA", "selector"))} | '
                  f'{cell(get("n16k64_wA", "random50"))} | {cell(get("n16k64_wA_8x1", "selector"))} | {cell(get("n8k64_wB", "selector"))} | '
                  f'{bf["event_ms"] * 1e3:.1f} |' if bf else '')


def linear(files):
    for f in files:
        r = json.loads(Path(f).read_text())
        print(f'GPU: {r["gpu"].get("device")}\n')
        print('| model | proj | T | policy | total ms | graph ms | host µs/call | quant µs | GEMM µs |')
        print('|---|---|---:|---|---:|---:|---:|---:|---:|')
        for x in r['rows']:
            print(f'| {x["model"]} | {x["proj"]} | {x["tokens"]} | {x["policy"]} | {x["total_ms"]:.4f} | {x["graph_ms"]:.4f} | '
                  f'{x["host_us"]:.1f} | {x["quant_us"]:.1f} | {x["gemm_us"]:.1f} |')


def model(files):
    rs = [json.loads(Path(f).read_text()) for f in files]
    specs = list(rs[0]['prefill'])
    print('| model | policy | weights GiB | peak GiB | ' + ' | '.join(f'prefill {s} ms' for s in specs) + ' |')
    print('|---|---|---:|---:|' + '---:|' * len(specs))
    for r in rs:
        print(f'| {r["model"]} | {r["policy"].split("/")[-1]} | {r["weights_gib"]:.2f} | {r.get("peak_total_gib", float("nan")):.2f} | '
              + ' | '.join(f'{r["prefill"][s].get("ms", float("nan")):.1f}' for s in specs) + ' |')
    print()
    bs = list(rs[0]['decode_eager'])
    print('| model | policy | ' + ' | '.join(f'decode B={b} eager tok/s | graph tok/s' for b in bs) + ' |')
    print('|---|---|' + '---:|---:|' * len(bs))
    for r in rs:
        cells = []
        for b in bs:
            e, g = r['decode_eager'].get(b, {}), r['decode_graph'].get(b, {})
            cells += [f'{e.get("tokens_per_s", float("nan")):.1f}', f'{g["tokens_per_s"]:.1f}' if 'tokens_per_s' in g else 'n/a']
        print(f'| {r["model"]} | {r["policy"].split("/")[-1]} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2:])
