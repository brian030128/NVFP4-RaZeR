#!/usr/bin/env python3
"""Experiment A tables: results/paper_extra/A/{A.md, A.csv, A_primary.csv, A_retention.csv, A_mixing.csv,
A_table_<model>_<corpus>.tex, A.json} from the A_formats.py records. CPU, seconds.

    PAPER_PYTHON experiments/paper_extra/A_analyze.py [--models llama8b] [--out A_OUT] [--dest results/paper_extra/A]

Names in every output: "IF4 (Cook et al.)", "MixFP4 (Zou et al.)", "MixFP4 (Zou et al.) + FourOverSix" and "IF4 (Cook
et al.) + FourOverSix" (the user's variants, "(our variant)" in the LaTeX), "FlipQuant (ours)" (code path TM-OPT+TC);
never a bare "MixFP4". Every comparison is a paired ΔNLL (nats per token = Δ log PPL)
over the same windows (token hashes compared), on WikiText-2 and C4: 1 SE in the CSVs, ± 2 SE in the markdown;
negative = the first policy has the lower NLL.
PRIMARY (results/paper_extra/A/PROTOCOL.md, fixed before the runs; amendment 2 adds the rule if4fo6), for R in the rules:
  (a) degradation R@g − R@1x16, g in {8x64, 16x64, 256x64};
  (b) gain over the rule's own base R@g − base(R), every g: base = the rule's own E2M1 candidate everywhere (NVFP4
      weights) with FourOverSix activations, e2m1 for IF4 and e2m1z for Zou; FourOverSix for Zou + FO6 and IF4 + FO6;
  (c) ours vs the rule, tc@g − R@g;
  (d) retention: gain_g = NLL(FourOverSix) − NLL(R@g) and retained = gain_g / gain_1x16, with a 95 % paired bootstrap
      interval over windows (10,000 resamples, seed 0); no fraction when R@1x16 is not better than FourOverSix by 2 SE.
CONTRAST (amendment 2): IF4 + FO6 − Zou + FO6 at every g, the two uniform candidates on the same FourOverSix base.
MECHANISM: the within-tile mixing of each rule's 1x16 choices at 8x64 / 16x64 / 256x64 (tiles whose blocks disagree;
the mean minority share inside those), per projection and overall; the uniform-format share per projection.
WEIGHT ERROR: the installed weights' total squared error, relative to FourOverSix and to NVFP4, next to ΔNLL.
SECONDARY: every policy against FourOverSix and against NVFP4 (the paper row, NVFP4 activations).
PAPER TABLES: LaTeX, per model and corpus: the four rules and ours at 1x16 / 8x64 / 16x64 / 256x64, with the reference
line (NVFP4, NVFP4 weights + FourOverSix act., FourOverSix, BF16), all from the same fake (c) path.
"""
import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper'))
import paper_common as P  # noqa: E402
from A_formats import OUT, POLICIES, UNITS  # noqa: E402

PARTS23 = Path('/home/dev/n16k64_campaign/deploy_eval/runs')
PARTS23_LABEL = {'bf16': 'BF16', 'nvfp4': 'NVFP4-fake', 'fo6': 'FourOverSix-fake', 'tc-8x64': 'tc-8x64-fake',
                 'tc-16x64': 'tc-16x64-fake', 'tc-256x64': 'tc-256x64-fake'}
PROJ = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
RULES = ('if4', 'zou', 'zoufo6', 'if4fo6')
RULE_NAME = {'if4': 'IF4 (Cook et al.)', 'zou': 'MixFP4 (Zou et al.)', 'zoufo6': 'MixFP4 (Zou et al.) + FourOverSix',
             'if4fo6': 'IF4 (Cook et al.) + FourOverSix',
             'tc': 'FlipQuant (ours)'}
REF_NAME = {'bf16': 'BF16', 'nvfp4': 'NVFP4', 'fo6': 'FourOverSix', 'e2m1': 'NVFP4 weights + FourOverSix act.',
            'e2m1z': 'NVFP4 weights (MixFP4 (Zou et al.) E2M1) + FourOverSix act.'}
BASE = dict(if4='e2m1', zou='e2m1z', zoufo6='fo6', if4fo6='fo6')
COARSE = ('8x64', '16x64', '256x64')
CORPORA = (('wiki', 'WikiText-2'), ('c4', 'C4'))
BOOT = 10000


def name(p):
    if p in REF_NAME:
        return REF_NAME[p]
    rule, g = p.split('-', 1)
    return f'{RULE_NAME[rule]} {g}'


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    return sum(d) / len(d), statistics.stdev(d) / math.sqrt(len(d)), len(d)


def read(p):
    try:
        r = json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None
    return r if r.get('status') == 'complete' else None


def map_fraction(entry):
    """The share of weights in E0M3 tiles of a fake:map policy, overall and per projection, from its .mixfp4map."""
    sys.path.insert(0, str(P.REPO / 'sm120'))
    from mixfp4_sm120 import mapio
    header, masks, _ = mapio.read_map(entry['map'])
    tr, tc = header['type_block']
    per, tot = {}, [0, 0]
    for m in header['modules']:
        rows, cols = m['weight_shape']
        mask = masks[m['name']]
        height = [min(tr, rows - i * tr) for i in range(mask.shape[0])]
        sel = sum(int(mask[i].sum()) * height[i] * tc for i in range(mask.shape[0]))
        for d in (per.setdefault(m['name'].rsplit('.', 1)[-1], [0, 0]), tot):
            d[0] += sel
            d[1] += rows * cols
    assert tot[0] == header['totals']['selected_weights'] and tot[1] == header['totals']['total_weights'], header['totals']
    return tot[0] / tot[1], {k: v[0] / v[1] for k, v in per.items()}


def retention(fo6, r1, rg, seed=0):
    """gain_g = FO6 − R@g, gain_1 = FO6 − R@1x16 (per-window NLL lists); the ratio and its 95 % paired bootstrap interval."""
    n = len(fo6)
    g1 = [a - b for a, b in zip(fo6, r1)]
    gg = [a - b for a, b in zip(fo6, rg)]
    rng = random.Random(seed)
    ratios = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        s1 = sum(g1[i] for i in idx)
        if s1 != 0:
            ratios.append(sum(gg[i] for i in idx) / s1)
    ratios.sort()
    lo, hi = ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios)) - 1]
    return sum(gg) / sum(g1), lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default='llama8b')
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--dest', type=Path, default=P.REPO / 'results' / 'paper_extra' / 'A')
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    arm_rows, prim_rows, ret_rows, mix_rows, md, data = [], [], [], [], [], {}
    a1 = read(args.dest / 'a1_check.json') or (json.loads((args.dest / 'a1_check.json').read_text())
                                                if (args.dest / 'a1_check.json').exists() else None)
    for model in args.models.split(','):
        rep = {p: read(args.out / 'ppl' / model / p / 'report.json') for p in POLICIES}
        rep = {p: r for p, r in rep.items() if r is not None}
        if not {'fo6', 'nvfp4', 'e2m1', 'e2m1z'} <= set(rep):
            continue
        ev = {p: r['evaluations'][p] for p, r in rep.items()}
        tok = {p: json.dumps([r['data']['wiki']['token_sha256'], r['data']['c4_paper']['token_sha256']]) for p, r in rep.items()}
        assert len(set(tok.values())) == 1, f'{model}: the policies evaluated different windows'
        nll = {p: {c: e['evaluation'][c]['nll'] for c, _ in CORPORA} for p, e in ev.items()}
        ppl = {p: {c: e['evaluation'][c]['ppl'] for c, _ in CORPORA} for p, e in ev.items()}
        err = {p: e.get('installed_weight_sq_error') for p, e in ev.items()}
        d = data[model] = dict(policies={}, primary=[], retention=[], mixing={}, parts23_check={})
        old = read(PARTS23 / model / 'report.json')
        for p, lab in PARTS23_LABEL.items():
            if old is not None and p in ev and lab in old['evaluations']:
                d['parts23_check'][p] = all(nll[p][c] == old['evaluations'][lab]['evaluation'][c]['nll'] for c, _ in CORPORA)

        def comp(kind, rule, gran, a, b):
            if a not in nll or b not in nll:
                return None
            rec = dict(comparison=kind, rule=rule, granularity=gran, policy=a, reference=b)
            for c, _ in CORPORA:
                m, se, n = paired(nll[a][c], nll[b][c])
                rec.update({f'dnll_{c}': m, f'se_{c}': se, f'n_{c}': n})
            d['primary'].append(rec)
            prim_rows.append([model, kind, rule, RULE_NAME[rule], gran, a, b] +
                             [f"{rec[k]:.6f}" for k in ('dnll_wiki', 'se_wiki', 'dnll_c4', 'se_c4')])
            return rec

        def pm(r, c):
            if r is None:
                return '—'
            star = ' *' if abs(r['dnll_' + c]) > 2 * r['se_' + c] else ''
            return f"{r['dnll_' + c]:+.4f} ± {2 * r['se_' + c]:.4f}{star}"
        md.append(f'## {P.TITLES[model]}\n')
        md += ['### Primary (a): degradation when coarsened, R@g − R@1x16\n', '| rule | g | WikiText-2 | C4 |', '|---|---|---:|---:|']
        for R in RULES:
            for g in COARSE:
                r = comp('degradation', R, g, f'{R}-{g}', f'{R}-1x16')
                md.append(f'| {RULE_NAME[R]} | {g} | {pm(r, "wiki")} | {pm(r, "c4")} |')
        md += ['\n### Primary (b): gain over the rule\'s own base, R@g − base\n',
               'Base: each rule\'s own E2M1 candidate everywhere with FourOverSix activations (e2m1 for IF4, e2m1z for MixFP4 '
               '(Zou et al.): NVFP4 weights); FourOverSix for MixFP4 (Zou et al.) + FourOverSix and IF4 (Cook et al.) + '
               'FourOverSix. The same FourOverSix activations everywhere.\n',
               '| rule | base | g | WikiText-2 | C4 |', '|---|---|---|---:|---:|']
        for R in RULES:
            for g in UNITS:
                r = comp('gain_over_base', R, g, f'{R}-{g}', BASE[R])
                md.append(f'| {RULE_NAME[R]} | {REF_NAME[BASE[R]]} | {g} | {pm(r, "wiki")} | {pm(r, "c4")} |')
        md += ['\n### Primary (c): FlipQuant (ours; TM-OPT+TC maps) against the rule at the same tile, tc@g − R@g\n',
               '| rule | g | WikiText-2 | C4 |', '|---|---|---:|---:|']
        for R in RULES:
            for g in COARSE:
                r = comp('ours_minus_rule', R, g, f'tc-{g}', f'{R}-{g}')
                md.append(f'| {RULE_NAME[R]} | {g} | {pm(r, "wiki")} | {pm(r, "c4")} |')
        md += ['\n### Contrast (amendment 2): IF4 (Cook et al.) + FourOverSix − MixFP4 (Zou et al.) + FourOverSix, at the same g\n',
               'The two uniform candidates (IF4\'s INT4 with the shared max/6 scale; Zou\'s E1M2 with its own max/7 scale) on the '
               'same FourOverSix E2M1 base, each with its own tie rule.\n', '| g | WikiText-2 | C4 |', '|---|---:|---:|']
        for g in UNITS:
            r = comp('if4fo6_minus_zoufo6', 'if4fo6', g, f'if4fo6-{g}', f'zoufo6-{g}')
            md.append(f'| {g} | {pm(r, "wiki")} | {pm(r, "c4")} |')
        md.append('\n`*` = |Δ| > 2 SE.\n')
        # (d) retention of the 1x16 gain over FourOverSix
        md += ['### Primary (d): how much of the 1x16 gain over FourOverSix survives at the tile\n',
               'gain_g = NLL(FourOverSix) − NLL(R@g), ± 2 SE (positive = better than FourOverSix); retained = gain_g / '
               'gain_1x16 with a 95 % paired bootstrap interval over windows. No fraction where R@1x16 is not better than '
               'FourOverSix by 2 SE.\n', '| rule | corpus | gain 1x16 | g | gain_g | retained [95 %] |', '|---|---|---:|---|---:|---:|']
        for R in RULES:
            if f'{R}-1x16' not in nll:
                continue
            for c, title in CORPORA:
                g1, se1, _ = paired(nll['fo6'][c], nll[f'{R}-1x16'][c])
                sig = g1 > 2 * se1
                for g in COARSE:
                    if f'{R}-{g}' not in nll:
                        continue
                    gg, seg, _ = paired(nll['fo6'][c], nll[f'{R}-{g}'][c])
                    rec = dict(rule=R, corpus=c, gain_1x16=g1, se_1x16=se1, significant_1x16=sig, granularity=g,
                               gain_g=gg, se_g=seg)
                    if sig:
                        rec['retained'], rec['retained_lo'], rec['retained_hi'] = retention(nll['fo6'][c], nll[f'{R}-1x16'][c],
                                                                                           nll[f'{R}-{g}'][c])
                    d['retention'].append(rec)
                    ret_rows.append([model, R, RULE_NAME[R], c, g, f'{g1:.6f}', f'{se1:.6f}', str(sig), f'{gg:.6f}', f'{seg:.6f}'] +
                                    ([f"{rec['retained']:.4f}", f"{rec['retained_lo']:.4f}", f"{rec['retained_hi']:.4f}"] if sig else ['', '', '']))
                    kept = (f"{100 * rec['retained']:.0f} % [{100 * rec['retained_lo']:.0f}, {100 * rec['retained_hi']:.0f}]"
                            if sig else 'no fraction: the 1x16 gain is not significant')
                    md.append(f'| {RULE_NAME[R]} | {title} | {g1:+.4f} ± {2 * se1:.4f} | {g} | {gg:+.4f} ± {2 * seg:.4f} | {kept} |')
        # mechanism: within-tile mixing of the 1x16 choices
        md += ['\n### Mechanism: within-tile mixing of each rule\'s 1x16 choices\n',
               'Tiles whose 16-blocks disagree at 1x16 (some prefer E2M1, some the uniform grid), % of tiles; in brackets, '
               'the mean minority share inside those tiles, %.\n',
               '| rule | tile | all | ' + ' | '.join(PROJ) + ' |', '|---|---|---:|' + '---:|' * len(PROJ)]
        for R in RULES:
            mixing = (ev.get(f'{R}-1x16', {}).get('format') or {}).get('mixing') or {}
            d['mixing'][R] = mixing
            for g in COARSE:
                by = mixing.get(g)
                if not by:
                    continue
                cells = []
                for k in ('all',) + PROJ:
                    x = by.get(k)
                    if x is None:
                        cells.append('—')
                        continue
                    cells.append(f"{100 * x['mixed_fraction']:.1f} ({100 * x['mean_minority_share']:.1f})" if x['mixed'] else '0.0')
                    mix_rows.append([model, R, RULE_NAME[R], g, k, str(x['tiles']), str(x['mixed']), f"{x['mixed_fraction']:.6f}",
                                     '' if x['mean_minority_share'] is None else f"{x['mean_minority_share']:.6f}"])
                md.append(f'| {RULE_NAME[R]} | {g} | ' + ' | '.join(cells) + ' |')
        # every policy: PPL, secondary comparisons, weight error, uniform share
        md += ['\n### Every policy: PPL; ΔNLL vs FourOverSix and vs NVFP4 (paper row); installed-weight squared error\n',
               '| policy | uniform share | PPL WikiText-2 | PPL C4 | ΔNLL vs FourOverSix, WikiText-2 | C4 | vs NVFP4, WikiText-2 | C4 | '
               'weight error vs FourOverSix | vs NVFP4 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for p in POLICIES:
            if p not in ev:
                continue
            rec = dict(ppl_wiki=ppl[p]['wiki'], ppl_c4=ppl[p]['c4'], weight_sq_error=err[p])
            for ref in ('fo6', 'nvfp4'):
                for c, _ in CORPORA:
                    m, se, _ = paired(nll[p][c], nll[ref][c])
                    rec[f'dnll_{c}_{ref}'], rec[f'se_{c}_{ref}'] = m, se
                rec[f'weight_error_vs_{ref}'] = (err[p] / err[ref] - 1) if err.get(p) is not None and err.get(ref) else None
            if ev[p].get('format') and ev[p]['format']['rule'] in RULES:
                f = ev[p]['format']
                rec.update(uniform_frac=f['uniform_fraction'], uniform_by_projection=f['uniform_fraction_by_projection'],
                           zero_scale_blocks=f['zero_scale_blocks'])
                rule, gran = f['rule'], f['unit']
            elif p.startswith('tc-'):
                frac, by = map_fraction(ev[p])
                rec.update(uniform_frac=frac, uniform_by_projection=by)
                rule, gran = 'tc', p[3:]
            else:
                rule, gran = p, '-'
            d['policies'][p] = rec
            deg = next((x for x in d['primary'] if x['comparison'] == 'degradation' and x['policy'] == p), None)
            gain = next((x for x in d['primary'] if x['comparison'] == 'gain_over_base' and x['policy'] == p), None)

            def fmt(x, k):
                return '' if x is None else f'{x[k]:.6f}'
            arm_rows.append([model, rule, RULE_NAME.get(rule, REF_NAME.get(rule, rule)), gran] +
                            [f"{rec[k]:.6f}" for k in ('dnll_wiki_fo6', 'se_wiki_fo6', 'dnll_c4_fo6', 'se_c4_fo6')] +
                            [f"{rec['uniform_frac']:.6f}" if 'uniform_frac' in rec else ''] +
                            [f"{rec[k]:.6f}" for k in ('dnll_wiki_nvfp4', 'se_wiki_nvfp4', 'dnll_c4_nvfp4', 'se_c4_nvfp4')] +
                            [fmt(deg, k) for k in ('dnll_wiki', 'se_wiki', 'dnll_c4', 'se_c4')] +
                            [gain['reference'] if gain else ''] + [fmt(gain, k) for k in ('dnll_wiki', 'se_wiki', 'dnll_c4', 'se_c4')] +
                            ['' if err[p] is None else f'{err[p]:.6e}'] +
                            ['' if rec[f'weight_error_vs_{r}'] is None else f"{rec[f'weight_error_vs_{r}']:.6f}" for r in ('fo6', 'nvfp4')] +
                            [f"{rec['ppl_wiki']:.4f}", f"{rec['ppl_c4']:.4f}"])

            def two(k):
                return f"{rec['dnll_' + k]:+.4f} ± {2 * rec['se_' + k]:.4f}"

            def pct(k):
                return '—' if rec[k] is None else f'{100 * rec[k]:+.1f} %'
            share = f"{100 * rec['uniform_frac']:.2f} %" if 'uniform_frac' in rec else '—'
            md.append(f"| {name(p)} | {share} | {rec['ppl_wiki']:.4f} | {rec['ppl_c4']:.4f} | {two('wiki_fo6')} | {two('c4_fo6')} | "
                      f"{two('wiki_nvfp4')} | {two('c4_nvfp4')} | {pct('weight_error_vs_fo6')} | {pct('weight_error_vs_nvfp4')} |")
        md += ['\n### Uniform-format share of the weights by projection, %\n',
               '| policy | all | ' + ' | '.join(PROJ) + ' |', '|---|---:|' + '---:|' * len(PROJ)]
        for p in [f'{R}-{g}' for R in RULES for g in UNITS] + [f'tc-{g}' for g in COARSE]:
            x = d['policies'].get(p, {})
            by = x.get('uniform_by_projection')
            if by:
                md.append(f"| {name(p)} | {100 * x['uniform_frac']:.2f} | " +
                          ' | '.join(f'{100 * by[k]:.2f}' if k in by else '—' for k in PROJ) + ' |')
        zs = {p: x['zero_scale_blocks'] for p, x in d['policies'].items() if x.get('zero_scale_blocks') is not None}
        if zs:
            md.append('\nBlocks with a zero (underflowed) E4M3 scale, which become all-zero blocks: ' +
                      ('0 in every arm.' if not any(zs.values()) else ', '.join(f'{name(p)} {n}' for p, n in zs.items()) + '.'))
        # LaTeX tables in the user's layout
        for c, title in CORPORA:
            def cell(p):
                return '--' if p not in ppl else f'{ppl[p][c]:.2f}'
            rows = [f"IF4 (Cook et al.) & {' & '.join(cell(f'if4-{g}') for g in UNITS)} \\\\",
                    f"MixFP4 (Zou et al.) & {' & '.join(cell(f'zou-{g}') for g in UNITS)} \\\\",
                    f"MixFP4 (Zou et al.) + FourOverSix (our variant) & {' & '.join(cell(f'zoufo6-{g}') for g in UNITS)} \\\\",
                    f"IF4 (Cook et al.) + FourOverSix (our variant) & {' & '.join(cell(f'if4fo6-{g}') for g in UNITS)} \\\\",
                    f"FlipQuant (ours) & -- & {' & '.join(cell(f'tc-{g}') for g in COARSE)} \\\\"]
            refs = ', '.join(f'{REF_NAME[p]} {cell(p)}' for p in ('nvfp4', 'e2m1', 'fo6', 'bf16'))
            tex = ('% ' + f'{P.TITLES[model]}, {title} perplexity, fake (c) simulator (results/paper_extra/A). FlipQuant (ours) here '
                   'is simulated (fake (c)) and so differs slightly from the native main-table numbers.\n'
                   '\\begin{tabular}{lcccc}\n\\toprule\nMethod & 1x16 & 8x64 & 16x64 & 256x64 \\\\\n\\midrule\n' +
                   '\n'.join(rows) + f'\n\\midrule\n\\multicolumn{{5}}{{l}}{{\\footnotesize Reference: {refs}}} \\\\\n'
                   '\\bottomrule\n\\end{tabular}\n')
            (args.dest / f'A_table_{model}_{c}.tex').write_text(tex)
            md += [f'\n### Paper table ({title}): `A_table_{model}_{c}.tex`\n', '```latex', tex.rstrip(), '```']
        md.append(f"\nRe-run references equal to the Parts 2-3 fake (c) records, window by window: {d['parts23_check']}\n")
    head = ['model', 'rule', 'rule_name', 'granularity', 'dnll_wiki', 'se_wiki', 'dnll_c4', 'se_c4', 'uniform_frac',
            'dnll_wiki_nvfp4', 'se_wiki_nvfp4', 'dnll_c4_nvfp4', 'se_c4_nvfp4',
            'dnll_wiki_vs_1x16', 'se_wiki_vs_1x16', 'dnll_c4_vs_1x16', 'se_c4_vs_1x16',
            'base', 'dnll_wiki_vs_base', 'se_wiki_vs_base', 'dnll_c4_vs_base', 'se_c4_vs_base',
            'weight_sq_error', 'weight_error_vs_fo6', 'weight_error_vs_nvfp4', 'ppl_wiki', 'ppl_c4']
    (args.dest / 'A.csv').write_text('\n'.join(','.join(f'"{x}"' if ',' in x else x for x in r) for r in [head] + arm_rows) + '\n')
    phead = ['model', 'comparison', 'rule', 'rule_name', 'granularity', 'policy', 'reference', 'dnll_wiki', 'se_wiki', 'dnll_c4', 'se_c4']
    (args.dest / 'A_primary.csv').write_text('\n'.join(','.join(r) for r in [phead] + prim_rows) + '\n')
    rhead = ['model', 'rule', 'rule_name', 'corpus', 'granularity', 'gain_1x16', 'se_1x16', 'significant_1x16', 'gain_g', 'se_g',
             'retained', 'retained_lo95', 'retained_hi95']
    (args.dest / 'A_retention.csv').write_text('\n'.join(','.join(r) for r in [rhead] + ret_rows) + '\n')
    mhead = ['model', 'rule', 'rule_name', 'tile', 'projection', 'tiles', 'mixed_tiles', 'mixed_fraction', 'mean_minority_share']
    (args.dest / 'A_mixing.csv').write_text('\n'.join(','.join(r) for r in [mhead] + mix_rows) + '\n')
    a1_md = ''
    if a1 is not None:
        s = a1['summary']
        a1_md = ('## A1 checks (a1_check.json)\n\n'
                 f"- IF4 against the official fouroversix reference ({a1.get('fouroversix_commit', '')[:8]}): choice equal on every "
                 f"module: {s['if4_choice_equal_all']}; dequantized values bitwise equal: {s['if4_dequantized_bitwise_equal_all']}.\n"
                 f"- IF4's FP candidate vs MixFP4 (Zou et al.)'s E2M1 candidate: {s['if4_fp_vs_zou_e2m1_elements_differing']} of "
                 f"{s['elements']} elements differ (BF16; operation order), so each rule has its own E2M1 base: e2m1 for IF4, "
                 "e2m1z for MixFP4 (Zou et al.).\n"
                 f"- MixFP4 (Zou et al.) against the repo's quant_nvif4: {s['zou_vs_repo_nvif4_elements_differing']} of "
                 f"{s['elements']} elements differ.\n"
                 f"- MixFP4 (Zou et al.)'s E1M2 candidate against the repo's E0M3 alpha = 1 candidate: "
                 f"{s['zou_e1m2_vs_repo_e0m3_elements_differing']} elements differ.\n"
                 f"- The tile rule at 1x16 equals the per-block rule, every rule and module: {s['tile_1x16_equals_block_all']}.\n")
        chk = read(args.dest / 'if4fo6_check.json') or (json.loads((args.dest / 'if4fo6_check.json').read_text())
                                                          if (args.dest / 'if4fo6_check.json').exists() else None)
        if chk:
            mods = chk['modules'].values()
            a1_md += (f"- IF4 (Cook et al.) + FourOverSix (amendment 2; if4fo6_check.json, {len(chk['modules'])} modules): FP "
                      f"candidate equals MixFP4 (Zou et al.) + FourOverSix's bitwise: {all(m['fp_equals_zoufo6_fp'] for m in mods)}; "
                      f"INT4 candidate equals IF4's as installed: {all(m['int_equals_if4_int_installed'] for m in mods)}; ties "
                      f"keep FP: {all(m['choice_is_strictly_lower'] for m in mods)}; the existing rules' outputs unchanged: "
                      f"{sum(chk['unchanged'].values())} of {len(chk['unchanged'])}.\n")
        a1_md += '\n'
    (args.dest / 'A.md').write_text(
        '# Experiment A: IF4 (Cook et al.) and MixFP4 (Zou et al.) per-block selection, coarsened to hardware tiles\n\n'
        'One fake (c) simulator for every row: the listed weights with FourOverSix per-token activations (NVFP4 activations '
        'for the NVFP4 row only). FlipQuant (ours; the TM-OPT+TC maps) here is simulated (fake (c)) and so differs slightly '
        'from the native main-table numbers. ΔNLL in nats per token (= Δ log PPL), paired over windows, ± 2 SE.\n\n' +
        a1_md + '\n'.join(md) + '\n')
    (args.dest / 'A.json').write_text(json.dumps(data, indent=1) + '\n')
    print('wrote', ', '.join(str(p.name) for p in sorted(args.dest.glob('A*'))))


if __name__ == '__main__':
    main()
