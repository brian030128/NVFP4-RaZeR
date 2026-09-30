"""Deviation 2's GEMM tables (results/paper/PROTOCOL_GEMM_ISOLATED.md), rendered by step 07 from <out>/gemm_isolated.

The primary GEMM numbers since deviation 2: isolated launches, cold weights, CUPTI device time, the median over all
rounds; the FlipQuant (ours) maps with the typical and the worst module's tags per projection. Per-forward sums as step
06: each projection's time times its module count; the quantizer's launches net of the reuse measured in step 05.
"""
import statistics

import paper_common as P

DASH = '—'
TOLERANCE_PP = 1.0
TITLES = dict(stock_wA='stock wA', stock_wA_nvfp4='stock wA (NVFP4 weights)', stock_wB='stock wB',
              mixed_16x64_typical='FlipQuant (ours) 16x64, typical', mixed_16x64_worst='FlipQuant (ours) 16x64, worst',
              mixed_256x64_typical='FlipQuant (ours) 256x64, typical', mixed_256x64_worst='FlipQuant (ours) 256x64, worst',
              n8k64_wB_typical='FlipQuant (ours) 8x64, typical', n8k64_wB_worst='FlipQuant (ours) 8x64, worst')
UNIT_CONFIGS = {'16x64': ('mixed_16x64_typical', 'mixed_16x64_worst'), '256x64': ('mixed_256x64_typical', 'mixed_256x64_worst'),
                '8x64': ('n8k64_wB_typical', 'n8k64_wB_worst')}
KIND_OF_UNIT = {'16x64': 'tc_16x64', '256x64': 'tc_256x64', '8x64': 'tc_8x64'}
# (prefill policy, prefill reference, isolated configuration, isolated reference, the old step-06 pair)
CONSISTENCY = [('ours-16x64', 'fo6', 'mixed_16x64_{v}', 'stock_wA', ('mixed_16x64', 'stock_wA')),
               ('ours-256x64', 'fo6', 'mixed_256x64_{v}', 'stock_wA', ('mixed_256x64', 'stock_wA')),
               ('ours-8x64', 'fo6-wB', 'n8k64_wB_{v}', 'stock_wB', ('n8k64_wB', 'stock_wB')),
               ('ours-8x64', 'fo6', 'n8k64_wB_{v}', 'stock_wA', ('n8k64_wB', 'stock_wA'))]
METHOD = ('isolated launches, cold weights (distinct weight copies >= 4x L2 plus a 512 MiB read-flush before every '
          'call), CUPTI device time, median of 3 rounds x 30 repetitions in a rotated order')
ALT_METHOD = 'CUPTI, back-to-back calls, L2-warm, densest module'


def per_forward(rec, reuse):
    """{(config, T): us} GEMM sums and {(act, T): us} quantizer sums, or None if the record lacks a projection."""
    projs = rec['projections']
    by = {(r['config'], r['proj'], r['tokens']): r for r in rec['rows']}
    timed = [p for p in projs if any(r['proj'] == p for r in rec['rows'])]
    tokens = sorted({r['tokens'] for r in rec['rows']})
    cfgs = sorted({r['config'] for r in rec['rows']})
    sums, qsums = {}, {}
    for t in tokens:
        for c in cfgs:
            if all((c, p, t) in by for p in timed):
                sums[(c, t)] = sum(projs[p]['modules'] * by[(c, p, t)]['gemm_us'] for p in timed)
        for c, act in (('stock_wA', 'four_over_six_rows'), ('stock_wA_nvfp4', 'nvfp4_rows')):
            if all((c, p, t) in by for p in timed):
                launches = {p: projs[p]['modules'] * (1 - (reuse[p]['quant_reused'] / reuse[p]['calls'] if reuse and p in reuse
                                                           and reuse[p]['calls'] else 0.0)) for p in timed}
                qsums[(act, t)] = sum(launches[p] * by[(c, p, t)]['quant_us'] for p in timed)
    return sums, qsums, tokens, timed, by


def pct(a, b):
    return f'{100 * (a / b - 1):+.1f} %'


def sections(out, models, prefill, table, read, old_dir='gemm'):
    """-> dict of rendered texts and a JSON-able dict. Keys: main, side_by_side, appendix_256, consistency_text,
    consistency_summary, detail, tags, telemetry; data."""
    texts = {k: [] for k in ('main', 'side', 'appendix_256', 'detail', 'tags', 'telemetry')}
    data, check_rows, flagged = {}, [], {}
    for model in models:
        rec = read(out / 'gemm_isolated' / f'{model}.json')
        if rec is None:
            continue
        reuse = next((v['quant_reuse'] for v in prefill.get(model, {}).values() if v.get('quant_reuse')), None)
        sums, qsums, tokens, timed, by = per_forward(rec, reuse)
        complete = len(timed) == len(rec['projections'])
        scope = 'every quantized text Linear' if complete else f'ONLY {", ".join(timed)} (a subset)'
        d = data[model] = dict(per_forward_gemm_us={f'{c}@{t}': v for (c, t), v in sums.items()},
                               per_forward_quant_us={f'{a}@{t}': v for (a, t), v in qsums.items()}, quant_reuse=reuse,
                               complete=complete, tags={p: rec['projections'][p]['tags'] for p in timed},
                               power_limit_w=rec.get('power_limit_w'))
        # main: 16x64 and 8x64, absolute and overheads
        cols = ['stock_wA', 'stock_wB', 'mixed_16x64_typical', 'mixed_16x64_worst', 'n8k64_wB_typical', 'n8k64_wB_worst']
        rows = [[t] + [f'{sums[(c, t)]:.0f}' if (c, t) in sums else DASH for c in cols] for t in tokens]
        ratios = [('mixed_16x64_typical', 'stock_wA'), ('mixed_16x64_worst', 'stock_wA'), ('n8k64_wB_typical', 'stock_wB'),
                  ('n8k64_wB_worst', 'stock_wB'), ('n8k64_wB_typical', 'stock_wA'), ('n8k64_wB_worst', 'stock_wA')]
        rrows = [[t] + [pct(sums[(a, t)], sums[(b, t)]) if (a, t) in sums and (b, t) in sums else DASH for a, b in ratios]
                 for t in tokens]
        qrows = [[t] + [f'{qsums[(a, t)]:.0f}' if (a, t) in qsums else DASH for a in ('nvfp4_rows', 'four_over_six_rows')] +
                 [pct(qsums[('four_over_six_rows', t)], qsums[('nvfp4_rows', t)])
                  if ('four_over_six_rows', t) in qsums and ('nvfp4_rows', t) in qsums else DASH] for t in tokens]
        texts['main'].append(
            f'#### {P.TITLES[model]}: GEMM kernel time per forward, µs ({scope}; {METHOD})\n\n' +
            table(['T'] + [TITLES[c] for c in cols], rows) + '\n\nOverheads against the same-placement stock kernel '
            '(and 8x64 against stock wA):\n\n' + table(['T'] + [f'{TITLES[a]} vs {TITLES[b]}' for a, b in ratios], rrows) +
            '\n\nThe activation quantizer per forward, µs (isolated launches; FourOverSix and NVFP4 share the stock GEMM):\n\n' +
            table(['T', 'NVFP4 quantizer', 'FourOverSix quantizer', 'FourOverSix vs NVFP4'], qrows) +
            ('\n\nQuantizer launches net of the reuse measured in step 05.' if reuse else
             '\n\nNo step-05 record: every Linear counted as quantizing its own input (an upper bound).'))
        # appendix: 256x64
        cols2 = ['stock_wA', 'mixed_256x64_typical', 'mixed_256x64_worst']
        rows2 = [[t] + [f'{sums[(c, t)]:.0f}' if (c, t) in sums else DASH for c in cols2] +
                 [pct(sums[(c, t)], sums[('stock_wA', t)]) if (c, t) in sums and ('stock_wA', t) in sums else DASH
                  for c in cols2[1:]] for t in tokens]
        texts['appendix_256'].append(f'#### {P.TITLES[model]}: GEMM kernel time per forward, µs, 256x64 ({scope}; {METHOD})\n\n' +
                                     table(['T'] + [TITLES[c] for c in cols2] + [f'{TITLES[c]} vs stock wA' for c in cols2[1:]], rows2))
        # side by side: the old step-06 numbers, the new ones, end to end
        old = read(out / old_dir / f'{model}.json')
        osums = {}
        if old is not None:
            oby = {(r['config'], r['proj'], r['tokens']): r['gemm_us'] for r in old['rows']}
            omods = {p: v['modules'] for p, v in old['projections'].items()}
            for c in ('stock_wA', 'stock_wB', 'mixed_16x64', 'mixed_256x64', 'n8k64_wB'):
                for t in tokens:
                    if all((c, p, t) in oby for p in omods):
                        osums[(c, t)] = sum(omods[p] * oby[(c, p, t)] for p in omods)
        pre = prefill.get(model, {})

        def e2e(pol, ref, s):
            ea, eb = pre.get(pol, {}).get('graph', {}).get(s), pre.get(ref, {}).get('graph', {}).get(s)
            if not ea or not eb:
                return None
            common = sorted(set(ea['rounds']) & set(eb['rounds']))
            return statistics.median(100 * (ea['rounds'][r] / eb['rounds'][r] - 1) for r in common) if common else None
        srows, sdata = [], []
        for s in P.PREFILL_SHAPES:
            b_, p_ = (int(v) for v in s.split('x'))
            t = b_ * p_
            if t not in tokens:
                continue
            row, rec_s = [s], dict(shape=s, tokens=t)
            for label, old_pair, new_cfgs, new_ref, pol, ref in (
                    ('16x64', ('mixed_16x64', 'stock_wA'), ('mixed_16x64_typical', 'mixed_16x64_worst'), 'stock_wA', 'ours-16x64', 'fo6'),
                    ('8x64', ('n8k64_wB', 'stock_wB'), ('n8k64_wB_typical', 'n8k64_wB_worst'), 'stock_wB', 'ours-8x64', 'fo6-wB')):
                o = (100 * (osums[(old_pair[0], t)] / osums[(old_pair[1], t)] - 1)
                     if (old_pair[0], t) in osums and (old_pair[1], t) in osums else None)
                n_ = [100 * (sums[(c, t)] / sums[(new_ref, t)] - 1) if (c, t) in sums and (new_ref, t) in sums else None for c in new_cfgs]
                e = e2e(pol, ref, s)
                rec_s[label] = dict(old=o, typical=n_[0], worst=n_[1], e2e=e)
                row += [DASH if v is None else f'{v:+.1f} %' for v in (o, n_[0], n_[1], e)]
            srows.append(row)
            sdata.append(rec_s)
        d['side_by_side'] = sdata
        texts['side'].append(
            f'#### {P.TITLES[model]}: overhead against the same-placement stock kernel, three ways\n\n'
            f'GEMM per forward, {ALT_METHOD} (old step 06); GEMM per forward, isolated and cold (new, typical / worst '
            'tags); and the end-to-end CUDA-graph prefill (FlipQuant (ours) vs FourOverSix with the same placement, the '
            'median over rounds of the per-round ratio).\n\n' +
            table(['batch x prompt', '16x64: GEMM, old', '16x64: GEMM, new typical', '16x64: GEMM, new worst', '16x64: end to end',
                   '8x64 (wB): GEMM, old', '8x64 (wB): GEMM, new typical', '8x64 (wB): GEMM, new worst', '8x64 (wB): end to end'], srows))
        # consistency against end to end, typical (primary) and worst
        flagged[model] = dict(typical=0, worst=0, rows=0)
        d['consistency'] = []
        if complete:
            mods = {p: rec['projections'][p]['modules'] for p in timed}
            pf = lambda c, t: sum(mods[p] * by[(c, p, t)]['gemm_us'] for p in mods) / 1e3  # noqa: E731  (ms)
            for pol, ref, cfg, rcfg, _ in CONSISTENCY:
                if pol not in pre or ref not in pre:
                    continue
                for s in P.PREFILL_SHAPES:
                    ea, eb = pre[pol].get('graph', {}).get(s), pre[ref].get('graph', {}).get(s)
                    b_, p_ = (int(v) for v in s.split('x'))
                    t = b_ * p_
                    if not ea or not eb:
                        continue
                    common = sorted(set(ea['rounds']) & set(eb['rounds']))
                    diffs = [ea['rounds'][r] - eb['rounds'][r] for r in common]
                    if not diffs:
                        continue
                    d_e2e, t_ref = statistics.median(diffs), eb['ms']
                    entry = dict(ours=pol, ref=ref, shape=s, tokens=t, e2e_ms=d_e2e, e2e_pct=100 * d_e2e / t_ref)
                    cells = []
                    for v in ('typical', 'worst'):
                        c = cfg.format(v=v)
                        if not all((x, p, t) in by for x in (c, rcfg) for p in mods):
                            cells += [DASH, DASH, '']
                            continue
                        d_gemm = pf(c, t) - pf(rcfg, t)
                        gap = 100 * (d_e2e - d_gemm) / t_ref
                        flag = abs(gap) > TOLERANCE_PP
                        flagged[model][v] += flag
                        entry[v] = dict(gemm_ms=d_gemm, gemm_pct=100 * d_gemm / t_ref, gap_pp=gap, flag=flag)
                        cells += [f'{d_gemm:+.3f} ({100 * d_gemm / t_ref:+.1f} %)', f'{gap:+.1f}', 'FLAG' if flag else '']
                    flagged[model]['rows'] += 1
                    d['consistency'].append(entry)
                    check_rows.append([P.TITLES[model], f'{pol} vs {ref}', s, f'{d_e2e:+.3f} ({entry["e2e_pct"]:+.1f} %)'] + cells)
        # per-shape detail, tags, telemetry
        cfgs = [c for c in TITLES if any(r['config'] == c for r in rec['rows'])]
        for p in timed:
            for t in tokens:
                texts['detail'].append([P.TITLES[model], p, 'x'.join(str(v) for v in rec['projections'][p]['shape']),
                                        rec['projections'][p]['modules'], t] +
                                       [(f"{by[(c, p, t)]['gemm']['median']:.1f} [{by[(c, p, t)]['gemm']['q1']:.1f}, "
                                         f"{by[(c, p, t)]['gemm']['q3']:.1f}]" + (f" (w{by[(c, p, t)]['width']})" if by[(c, p, t)]['width'] else ''))
                                        if (c, p, t) in by else DASH for c in cfgs] +
                                       [f"{by[(c, p, t)]['quant_us']:.1f}" if (c, p, t) in by else DASH for c in ('stock_wA_nvfp4', 'stock_wA')])
        for u in ('16x64', '8x64', '256x64'):
            kind = KIND_OF_UNIT[u]
            for p in timed:
                tg = rec['projections'][p]['tags'].get(kind)
                if tg:
                    texts['tags'].append([P.TITLES[model], u, p, rec['projections'][p]['modules'],
                                          f"{tg['typical']['module']} ({100 * tg['typical']['e0m3_share']:.2f} %)",
                                          f"{tg['worst']['module']} ({100 * tg['worst']['e0m3_share']:.2f} %)"])
        blocks = rec.get('blocks') or []            # one per (projection, T): every configuration and round
        smp = rec.get('sampler') or {}
        if blocks:
            cap = [b['power_cap_ms'] for b in blocks]
            mp = [b['mean_power_w'] for b in blocks if b.get('mean_power_w') is not None]
            tele = dict(blocks=len(blocks), seconds=sum(b['seconds'] for b in blocks), sampler=smp,
                        block_mean_power_w_max=max(mp) if mp else None, block_mean_power_w_median=statistics.median(mp) if mp else None,
                        power_cap_ms=sum(cap), blocks_with_power_cap=sum(1 for v in cap if v > 0), power_limit_w=rec.get('power_limit_w'))
            d['telemetry'] = tele
            fmt = lambda v, f='{:.0f}': DASH if v is None else f.format(v)  # noqa: E731
            texts['telemetry'].append([P.TITLES[model], tele['blocks'], f"{tele['seconds']:.0f}",
                                       f"{fmt(smp.get('sm_mhz_min'))} / {fmt(smp.get('sm_mhz_median'))} / {fmt(smp.get('sm_mhz_max'))}",
                                       fmt(smp.get('busy_sm_mhz_median')),
                                       f"{fmt(tele['block_mean_power_w_median'])} / {fmt(tele['block_mean_power_w_max'])}",
                                       fmt(smp.get('power_w_max')),
                                       f"{tele['power_cap_ms']:.1f} ms in {tele['blocks_with_power_cap']} of {tele['blocks']} blocks; "
                                       f"{smp.get('sw_power_cap_samples', DASH)} of {smp.get('samples', DASH)} samples",
                                       f"{tele['power_limit_w']:.0f} W"])
    res = {}
    res['main'] = ('### GEMM latency (primary; deviation 2: isolated launches, cold weights)\n\n' +
                   ('\n\n'.join(texts['main']) if texts['main'] else DASH))
    res['side_by_side'] = ('### GEMM overheads: old method, new method, end to end\n\n' +
                           ('\n\n'.join(texts['side']) if texts['side'] else DASH)) if texts['side'] else ''
    res['appendix_256'] = ('### GEMM latency, 256x64 (primary; deviation 2)\n\n' + '\n\n'.join(texts['appendix_256'])
                           if texts['appendix_256'] else '')
    res['consistency_text'] = (
        '### GEMM vs end-to-end consistency (deviation 2: isolated, cold GEMM)\n\n'
        'Per-forward GEMM time difference (typical tags, the primary; worst tags, the bound) against the end-to-end '
        'CUDA-graph prefill difference, in ms and as % of the reference prefill; FLAG when they differ by more than '
        f'{TOLERANCE_PP:g} % of the reference prefill.\n\n' +
        table(['model', 'comparison', 'batch x prompt', 'end to end, ms (%)', 'GEMM, typical, ms (%)', 'gap, pp', 'check',
               'GEMM, worst, ms (%)', 'gap, pp', 'check'], check_rows, 3)) if check_rows else ''
    res['consistency_summary'] = ('GEMM vs end-to-end consistency (deviation 2; isolated, cold GEMM; tolerance '
                                  f'{TOLERANCE_PP:g} % of the reference prefill): ' +
                                  '; '.join(f"{P.TITLES[m]} {v['typical']} (typical) / {v['worst']} (worst) of {v['rows']} rows flagged"
                                            for m, v in flagged.items()) + ' (appendix).') if flagged else ''
    detail_cfgs = list(TITLES)
    res['detail'] = ('### GEMM kernel time per shape, µs, isolated and cold: median [interquartile range] (wN = the tile '
                     "table's CTA width)\n\n" + table(['model', 'projection', 'out x in', 'modules', 'T'] + [TITLES[c] for c in detail_cfgs] +
                                                     ['NVFP4 quantizer', 'FourOverSix quantizer'], texts['detail'], 3)) if texts['detail'] else ''
    res['tags'] = ('### The map tags timed (deviation 2): per projection, the typical module (the lower median of the E0M3 '
                   'tile share) and the worst (the densest), with their E0M3 tile shares\n\n' +
                   table(['model', 'unit', 'projection', 'modules', 'typical', 'worst'], texts['tags'], 3)) if texts['tags'] else ''
    res['telemetry'] = ('### GPU telemetry during the isolated GEMM runs\n\nNVML before and after every (projection, T) '
                        'block (its mean power from the energy counter, and the power-cap violation time); the SM clock and '
                        'the samples with the software power cap active from the 100 ms nvidia-smi sampler ("busy": samples '
                        'above 100 W).\n\n' +
                        table(['model', '(projection, T) blocks', 'seconds', 'SM clock min / median / max, MHz', 'busy SM clock median, MHz',
                               'block mean power median / max, W', 'sampled power max, W', 'power cap', 'power limit'],
                              texts['telemetry']) +
                        '\n\nThe power limit is 500 W (the default is 600 W); changing it needs root, so it was recorded, not '
                        'changed.') if texts['telemetry'] else ''
    res['data'] = dict(models=data, tolerance_pp=TOLERANCE_PP, flagged=flagged, method=METHOD, alternative_method=ALT_METHOD)
    return res
