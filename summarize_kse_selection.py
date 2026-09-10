"""Turn the rebuilt k=3 election into the report's tile-location section.

Reads the per-model JSON that analyze_kse_selection.py writes and answers "which
tiles, and where": how the elected tiles split across projection kinds, depth and
individual matrices, and whether their positions inside a matrix are structured
or look like an unstructured scatter. The uniform reference for every geometric
claim is a seeded Monte Carlo that keeps each matrix's elected count and its tile
grid and only randomises the positions.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

START = '<!-- BEGIN TILE SELECTION -->'
END = '<!-- END TILE SELECTION -->'
LABEL = dict(llama8b='Llama-3.1-8B', qwen4b='Qwen3-4B', qwen27b='Qwen3.8-27B')
ORDER = ('llama8b', 'qwen4b', 'qwen27b')
REPS = 400


def short(name):
    return name.replace('model.language_model.', '').replace('model.', '') \
               .replace('layers.', 'L').replace('self_attn.', '').replace('linear_attn.', '') \
               .replace('mlp.', '')


# Matrices that read the residual stream directly, so a column index means the same
# hidden channel band in every layer. down_proj and o_proj read layer-internal spaces.
RESIDUAL = ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj',
            'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b')


def measure(live, positions):
    """Geometry of one placement: positions[i] = (rows, cols) lists for live[i]."""
    rows_used = cols_used = max_row = max_col = 0
    tile_recur = Counter()      # tiles per (kind, column), within-matrix repeats included
    mat_recur = Counter()       # matrices per (kind, column), repeats collapsed
    res_tiles = Counter()       # tiles per residual-stream channel band
    res_mats = Counter()
    for m, (sr, sc) in zip(live, positions):
        rr, cc = Counter(sr), Counter(sc)
        rows_used += len(rr)
        cols_used += len(cc)
        max_row = max(max_row, max(rr.values()))
        max_col = max(max_col, max(cc.values()))
        for c, v in cc.items():
            tile_recur[(m['kind'], m['grid'][1], c)] += v
            mat_recur[(m['kind'], m['grid'][1], c)] += 1
            if m['kind'] in RESIDUAL:
                res_tiles[c] += v
                res_mats[c] += 1
    return dict(rows_used=rows_used, cols_used=cols_used, max_row=max_row, max_col=max_col,
                max_tile_recur=max(tile_recur.values()), max_mat_recur=max(mat_recur.values()),
                max_res_tiles=max(res_tiles.values()) if res_tiles else 0,
                max_res_mats=max(res_mats.values()) if res_mats else 0,
                _tile_recur=tile_recur, _res_tiles=res_tiles, _res_mats=res_mats)


def mc_reference(live, reps=REPS, seed=0):
    """Uniform placement keeping each matrix's elected count and its tile grid."""
    rng = random.Random(seed)
    acc = Counter()
    keys = ('rows_used', 'cols_used', 'max_row', 'max_col', 'max_tile_recur',
            'max_mat_recur', 'max_res_tiles', 'max_res_mats')
    for _ in range(reps):
        pos = []
        for m in live:
            nr, nc = m['grid']
            cells = rng.sample(range(nr * nc), m['selected'])
            pos.append(([c // nc for c in cells], [c % nc for c in cells]))
        g = measure(live, pos)
        for k in keys:
            acc[k] += g[k]
    return {k: acc[k] / reps for k in keys}


def stats(r):
    mods = r['modules']
    sel = r['counts'][str(r['k'])]
    live = [m for m in mods if m['selected']]
    s = dict(model=r['model'], label=LABEL[r['model']], total=r['total_tiles'], selected=sel,
             one_in=round(r['total_tiles'] / sel), modules=len(mods), modules_hit=len(live),
             counts=r['counts'])

    by_kind = defaultdict(lambda: [0, 0])
    by_layer = Counter()
    by_block = defaultdict(lambda: [0, 0])
    for m in mods:
        by_kind[m['kind']][0] += m['tiles']
        by_kind[m['kind']][1] += m['selected']
        by_block[m['block']][0] += m['tiles']
        by_block[m['block']][1] += m['selected']
        by_layer[m['layer']] += m['selected']
    s['by_kind'] = {k: dict(tiles=v[0], selected=v[1], per_million=1e6 * v[1] / v[0],
                            share=v[1] / sel) for k, v in by_kind.items()}
    s['by_block'] = {k: dict(tiles=v[0], selected=v[1], per_million=1e6 * v[1] / v[0],
                             share=v[1] / sel) for k, v in by_block.items()}
    nlayers = max(by_layer) + 1
    s['n_layers'] = nlayers
    s['by_layer'] = {str(i): by_layer.get(i, 0) for i in range(nlayers)}
    q = [0, 0, 0, 0]
    for i in range(nlayers):
        q[min(3, 4 * i // nlayers)] += by_layer.get(i, 0)
    s['by_quarter'] = [dict(count=c, share=c / sel) for c in q]
    ranked = sorted(live, key=lambda m: -m['selected'])
    s['top_modules'] = [dict(name=short(m['name']), selected=m['selected'], tiles=m['tiles'],
                             grid=m['grid'], share=m['selected'] / sel) for m in ranked[:10]]
    s['top5_share'] = sum(m['selected'] for m in ranked[:5]) / sel
    s['max_module'] = ranked[0]['selected']
    s['single_tile_modules'] = sum(1 for m in live if m['selected'] == 1)

    # geometry: distinct rows/columns actually touched, and the worst concentration
    g = measure(live, [(m['tiles_rows'], m['tiles_cols']) for m in live])
    res_mats = sum(1 for m in live if m['kind'] in RESIDUAL)
    s['geometry'] = {k: v for k, v in g.items() if not k.startswith('_')}
    s['geometry']['mc'] = mc_reference(live)
    s['geometry']['residual_matrices'] = res_mats
    s['geometry']['residual_tiles'] = sum(g['_res_tiles'].values())
    # the single worst within-matrix column pile-up, for the "where inside" example
    worst = None
    for m in live:
        cc, rr = Counter(m['tiles_cols']), Counter(m['tiles_rows'])
        c, v = cc.most_common(1)[0]
        if worst is None or v > worst['count']:
            worst = dict(name=short(m['name']), selected=m['selected'], col=c, count=v,
                         channels=[64 * c, 64 * c + 63], distinct_cols=len(cc),
                         distinct_rows=len(rr), grid=m['grid'])
    s['worst_column'] = worst
    # the fullest single row: the largest share of one row's column bands elected at once
    worst = None
    for m in live:
        c, v = Counter(m['tiles_rows']).most_common(1)[0]
        fill = v / m['grid'][1]
        if v >= 8 and (worst is None or fill > worst['fill']):
            worst = dict(name=short(m['name']), selected=m['selected'], row=c, count=v,
                         fill=fill, grid=m['grid'])
    s['worst_row'] = worst
    s['hot_columns'] = [dict(kind=k, cols=nc, col=c, hits=v)
                        for (k, nc, c), v in g['_tile_recur'].most_common(5)]
    s['hot_residual'] = [dict(col=c, channels=[64 * c, 64 * c + 63], tiles=v,
                              matrices=g['_res_mats'][c])
                         for c, v in g['_res_tiles'].most_common(5)]

    binding = Counter()
    t_ce, t_kl = [], []
    for m in live:
        binding.update(m['binding'])
        t_ce += m['t_ce']
        t_kl += m['t_kl']
    med = lambda x: sorted(x)[len(x) // 2]
    s['binding'] = dict(ce=binding['ce'], kl=binding['kl'], ce_share=binding['ce'] / sel)
    s['t'] = dict(ce_median=med(t_ce), kl_median=med(t_kl), ce_min=min(t_ce), kl_min=min(t_kl))
    return s


def tables(S):
    L = ['| Model | Tiles | Elected | Rate | Matrices holding one | Largest single matrix |',
         '|---|---:|---:|---:|---:|---:|']
    for s in S:
        L.append(f'| {s["label"]} | {s["total"]:,} | {s["selected"]:,} | 1 in {s["one_in"]:,} | '
                 f'{s["modules_hit"]}/{s["modules"]} | {s["max_module"]:,} |')
    L.append('')
    kinds = []
    for s in S:
        for k in s['by_kind']:
            if k not in kinds:
                kinds.append(k)
    L += ['| Projection | ' + ' | '.join(f'{s["label"]} /1M | share' for s in S) + ' |',
          '|---' * (1 + 2 * len(S)) + '|']
    for k in kinds:
        cells = []
        for s in S:
            v = s['by_kind'].get(k)
            cells += ['—', '—'] if v is None else [f'{v["per_million"]:.1f}',
                                                   f'{100 * v["share"]:.1f}%']
        L.append(f'| `{k}` | ' + ' | '.join(cells) + ' |')
    L.append('')
    L += ['| Depth quarter | ' + ' | '.join(s['label'] for s in S) + ' |',
          '|---' * (1 + len(S)) + '|']
    for i, name in enumerate(('first', 'second', 'third', 'last')):
        L.append(f'| {name} | ' + ' | '.join(
            f'{s["by_quarter"][i]["count"]:,} ({100 * s["by_quarter"][i]["share"]:.1f}%)'
            for s in S) + ' |')
    L.append('')
    return L


def report_section(S, job, W):
    """The compact block that goes into MIXFP4_REPORT.md."""
    hi = lambda s, k: max(s['by_kind'].items(), key=lambda kv: kv[1][k])
    L = ['The rule is a threshold, so the map it produces is an outcome rather than a design: '
         'nothing in it constrains which matrices, which layers, or which part of a matrix the '
         'elected tiles come from. Rebuilt from the frozen calibration scores, every model\'s '
         'per-k count reproduces the published election exactly.', '',
         '### The elected tiles are spread over matrices and concentrated inside them', '',
         '| Model | Tiles | Elected | Rate | Matrices holding at least one | Largest single matrix |',
         '|---|---:|---:|---:|---:|---:|']
    for s in S:
        L.append(f'| {s["label"]} | {s["total"]:,} | {s["selected"]:,} | 1 in {s["one_in"]:,} | '
                 f'{s["modules_hit"]}/{s["modules"]} | {s["max_module"]:,} '
                 f'({100 * s["top_modules"][0]["share"]:.0f}%) |')
    L += ['', 'Almost every weight matrix contributes something, so this is not a rule that fires '
          'on a handful of layers. The mass is nevertheless very uneven: the five largest '
          'contributors hold ' + ', '.join(f'{100 * s["top5_share"]:.0f}% ({s["label"]})' for s in S)
          + ' of all elected tiles, and the single largest is '
          + ', '.join(f'`{s["top_modules"][0]["name"]}` with {s["max_module"]:,}' for s in S) + '.', '',
          '### Which projections, and how deep', '',
          '| Projection | ' + ' | '.join(f'{s["label"]} per 1M | share' for s in S) + ' |',
          '|---' * (1 + 2 * len(S)) + '|']
    kinds = []
    for s in S:
        for k in s['by_kind']:
            if k not in kinds:
                kinds.append(k)
    for k in kinds:
        cells = []
        for s in S:
            v = s['by_kind'].get(k)
            cells += ['—', '—'] if v is None else [f'{v["per_million"]:.0f}',
                                                   f'{100 * v["share"]:.1f}%']
        L.append(f'| `{k}` | ' + ' | '.join(cells) + ' |')
    tops = [hi(s, 'per_million') for s in S]
    peak = max(zip(S, tops), key=lambda t: t[1][1]['per_million'])
    rates = ', '.join(f'{t[1]["per_million"]:.0f}' for t in tops)
    lead = (f'`{tops[0][0]}` carries the highest rate on all three models ({rates} per 1M)'
            if len({t[0] for t in tops}) == 1 else
            'the highest rate goes to ' + ', '.join(
                f'`{t[0]}` on {s["label"]} ({t[1]["per_million"]:.0f} per 1M)'
                for s, t in zip(S, tops)))
    L += ['', 'Rates are elected tiles per million tiles of that kind, so they are comparable '
          'across matrices of very different sizes; shares are of the model\'s elected set. '
          f'By rate, {lead}, while the largest share of the elected set goes to the MLP '
          '`down_proj` on the two dense models and to the hybrid model\'s linear-attention '
          '`in_proj_qkv`. No kind is exempt and none is anywhere near saturated: the most '
          f'enriched projection in the panel is `{peak[1][0]}` on {peak[0]["label"]} at '
          f'{peak[1][1]["per_million"]:.0f} per million, one tile in '
          f'{round(1e6 / peak[1][1]["per_million"]):,}.', '',
          '| Depth quarter | ' + ' | '.join(s['label'] for s in S) + ' |',
          '|---' * (1 + len(S)) + '|']
    for i, name in enumerate(('first', 'second', 'third', 'last')):
        L.append(f'| {name} | ' + ' | '.join(
            f'{s["by_quarter"][i]["count"]:,} ({100 * s["by_quarter"][i]["share"]:.0f}%)'
            for s in S) + ' |')
    L += ['', 'Depth is the strongest single predictor: about half of every model\'s elected '
          'tiles are in its last quarter of layers ('
          + ', '.join(f'{100 * s["by_quarter"][3]["share"]:.0f}%' for s in S)
          + '), and the last layer alone holds '
          + ', '.join(f'{s["by_layer"][str(s["n_layers"] - 1)]:,} of {s["selected"]:,} on '
                      f'{s["label"]}' for s in S)
          + '. The first layers are the secondary peak, so the profile is a shallow U with a '
          'much heavier top end, not a monotone trend.', '',
          '### Position inside a matrix', '',
          'The reference for every row below is a seeded uniform placement that keeps each '
          'matrix\'s elected count and its tile grid and randomises only the positions, so a '
          'departure is structure the rule found rather than an artefact of where the tiles are.',
          '', '| Statistic | ' + ' | '.join(f'{s["label"]} obs. | unif.' for s in S) + ' |',
          '|---' * (1 + 2 * len(S)) + '|']
    worst = max((dict(s['worst_column'], label=s['label']) for s in S), key=lambda w: w['count'])
    rowworst = max((dict(s['worst_row'], label=s['label']) for s in S if s['worst_row']),
                   key=lambda w: w['fill'])
    rows = [('Most tiles in one row band (output channels)', 'max_row'),
            ('Most tiles in one column band (input channels)', 'max_col'),
            ('Most tiles on one column index, one projection kind', 'max_tile_recur'),
            ('Most matrices sharing a column index, one kind', 'max_mat_recur'),
            ('Most tiles on one residual channel band', 'max_res_tiles'),
            ('Most residual-reading matrices sharing that band', 'max_res_mats')]
    for label, key in rows:
        cells = []
        for s in S:
            cells += [f'{s["geometry"][key]:,}', f'{s["geometry"]["mc"][key]:.0f}']
        L.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    L += ['', 'Both axes are clustered well beyond chance on every model, in two different '
          'shapes. Along the input axis, one 64-channel band collects tiles from many output '
          f'rows: {worst["count"]} of the {worst["selected"]} tiles elected in `{worst["name"]}` '
          f'on {worst["label"]} share the single input band '
          f'{worst["channels"][0]}–{worst["channels"][1]} while spreading over '
          f'{worst["distinct_rows"]} of {worst["grid"][0]} output bands. Along the output axis the '
          'pattern is the mirror image — one row band, meaning one group of eight output '
          f'channels, elected across much of its own row: {rowworst["count"]} of the '
          f'{rowworst["grid"][1]} column bands in row {rowworst["row"]} of `{rowworst["name"]}` on '
          f'{rowworst["label"]} are elected, {100 * rowworst["count"] / rowworst["selected"]:.0f}% '
          'of everything that matrix contributes.', '',
          'The clustering also runs across layers, not only inside a matrix. For the matrices that '
          'read the residual stream directly (`q/k/v_proj`, `gate/up_proj`, the hybrid model\'s '
          '`in_proj_*`) a column index means the same hidden channels in every layer, and the '
          'elected tiles pile onto a few such bands:', '',
          '| Model | Hidden channels | Elected tiles there | Residual-reading matrices hit | '
          'Uniform placement |', '|---|---|---:|---:|---:|']
    for s in S:
        h = s['hot_residual'][0]
        L.append(f'| {s["label"]} | {h["channels"][0]}–{h["channels"][1]} | {h["tiles"]} | '
                 f'{h["matrices"]}/{s["geometry"]["residual_matrices"]} | '
                 f'{s["geometry"]["mc"]["max_res_mats"]:.0f} matrices |')
    L += ['', 'On the two Qwen models this is the clearest structure in the whole map: one '
          '64-channel band of the residual stream is elected in '
          + ' and '.join(f'{s["hot_residual"][0]["matrices"]} of '
                          f'{s["geometry"]["residual_matrices"]} matrices on {s["label"]}'
                          for s in S[1:])
          + f', against {S[1]["geometry"]["mc"]["max_res_mats"]:.0f} and '
          f'{S[2]["geometry"]["mc"]["max_res_mats"]:.0f} under uniform placement. On '
          f'{S[0]["label"]} the cross-layer version is much weaker '
          f'({S[0]["hot_residual"][0]["matrices"]} against '
          f'{S[0]["geometry"]["mc"]["max_res_mats"]:.0f}) and the concentration lives inside '
          'single matrices instead. Either way the elected set is a property of particular input '
          'channels rather than of particular matrices. This report does not establish the '
          'mechanism; it is consistent with the known concentration of activation magnitude in a '
          'small number of channels, which is exactly where a uniform grid and a log-spaced grid '
          'differ most.', '', '### Which objective binds', '',
          'Both objectives must clear the bar, but the one that decides is model dependent: '
          + ', '.join(f'{s["label"]} {100 * s["binding"]["ce_share"]:.0f}% CE' for s in S)
          + '. Median t statistics at elected tiles are around '
          + ', '.join(f'{s["t"]["ce_median"]:.1f}/{s["t"]["kl_median"]:.1f}' for s in S)
          + ' (CE/KL), so the elected set is not sitting on the threshold — it clears it '
          'comfortably on both. Neither objective is redundant: each is the binding constraint for a '
          'substantial share of the elected tiles on all three models.', '']
    if W:
        L += weights_block(S, W)
    L += [f'[Per-model tables, top matrices and hot column indices]({job.as_posix()}/REPORT.md)', '']
    return L


def weights_block(S, W):
    L = ['### Would weight MSE have found them?', '',
         'The selector is a task loss because weight error is the wrong objective; the localized '
         'map makes that checkable. For every tile, compare the squared weight error of the '
         'canonical FourOverSix E2M1 candidate with the E0M3 candidate the rule would switch to.',
         '', '| Model | MSE prefers E0M3, all tiles | MSE prefers E0M3, elected tiles | '
         'Elected among the top-N tiles by MSE gain |', '|---|---:|---:|---:|']
    for s in S:
        w = W[s['model']]
        L.append(f'| {s["label"]} | {100 * w["base_rate"]:.1f}% | {100 * w["elected_rate"]:.1f}% | '
                 f'{w["top_n_overlap"]["elected"]:,} of {w["top_n_overlap"]["n"]:,} |')
    L += ['', 'The two criteria are close to unrelated. Weight MSE prefers E0M3 for a large '
          'minority of all tiles, so it cannot be used as a filter at this count, and the tiles it '
          'ranks highest are almost never the ones the task gradient elects. The elected set is '
          'therefore not a subset that MSE could have produced more cheaply.', '']
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True, help='results/kse_selection/job_<id>')
    ap.add_argument('--weights', help='results/kse_selection/weights_<id> (one or more, comma sep)')
    ap.add_argument('--update-report', action='store_true')
    args = ap.parse_args()
    job = Path(args.job)
    S = []
    for m in ORDER:
        r = json.loads((job / f'{m}.json').read_text())
        assert r['status'] == 'complete' and r['election_reproduced'], m
        S.append(stats(r))
    (job / 'summary.json').write_text(json.dumps(S, indent=1) + '\n')

    L = ['# Where the k=3 rule puts its E0M3 tiles', '',
         f'Rebuilt from the frozen calibration scores in `{job.as_posix()}`; every model\'s '
         'per-k counts reproduce the published election exactly.', ''] + tables(S)
    for s in S:
        g, mc = s['geometry'], s['geometry']['mc']
        L += [f'## {s["label"]}', '',
              f'{s["selected"]:,} elected tiles in {s["modules_hit"]} of {s["modules"]} matrices; '
              f'the top five matrices hold {100 * s["top5_share"]:.1f}% and '
              f'{s["single_tile_modules"]} matrices hold exactly one.', '',
              '| Matrix | Grid (rows x cols) | Tiles | Elected | Share |',
              '|---|---|---:|---:|---:|']
        for m in s['top_modules']:
            L.append(f'| `{m["name"]}` | {m["grid"][0]} x {m["grid"][1]} | {m["tiles"]:,} | '
                     f'{m["selected"]:,} | {100 * m["share"]:.1f}% |')
        L += ['', '| Geometry statistic | Observed | Uniform placement |', '|---|---:|---:|',
              f'| Distinct row bands touched | {g["rows_used"]:,} | {mc["rows_used"]:.0f} |',
              f'| Distinct column bands touched | {g["cols_used"]:,} | {mc["cols_used"]:.0f} |',
              f'| Most tiles in one row band | {g["max_row"]:,} | {mc["max_row"]:.1f} |',
              f'| Most tiles in one column band | {g["max_col"]:,} | {mc["max_col"]:.1f} |',
              f'| Most tiles on one column index, one kind | {g["max_tile_recur"]:,} | '
              f'{mc["max_tile_recur"]:.1f} |',
              f'| Most matrices sharing a column index, one kind | {g["max_mat_recur"]:,} | '
              f'{mc["max_mat_recur"]:.1f} |',
              f'| Most tiles on one residual channel band | {g["max_res_tiles"]:,} | '
              f'{mc["max_res_tiles"]:.1f} |',
              f'| Most residual-reading matrices sharing that band | {g["max_res_mats"]:,} | '
              f'{mc["max_res_mats"]:.1f} |', '',
              f'{g["residual_tiles"]:,} of the elected tiles sit in the {g["residual_matrices"]} '
              'matrices that read the residual stream, where a column index is the same hidden '
              'channel band in every layer:', '',
              '| Hidden channels | Tiles | Matrices |', '|---|---:|---:|']
        for h in s['hot_residual']:
            L.append(f'| {h["channels"][0]}–{h["channels"][1]} | {h["tiles"]} | {h["matrices"]} |')
        L += ['', f'Binding objective: CE for {s["binding"]["ce"]:,} tiles '
              f'({100 * s["binding"]["ce_share"]:.1f}%), KL for {s["binding"]["kl"]:,}. Median t at '
              f'elected tiles: CE {s["t"]["ce_median"]:.2f}, KL {s["t"]["kl_median"]:.2f}.', '',
              '| Kind | Column index | Tiles |', '|---|---:|---:|']
        for h in s['hot_columns']:
            L.append(f'| `{h["kind"]}` (of {h["cols"]}) | {h["col"]} | {h["hits"]} |')
        L.append('')
    (job / 'REPORT.md').write_text('\n'.join(L) + '\n')

    W = {}
    for d in (args.weights or '').split(','):
        if d:
            for f in sorted(Path(d).glob('*.json')):
                w = json.loads(f.read_text())
                assert w['status'] == 'complete'
                W[w['model']] = w
    assert not W or set(W) == set(ORDER), sorted(W)
    section = report_section(S, job, W)
    (job / 'SECTION.md').write_text('\n'.join(section) + '\n')
    if args.update_report:
        path = Path('MIXFP4_REPORT.md')
        text = path.read_text()
        block = START + '\n' + '\n'.join(section) + END
        assert text.count(START) == text.count(END) == 1
        text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
        path.write_text(text)
        print(f'updated {path}')
    print('\n'.join(section))


if __name__ == '__main__':
    main()
