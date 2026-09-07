"""Describe domain agreement in the fitted tile scores; run in Slurm."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Submit even CPU analysis through Slurm.')
import argparse
import json
import math
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='results/task_sensitivity_domains')
ap.add_argument('--partial', action='store_true')
ap.add_argument('--prefix32', action='store_true')
args = ap.parse_args()
torch.set_num_threads(min(2, int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))))
root = Path(args.root)

def load(path):
    d = torch.load(path, weights_only=True, map_location='cpu')
    return d['means'], d['ses']

def combine(a, b):
    means = {k: (a[0][k].double()+b[0][k].double())/2 for k in a[0]}
    ses = {k: ((32*31*(a[1][k].double().square()+b[1][k].double().square())
                + 16*(a[0][k].double()-b[0][k].double()).square())/(64*63)).sqrt() for k in means}
    return means, ses

def cosine(a, b):
    dot = aa = bb = 0.
    for n in a:
        x, y = a[n].double(), b[n].double()
        dot += float((x*y).sum())
        aa += float(x.square().sum())
        bb += float(y.square().sum())
    return dot/math.sqrt(aa*bb) if aa*bb else None

out = {}
paths = [root/f'seed{s}' for s in [20260912, 20260913]]
paths += [root/'panel'/m for m in ['qwen3-4b', 'llama-3.1-8b-local']]
paths += [root/'teacher'/m for m in ['qwen3-4b', 'llama-3.1-8b-local']]
for path in paths:
    report = path/'report.json'
    if not report.exists():
        assert args.partial, report
        continue
    r = json.loads(report.read_text())
    is_panel = path.parent.name == 'panel'
    is_teacher = path.parent.name == 'teacher'
    use_prefix = args.prefix32 or is_teacher
    cfiles = ([path/'c4_scores.pt'] if is_teacher else
              [path/('ca_scores.pt' if is_panel else 'c4a_scores.pt'),
               path/('cb_scores.pt' if is_panel else 'c4b_scores.pt')])
    if use_prefix:
        cfiles = cfiles[:1]
    if not all(p.exists() for p in cfiles):
        assert args.partial, cfiles
        continue
    if is_teacher:
        wiki = load(path/'wiki_scores.pt')
    elif use_prefix:
        wiki = load(path/('wa_scores.pt' if is_panel else 'wiki32_scores.pt'))
    elif is_panel:
        wiki = combine(load(path/'wa_scores.pt'), load(path/'wb_scores.pt'))
    else:
        p = path/'wiki64_scores.pt'
        if not p.exists():
            p = Path(f'results/task_sensitivity_four_over_six/seed{r["seed"]}/scores.pt')
        wiki = load(p)
    c4_split_cosine = None
    if use_prefix:
        c4 = load(cfiles[0])
    else:
        halves = [load(p) for p in cfiles]
        c4_split_cosine = cosine(halves[0][0], halves[1][0])
        c4 = combine(*halves)
        del halves
    counts = dict(tiles=0, same_point_sign=0, both_stably_negative=0,
                  wiki_stably_negative=0, c4_stably_negative=0,
                  wiki_good_c4_bad=0, c4_good_wiki_bad=0)
    xy = xx = yy = 0.
    noise_x = noise_y = 0.
    per_module = {}
    for n in wiki[0]:
        xn = wiki[0][n]+2*wiki[1][n] < 0
        yn = c4[0][n]+2*c4[1][n] < 0
        x, sx, y, sy = [v.double() for v in (wiki[0][n], wiki[1][n], c4[0][n], c4[1][n])]
        counts['tiles'] += x.numel()
        counts['same_point_sign'] += int(((x < 0) == (y < 0)).sum())
        counts['both_stably_negative'] += int((xn & yn).sum())
        counts['wiki_stably_negative'] += int(xn.sum())
        counts['c4_stably_negative'] += int(yn.sum())
        counts['wiki_good_c4_bad'] += int((xn & (y-2*sy > 0)).sum())
        counts['c4_good_wiki_bad'] += int((yn & (x-2*sx > 0)).sum())
        dot, nx, ny = float((x*y).sum()), float(x.square().sum()), float(y.square().sum())
        xy, xx, yy = xy+dot, xx+nx, yy+ny
        noise_x += float(sx.square().sum())
        noise_y += float(sy.square().sum())
        per_module[n] = {'cosine': dot/math.sqrt(nx*ny) if nx*ny else None,
                         'wiki_score_l2_squared': nx, 'c4_score_l2_squared': ny,
                         'joint_negative_tiles': int((xn & yn).sum())}
    result = {**counts, 'score_cosine': xy/math.sqrt(xx*yy) if xx*yy else None, 'per_module': per_module,
              'windows_per_domain': 32 if use_prefix else 64,
              'score_objective': 'teacher_kl' if is_teacher else 'observed_nll',
              'c4_split_half_cosine': c4_split_cosine,
              'squared_se_over_squared_mean': {'wiki': noise_x/xx if xx else None,
                                                'c4': noise_y/yy if yy else None},
              'interpretation': 'Fitting-score agreement, not held-out loss prediction or a simultaneous significance test.'}
    map_path = (Path(r['parent_report']).with_name('wiki_candidate.json') if is_teacher else
                path/'wiki_candidate.json' if is_panel else
                Path(f'results/task_sensitivity_four_over_six/seed{r["seed"]}/candidate_type_map.json'))
    if map_path.exists():
        spec = json.loads(map_path.read_text())
        result['reference_wiki_map_predicted_loss'] = {}
        for label, scores in [('wiki', wiki), ('c4', c4)]:
            result['reference_wiki_map_predicted_loss'][label] = sum(
                float(scores[0][n].flatten()[entry['e0m3_flat_indices']].double().sum())
                for n, entry in spec['modules'].items())
        if not is_teacher:
            result['reference_wiki_map_predicted_nll'] = result['reference_wiki_map_predicted_loss']
        result['forecast_scope'] = 'Sum of tile gradient means in score-objective units; not a measured NLL change or a whole-map confidence interval.'
    out[str(path.relative_to(root))] = result
    print(str(path), {k: v for k, v in result.items() if k not in ('per_module', 'interpretation')}, flush=True)
    del wiki, c4
with (root/('score_agreement_prefix32.json' if args.prefix32 else 'score_agreement.json')).open('w') as f:
    json.dump(out, f, indent=2, allow_nan=False)
