"""Fixed calibration subsets and streaming, globally ranked tile selection."""
import math
import torch
from quantize.relinearized_format import common_descent_scores


def calibration_subsets():
    result = {}
    for source, offset in [('c4', 0), ('math', 64), ('code', 128)]:
        for n in (16, 32, 64):
            result[f'{source}_{n}'] = list(range(offset, offset + n))
    for label, offsets in [('c4_math64', (0, 64)), ('c4_code64', (0, 128)),
                           ('math_code64', (64, 128))]:
        result[label] = [i for start in offsets for i in range(start, start + 32)]
    for n in (16, 32, 64, 128, 192):
        counts = [n // 3 + (i < n % 3) for i in range(3)]
        result[f'pooled{n}'] = [i for start, count in zip((0, 64, 128), counts)
                               for i in range(start, start + count)]
    return result


def select_stream(modules, subsets=None, cap=256):
    """Yield (name, CE[192,T], KL[192,T]); retain only each module's top cap.

    Local top-cap retention is exact for global top-cap ranking, including ties
    resolved by original module order and flattened tile index.
    """
    subsets = calibration_subsets() if subsets is None else subsets
    if cap < 1 or any(len(v) < 2 or len(set(v)) != len(v) or
                      min(v) < 0 or max(v) >= 192 for v in subsets.values()):
        raise ValueError('Invalid calibration subset or tile cap')
    candidates = {p: [] for p in subsets}
    eligible = {p: 0 for p in subsets}
    sizes, offset = {}, 0
    for name, ce, kl in modules:
        if name in sizes or ce.shape != kl.shape or ce.ndim != 2 or ce.shape[0] != 192:
            raise ValueError('Expected unique modules with paired 192-row scores')
        sizes[name] = ce.shape[1]
        zero = torch.zeros(ce.shape[1], dtype=torch.bool)
        for policy, indices in subsets.items():
            upper = common_descent_scores(ce[indices], kl[indices], zero)
            if not torch.isfinite(upper).all():
                raise ValueError('Nonfinite calibration score')
            eligible[policy] += int((upper < 0).sum())
            # Stable sorting preserves the original tie-breaking convention.
            order = torch.argsort(upper, stable=True)[:cap]
            order = order[upper[order] < 0]
            candidates[policy].extend((float(upper[i]), offset + int(i), name, int(i)) for i in order)
            candidates[policy] = sorted(candidates[policy])[:cap]
        offset += ce.shape[1]
    maps, statistics = {}, {}
    for policy, entries in candidates.items():
        maps[policy] = {name: [] for name in sizes}
        for score, global_index, name, local_index in entries:
            maps[policy][name].append(local_index)
        for indices in maps[policy].values():
            indices.sort()
        counts = [sum(start <= i < start + 64 for i in subsets[policy]) for start in (0, 64, 128)]
        statistics[policy] = dict(calibration_sequences=len(subsets[policy]), source_counts=counts,
            selected_blocks=len(entries), eligible_negative_score_blocks=eligible[policy],
            total_type_blocks=offset, selected_fraction=len(entries) / offset,
            selected_fraction_of_eligible=len(entries) / eligible[policy] if eligible[policy] else 0.,
            cap=cap, cap_reached=len(entries) == cap,
            last_selected_upper_score=entries[-1][0] if entries else None)
    return maps, statistics, sizes


def selection_overlap(a, b):
    left = {(n, i) for n, values in a.items() for i in values}
    right = {(n, i) for n, values in b.items() for i in values}
    union = left | right
    return dict(intersection=len(left & right), union=len(union),
                jaccard=len(left & right) / len(union) if union else 1.)


def validate_losses(values, windows):
    if len(values) != windows or windows < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite, missing, or misaligned per-window losses')
    return math.exp(sum(values) / windows)
