"""Nested, task-aware search of one foldable MLP intermediate permutation.

Items are intact 16-channel groups. Four items share a down-projection K64
tile; four such groups share a gate/up N256 tile. One ordering serves all three
matrices, so SiLU/product commute and no activation gather or output scatter
is needed between these projections. This is an offline search, not a kernel.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from quantize.task_reorder import _balanced_assign, _sum_axis


@dataclass(frozen=True)
class SharedConfig:
    starts: int = 8
    rounds: int = 12
    swap_samples: int = 4096
    swap_passes: int = 2
    k: float = 3.
    validation_k: float = 1.
    seed: int = 0
    domain_robust: bool = False
    scratch_mb: int = 128


def aggregate_atoms(ce, kl, projection):
    """[sequence,row,K16] -> [intermediate16,other_type_tile,objective*sequence]."""
    if ce.shape != kl.shape or ce.ndim != 3:
        raise ValueError('Expected paired fine score tensors')
    items = []
    for score in (ce, kl):
        s, n, c = score.shape
        if projection in ('gate_proj', 'up_proj'):
            if n % 256 or c % 4:
                raise ValueError('Shared MLP pilot requires full 256x64 tiles')
            value = score.double().reshape(s, n // 16, 16, c // 4, 4).sum((2, 4))
        elif projection == 'down_proj':
            if n % 256 or c % 16:
                raise ValueError('Intermediate axis must contain complete 256-channel groups')
            value = score.double().reshape(s, n // 256, 256, c).sum(2).transpose(1, 2)
        else:
            raise ValueError(projection)
        items.append(value)
    return torch.stack(items).permute(2, 3, 0, 1).contiguous().flatten(2)


def select_sequences(features, indices):
    return features.reshape(*features.shape[:2], 2, -1)[..., indices].flatten(2)


def bounds(phi, k, scales, domain_robust=False):
    values = phi.reshape(*phi.shape[:-1], 2, -1)
    if domain_robust:
        if values.shape[-1] % 2 or values.shape[-1] < 4:
            raise ValueError('Need equal math/code halves, each with >=2 sequences')
        values = values.reshape(*values.shape[:-1], 2, values.shape[-1] // 2)
        upper = (values.mean(-1) + k * values.std(-1) / math.sqrt(values.shape[-1])).amax(-1)
    else:
        upper = values.mean(-1) + k * values.std(-1) / math.sqrt(values.shape[-1])
    return upper / scales


def values(phi, config, scale, temperature=0.):
    upper = bounds(phi, config.k, scale, config.domain_robust)
    if temperature == 0:
        return (-upper.amax(-1)).clamp_min(0)
    maximum = temperature * torch.logsumexp(upper / temperature, -1)
    return temperature * F.softplus(-maximum / temperature)


def phis(features, labels):
    return {name: _sum_axis(x, labels // (1 if name == 'down_proj' else 4), 0)
            for name, x in features.items()}


def objective(features, labels, scales, config, temperature=0.):
    return sum(float(values(phi, config, scales[name], temperature).sum())
               for name, phi in phis(features, labels).items())


def profits(features, labels, scales, config, temperature):
    result = None
    for name, phi in phis(features, labels).items():
        with torch.enable_grad():
            p = phi.detach().requires_grad_(True)
            gradient = torch.autograd.grad(values(p, config, scales[name], temperature).sum(), p)[0]
        profit = features[name].flatten(1) @ gradient.flatten(1).T
        if name != 'down_proj':
            profit = profit.repeat_interleave(4, 1)
        result = profit if result is None else result + profit
    return result


def exchange(features, labels, scales, config, temperature, generator):
    n = labels.numel()
    groups = n // 4
    i = torch.randint(n, (config.swap_samples,), generator=generator)
    j = torch.randint(n, (config.swap_samples,), generator=generator)
    profit = profits(features, labels, scales, config, temperature)
    profit[torch.arange(n), labels] = -torch.inf
    half = len(i) // 2
    destination = profit[i[:half]].argmax(-1)
    members = torch.argsort(labels, stable=True)
    j[:half] = members[4 * destination + torch.randint(4, (half,), generator=generator)]
    keep = labels[i] != labels[j]
    i, j = i[keep], j[keep]
    gain = torch.zeros(len(i), dtype=torch.float64)
    for name, phi in phis(features, labels).items():
        x = features[name]
        local = labels // (1 if name == 'down_proj' else 4)
        a, b = local[i], local[j]
        score = lambda p: values(p, config, scales[name], temperature).sum(-1)
        old = score(phi)
        batch = max(1, config.scratch_mb * 1024**2 // (8 * x[0].numel() * 12))
        for start in range(0, len(i), batch):
            sl = slice(start, start + batch)
            delta = x[j[sl]] - x[i[sl]]
            change = score(phi[a[sl]] + delta) + score(phi[b[sl]] - delta) - old[a[sl]] - old[b[sl]]
            gain[sl] += torch.where(a[sl] == b[sl], 0., change)
    candidates = torch.nonzero(gain > 1e-10).flatten()
    candidates = candidates[torch.argsort(gain[candidates], descending=True, stable=True)]
    result, touched = labels.clone(), set()
    for index in candidates.tolist():
        # Disjoint N256 parents imply disjoint affected tiles in all matrices.
        parents = {int(labels[i[index]]) // 4, int(labels[j[index]]) // 4}
        if parents & touched:
            continue
        result[i[index]], result[j[index]] = labels[j[index]], labels[i[index]]
        touched.update(parents)
    return result


def validate_candidate(train, validation, labels, scales, config):
    """Score the TRAIN-elected joint switch on unseen inner sequences.

    Do not re-elect on validation: that would reward a new favorable subset.
    Sum signed directional effects across all elected tiles and all modules,
    preserving sequence covariance, before taking CE/KL confidence bounds.
    """
    train_phi, val_phi = phis(train, labels), phis(validation, labels)
    total = None
    tiles = 0
    for name in train:
        mask = bounds(train_phi[name], config.k, scales[name], config.domain_robust).amax(-1) < 0
        tiles += int(mask.sum())
        effect = val_phi[name][mask].sum(0).reshape(2, -1)
        total = effect if total is None else total + effect
    # Fixed objective scales; raw effects across matrices have the same CE/KL
    # units, so add first rather than normalizing each module's contribution.
    joint_scale = torch.stack(list(scales.values())).square().mean(0).sqrt()
    upper = bounds(total.flatten(), config.validation_k, joint_scale, config.domain_robust)
    return dict(score=-float(upper.max()), train_tiles=tiles,
                normalized_validation_upper=upper.tolist(),
                validation_mean_ce=float(total[0].mean()), validation_mean_kl=float(total[1].mean()))


@torch.no_grad()
def search_shared(train, validation, config=None, progress=None):
    config = config or SharedConfig()
    if set(train) != {'gate_proj', 'up_proj', 'down_proj'} or set(validation) != set(train):
        raise ValueError('Need the three matched MLP projections')
    n = train['gate_proj'].shape[0]
    if n % 16 or any(x.shape[0] != n or x.shape[2] < 4 or x.shape[2] % 2
                      or not torch.isfinite(x).all() for x in [*train.values(), *validation.values()]):
        raise ValueError('Invalid shared features')
    scales = {name: x.reshape(*x.shape[:2], 2, -1).mean(-1).square().mean((0, 1)).sqrt().clamp_min(1e-30)
              for name, x in train.items()}
    identity = torch.arange(n) // 4
    baseline = objective(train, identity, scales, config)
    selected = validate_candidate(train, validation, identity, scales, config)
    best_labels = identity.clone()
    candidates = [dict(start='identity', temperature=None, fit_objective=baseline, **selected)]
    descriptors = torch.cat([(x.reshape(*x.shape[:2], 2, -1).mean(-1) / scales[name]).flatten(1)
                             for name, x in train.items()], 1)
    generator = torch.Generator().manual_seed(config.seed)
    orders = [('identity', torch.arange(n)), ('marginal', torch.argsort(descriptors.sum(1), stable=True))]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(config.seed)
        u, s, _ = torch.svd_lowrank(descriptors, q=2, niter=3)
    orders += [('spectral1', torch.argsort(u[:, 0], stable=True)),
               ('spectral2', torch.argsort(torch.atan2(u[:, 1] * s[1], u[:, 0] * s[0]), stable=True))]
    orders += [(f'random{i}', torch.randperm(n, generator=generator)) for i in range(config.starts)]
    initial_bounds = torch.cat([bounds(p, config.k, scales[name], config.domain_robust).flatten()
                                for name, p in phis(train, identity).items()])
    temperature = max(float(initial_bounds.abs().median()), 1e-8)
    starts = []
    for name, order in orders:
        labels = torch.empty_like(order)
        labels[order] = torch.arange(n) // 4
        starts.append((objective(train, labels, scales, config, temperature), name, labels))
    starts = starts[:1] + sorted(starts[1:], key=lambda x: x[0], reverse=True)[:config.starts - 1]
    trace = []
    for _, name, labels in starts:
        for temp in (temperature, temperature * .25, 0.):
            for iteration in range(config.rounds):
                before = objective(train, labels, scales, config, temp)
                profit = profits(train, labels, scales, config, temp)
                proposal = _balanced_assign(profit, torch.full((n // 4,), 4))
                if objective(train, proposal, scales, config, temp) > before + 1e-10:
                    labels = proposal
                for _ in range(config.swap_passes):
                    labels = exchange(train, labels, scales, config, temp, generator)
                after = objective(train, labels, scales, config, temp)
                entry = dict(start=name, temperature=temp, iteration=iteration,
                             continuation_objective=after, fit_objective=objective(train, labels, scales, config))
                trace.append(entry)
                if progress:
                    progress(entry)
                if after <= before + 1e-10:
                    break
            result = validate_candidate(train, validation, labels, scales, config)
            candidate = dict(start=name, temperature=temp,
                             fit_objective=objective(train, labels, scales, config), **result)
            candidates.append(candidate)
            if progress:
                progress(dict(candidate_validation=candidate))
            if result['score'] > selected['score'] + 1e-12:
                selected, best_labels = result, labels.clone()
    return dict(group_perm=torch.argsort(best_labels, stable=True), selected_validation=selected,
                identity_validation=candidates[0], candidates=candidates, trace=trace,
                fit_identity_objective=baseline, fit_objective=objective(train, best_labels, scales, config),
                objective_scales=scales)
