"""Weight-only spectral solver feasibility backend; estimates are not certificates."""
import torch

from quantize.spectral_selector import _check, reconstruct


@torch.no_grad()
def estimate(error, rank=8, iterations=64):
    """Deterministic block power estimate with a Ritz residual diagnostic.

    The Ritz value is a lower bound in exact arithmetic, not an upper bound.
    A small residual alone does not certify that the largest eigenvalue was found.
    """
    e = error.float()
    rank = min(rank, min(e.shape))
    gen = torch.Generator(device=e.device).manual_seed(1729)
    q = torch.randn(e.shape[1], rank, generator=gen, device=e.device)
    q = torch.linalg.qr(q, mode='reduced').Q
    for _ in range(iterations):
        q = torch.linalg.qr(e.T @ (e @ q), mode='reduced').Q
    y = e @ q
    values, vectors = torch.linalg.eigh(y.T @ y)
    v = q @ vectors[:, -1]
    ev = e @ v
    value = ev.square().sum()
    residual = torch.linalg.vector_norm(e.T @ ev - value*v)
    return dict(value=float(value), relative_residual=float(residual/value.clamp_min(1e-30)),
                rank=rank, iterations=iterations), v


@torch.no_grad()
def select_approx(w, base, alt):
    """16-step single-proposal descent on the fixed rank-8/64-step estimate.

    Rank every tile by its exact finite effect on the current estimated worst
    direction. Evaluate the best proposal against the full matrix estimate;
    stop if it does not improve. This is not exact coordinate descent.
    """
    _check(w)
    rows, cols = w.shape[0]//8, w.shape[1]//64
    mask = torch.zeros((rows, cols), dtype=torch.bool, device=w.device)
    error = base.float()-w.float()
    delta = alt.float()-base.float()
    current, v = estimate(error)
    history = [current]
    proposals = []
    reason = 'step_cap'
    for _ in range(16):
        # delta*v per tile has eight output coordinates. Inputs elsewhere fixed.
        dv = torch.einsum('rick,ck->rci', delta.reshape(rows,8,cols,64), v.reshape(cols,64))
        dv *= torch.where(mask, -1., 1.)[:, :, None]
        ev = (error @ v).reshape(rows,8)
        scores = 2*(dv*ev[:,None,:]).sum(-1)+dv.square().sum(-1)
        index = int(scores.argmin())
        row, col = divmod(index, cols)
        score = float(scores[row,col])
        if score >= 0:
            reason = 'no_directional_improvement'
            break
        trial_mask = mask.clone()
        trial_mask[row,col] = ~trial_mask[row,col]
        trial = reconstruct(base,alt,trial_mask).float()-w.float()
        measured, trial_v = estimate(trial)
        accepted = measured['value'] < current['value']*(1-1e-6)
        proposals.append(dict(index=index, directional_delta=score, estimate=measured, accepted=accepted))
        if not accepted:
            reason = 'proposal_rejected'
            break
        mask, error, current, v = trial_mask, trial, measured, trial_v
        history.append(current)
    return mask, dict(history=history, proposals=proposals, stop_reason=reason)
