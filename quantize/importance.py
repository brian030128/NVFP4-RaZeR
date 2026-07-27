"""
    Activation importance for quantization-error weighting.

    Why unweighted MSE is the wrong objective
    -----------------------------------------
    For a linear layer Y = X W^T, a weight perturbation dW changes the output by dY = X dW^T, so
    the expected squared output error is

        E || dY ||^2  =  tr( dW  S  dW^T ),        S = E[ x x^T ]   (input second moment)

    i.e. the error of dW measured in the S-weighted norm, ||dW||^2_S. Plain MSE is ||dW||^2_I.
    The two are related only through the spectrum of S:

        lambda_min(S) ||dW||_F^2  <=  ||dW||_S^2  <=  lambda_max(S) ||dW||_F^2

    so reducing ||dW||_F^2 by a factor (1 - g) guarantees a reduction in the quantity we care about
    ONLY IF (1 - g) * lambda_max < lambda_min, that is

        g  >  1 - 1 / kappa(S),      kappa(S) = lambda_max / lambda_min.

    LLM activations have strongly anisotropic S (outlier channels), with kappa easily 1e2-1e3. A
    Frobenius gain of a fraction of a percent -- which is what a coarse MixFP4 type block achieves
    over 4over6 -- therefore certifies nothing at all about the output error. That is exactly the
    regime the measurements landed in, and it is why an aggregate MSE win did not translate into a
    perplexity win.

    Diagonal estimate
    -----------------
    Using the full S is a per-layer d x d matrix (GPTQ/OBQ territory). The cheap and standard
    surrogate is its diagonal,

        importance_j  =  S_jj  =  E[ x_j^2 ],

    which turns the objective into a per-input-channel weighted MSE, sum_j S_jj * dW_ij^2. That is
    what `quant_mix_4_6(..., importance=...)` consumes. Sharing the same weighting across a scale
    block is exact when S is diagonal, and is a first-order approximation otherwise; the residual is
    bounded by the off-diagonal mass ||S - diag(S)||_2 * ||dW||_F^2, which is the quantity a
    conservative election margin is there to absorb.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def collect_importance(model, batches, device=None, name_filter=lambda n: "head" not in n):
    """
        Run a few calibration batches and record E[x_j^2] at the input of every nn.Linear.

        Returns {module_name: 1-D tensor of length in_features}, ready to hand to
        `quant_weight(..., importance=...)`.

        `batches` is any iterable of input_ids tensors. A handful (4-8 sequences) is plenty --
        this is a second moment per channel, not a covariance.
    """
    sums, counts, handles = {}, {}, []

    def make_hook(name):
        def hook(_module, inputs, _output):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            s = x.pow(2).sum(dim=0)
            if name in sums:
                sums[name] += s
                counts[name] += x.shape[0]
            else:
                sums[name] = s
                counts[name] = x.shape[0]
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name_filter(name):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        was_training = model.training
        model.eval()
        for batch in batches:
            if device is not None:
                batch = batch.to(device)
            model(batch)
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    return {n: (sums[n] / max(counts[n], 1)) for n in sums}


def importance_stats(importance):
    """Anisotropy of the collected importance: how far the diagonal is from uniform."""
    out = {}
    for name, imp in importance.items():
        imp = imp.float()
        out[name] = {
            "mean": imp.mean().item(),
            "max_over_mean": (imp.max() / imp.mean().clamp(min=1e-30)).item(),
            # condition number of the DIAGONAL surrogate; a lower bound on kappa(S)
            "kappa_diag": (imp.max() / imp.min().clamp(min=1e-30)).item(),
        }
    return out
