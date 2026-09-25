"""Chunked per-sequence CE/KL for run_multiround.py (--chunked-loss): no FP32 [batch, T, vocab] tensor for the batch.

run_multiround.py computes, per batch, lp = logits[:, :-1].float().log_softmax(-1), the FP32 teacher t, and
    ce = nll_loss(lp, next tokens, reduction='none').mean(-1),   kl = (t.exp() * (t - lp)).sum(-1).mean(-1),
and in scoring kl.sum().backward(). At batch 16 with Qwen3.8-27B's 248,320-token vocabulary every FP32 tensor of
that expression is 7.56 GiB. Here the same operations run over chunks of whole documents. The per-token values
(log-softmax rows, the CE gather, the KL row sums) are row-wise, and the per-sequence means are taken on the full
[batch, T-1] per-token tensor exactly as before.

Chunks hold at least two documents (the last absorbs an odd one). The KL row sum is a CUDA reduction whose launch
configuration depends on the number of rows: with few rows PyTorch splits each row across two thread blocks, which
changes the summation order. Two 512-token documents (1,022 rows) already exceed that threshold on this GPU, as the
full batch does, so every chunk reduces its rows exactly as the whole batch would. The bitwise unit tests check this.

Scoring: each chunk's loss graph is differentiated with torch.autograd.grad against a bf16 leaf holding the chunk's
logits. The chunk loss is kl_tok.mean(-1).sum(), so every token's upstream gradient is the same fl(1/(T-1)) that
the batch's MeanBackward produces. The chunk gradients are copied, not added, into one bf16 buffer for the logits
(last position zero, as SliceBackward leaves it), and the model's backward then starts from logits.backward(buffer).
The lm_head GEMM and everything before it are unchanged.
"""
import torch
import torch.nn.functional as F


def document_chunks(batch, per_chunk=2):
    """[(d0, d1)]: consecutive document ranges of per_chunk documents each; the last one absorbs a remainder."""
    if batch <= per_chunk:
        return [(0, batch)]
    ranges = [(d, d + per_chunk) for d in range(0, batch - batch % per_chunk, per_chunk)]
    if batch % per_chunk:
        ranges[-1] = (ranges[-1][0], batch)
    return ranges


@torch.no_grad()
def per_sequence_losses(logits, ids, teacher, device, per_chunk=2):
    """(ce, kl) per sequence, bitwise those of run_multiround's per_sequence_losses on the whole batch.

    logits: (B, T, V) bf16 model output; ids: (B, T); teacher: B CPU bf16 (1, T-1, V) log-probability rows."""
    b, t = ids.shape[0], ids.shape[1] - 1
    ce_tok = torch.empty(b, t, dtype=torch.float32, device=logits.device)
    kl_tok = torch.empty(b, t, dtype=torch.float32, device=logits.device)
    for d0, d1 in document_chunks(b, per_chunk):
        lp = logits[d0:d1, :-1].float().log_softmax(-1)
        tt = torch.cat(teacher[d0:d1]).to(device).float()
        ce_tok[d0:d1] = F.nll_loss(lp.transpose(1, 2), ids[d0:d1, 1:].to(lp.device), reduction='none')
        kl_tok[d0:d1] = (tt.exp() * (tt - lp)).sum(-1)
        del lp, tt
    return ce_tok.mean(-1), kl_tok.mean(-1)


def kl_logit_gradient(logits, teacher, device, per_chunk=2):
    """d(kl.sum())/d logits of run_multiround's scoring, bitwise, one chunk of documents at a time.

    logits: (B, T, V) bf16 (requires grad; not modified); returns a (B, T, V) bf16 gradient buffer for
    logits.backward(). Under torch.enable_grad()."""
    grad = torch.zeros_like(logits, requires_grad=False)
    source = logits.detach()
    for d0, d1 in document_chunks(logits.shape[0], per_chunk):
        leaf = source[d0:d1].requires_grad_()
        lp = leaf[:, :-1].float().log_softmax(-1)
        tt = torch.cat(teacher[d0:d1]).to(device).float()
        loss = (tt.exp() * (tt - lp)).sum(-1).mean(-1).sum()
        g, = torch.autograd.grad(loss, leaf)
        grad[d0:d1].copy_(g)
        del leaf, lp, tt, loss, g
    return grad
