"""flipquant's top-K teacher for run_train_map.py --teacher-topk K (results/topk_cal/PROTOCOL.md).

The definition is flipquant's (github brian030128/flipquant, calibration/train_map.py, commit c2674a9 "train_map
--teacher-topk: top-k + tail-bucket KL with the teacher on the GPU"; read at main d2dd92e):
- **The teacher:** per window, the top-K of the FP32 teacher log-probabilities. It keeps the values in BF16 and the
  token ids in int32, plus the exact log-mass of the rest of the vocabulary in FP32. All of it stays on the GPU.
- **The loss:** KL(teacher || student) over those K tokens plus one bucket for the rest. That is the exact KL of the
  coarse-grained distributions, a lower bound on the full KL.

tail_logprob, kl_topk_per_sequence, and topk_rows / batch_rows (TopKTeacher.append_logprobs / .batch) are flipquant's
expressions verbatim. kl_topk_tokens is kl_topk_per_sequence without its final token mean. It lets chunked_loss take
the per-sequence mean on the whole [B, T-1] per-token tensor, as it does for the full KL.
repro_local/realquant/test_topk_teacher.py checks these functions against flipquant's source.
"""
import torch


def tail_logprob(lp, idx):
    """log of the probability mass outside ``idx`` (exact: logsumexp over the masked remainder). The mask is a
    large finite negative, not -inf, so an empty remainder gives ~-1e30 with a finite gradient."""
    return lp.scatter(-1, idx, -1e30).logsumexp(-1)


def kl_topk_tokens(lp, values, idx, tail):
    """Per token: KL(teacher || student) over the teacher's top-k tokens plus one bucket for the rest of the
    vocabulary. ``values`` [B,T,k] and ``tail`` [B,T] are teacher log-probs, ``idx`` [B,T,k] their token ids."""
    q_tail = tail_logprob(lp, idx)
    top = (values.exp() * (values - lp.gather(-1, idx))).sum(-1)
    rest = torch.where(torch.isfinite(tail), tail.exp() * (tail - q_tail), torch.zeros_like(tail))
    return top + rest


def kl_topk_per_sequence(lp, values, idx, tail):
    """flipquant's kl_topk_per_sequence: the token mean of kl_topk_tokens, per sequence."""
    return kl_topk_tokens(lp, values, idx, tail).mean(-1)


def topk_rows(lp, k):
    """flipquant's TopKTeacher.append_logprobs for one window's FP32 teacher log-probabilities ``lp`` [1,T,V]:
    ((values bf16, ids int32, tail fp32) on lp's device, the window's mean tail mass)."""
    values, idx = lp.topk(k, dim=-1)
    tail = tail_logprob(lp, idx)
    return (values.bfloat16(), idx.int(), tail), float(tail.exp().mean())


def batch_rows(rows, device):
    """flipquant's TopKTeacher.batch: the windows' rows concatenated in order, as (values fp32, ids int64, tail)."""
    values, idx, tail = (torch.cat([r[j] for r in rows]).to(device) for j in range(3))
    return values.float(), idx.long(), tail
