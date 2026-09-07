"""Teacher-distribution objective in nats per predicted token."""
import torch
import torch.nn.functional as F


def teacher_kl(logits, teacher_logits):
    assert logits.shape == teacher_logits.shape and logits.shape[1] >= 2
    log_q = F.log_softmax(logits[:, :-1].float(), dim=-1)
    with torch.no_grad():
        log_p = F.log_softmax(teacher_logits[:, :-1].to(logits.device).float(), dim=-1)
    return F.kl_div(log_q, log_p, log_target=True, reduction='sum') / (logits.shape[0]*(logits.shape[1]-1))
