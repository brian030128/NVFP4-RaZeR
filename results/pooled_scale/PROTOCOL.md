# Frozen 4B/8B scale-transfer test

The five-model causal replay332374 passed its causal-robustness screen. It
improved all15 baseline comparisons,14 supported by descriptive2SE, with no
supported harm;14/15 retained at least half the previous-convention gain.
All five models passed exact prefix-independence interventions. These facts
do not reverse the failed source-diversity comparison in332349.

Now test Qwen/Qwen3-4B and the pinned local Meta-Llama-3.1-8B BASE checkpoint
d04e592bb4f6aa9cfee91e2e20afa771667e1d4b. These extend model size; they do not
constitute new architecture families or newly uninspected evaluation domains.

Keep the algorithm unchanged: one shared64 C4 +64 OpenWebMath +64 CodeParrot
table,512tokens per sequence; same teacher-KL/CE directional scores,2SE,
fixed FourOverSix/E0M3-alpha1 candidates,8x64 tiles, and256 cap. As in the
earlier procedure, scoring uses window-wide activation factors. Freeze the
map, then evaluate it under the per-token convention from the causal audit.
No row-specific recalibration, map adjustment, budget search or acceptance
gate. This tests the same complete scoring-and-causal-replay procedure at scale.

Use the same literature/science/government data recipe and recorded revisions
from the preceding confirmation:64 distinct sufficiently long test documents,
one512-token crop, seed20260925 per model/domain; exclude exact calibration
document hashes. Data-family losses have been inspected already, but no
outcome from these larger models chooses any setting.

Replay all controls from the same score table: FourOverSix, weight-MSE,
C4-only64, mixed64, pooled192. All evaluation uses per-token FP32 factors.
For both FourOverSix and the selected map, first128 logits must remain
bitwise identical when only the last384 input tokens change. Both mean
fitting CE and KL are audited under this same row convention after export.

Scale-transfer screen:>=0.01PPL gain over matched FourOverSix in>=5/6 cells,
no supported harm; lower point NLL than weight-MSE in>=4/6; exact prefix
independence in both models; mean fitting CE and KL both improve in both.
Report the C4-only and equal-token diversity controls without claiming that
source diversity is uniformly better. Preserve every result. No new kernel
throughput, generation accuracy, calibration-free or universal guarantee claim.
