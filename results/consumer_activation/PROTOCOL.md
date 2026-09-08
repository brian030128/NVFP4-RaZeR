# Consumer-aware activation formats — third direction

Declared after asymmetric array 331893. Conditional versus dynamic format cost
improved all six point estimates there, but asymmetric calibration still harmed
plain FourOverSix on Llama math/code, and its weight-centering ablation was often
better. Do not promote that pipeline as a general solution or choose an ablation
by domain. Test a separate runtime mechanism with fixed weights instead.

## Rule and limits

Freeze all non-head linear weights at canonical FourOverSix. Offline, compute
d_j=sum_i Wq_ij^2 from each consuming weight matrix and normalize by its mean.
No calibration data. Runtime inputs produce the same fixed E2M1 FourOverSix and
E0M3 alpha=1 candidates, with scale groups of 16 and format tiles of 16x64.
Choose E0M3 iff its sum of d_j-weighted squared activation errors is smaller in
that tile. Pad trailing token rows with zeros for election; remove padding from
the output. Biases/other components remain unchanged.

This is the diagonal approximation to actual layer-output error, not a spectral
minimax objective or a guarantee. Cross-channel interactions are omitted. The
failed conservative activation guard is NOT used. No tuning parameter, format
quota, loss gate, runtime gradient, teacher pass or feedback from labels occurs.
Per-channel output-importance weighting has prior art; novelty is not claimed.
The question is whether this cheap fixed rule gives useful task transfer.

## Panel and evaluation

Llama-3.2-1B-Instruct and OPT-350M remain development architectures. Add
Qwen3-0.6B as a third model/architecture in this particular continuation. Record
remote revisions before loading evaluation data. All models use native
Transformers 4.57.3 and eager attention. Metadata is frozen before reading text.

Fresh examples: Wiki test windows 64 through 95 at 512 tokens; GSM8K and MBPP
test examples 80 through 111, truncated to 512 tokens, using the same pinned
revisions as previous stages. These do not overlap the prior two stages' windows
or example indices. They are not claimed independent of all historical repo
experiments, nor representative generated-task benchmarks.

Compare fixed FourOverSix activations, ordinary activation-MSE mixed formats,
and the consumer-weighted rule, on identical fixed W4 weights. Evaluate every
policy on all domains, with no per-model/domain choice. Save token and metadata
hashes, per-example NLL, PPL, paired differences ± two SE, selected tile counts,
metadata bytes and reference runtime. Unfused timing is not a kernel claim.

## Screening criterion

Promising means improvements of at least 0.01 PPL against FourOverSix in at
least seven of nine model/domain cells, no supported harm, and improvement
against activation MSE in at least six point estimates. Failures remain in the
record; do not tune margins, channel weights or calibration sets. A pass still
needs more models/seeds, task decoding, hardware cost, and prior-art comparison.
