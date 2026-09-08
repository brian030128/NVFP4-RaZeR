# Online activation bound: first feasibility result

The proposed compact guard is not useful in its tested form: it returned the
baseline on all twelve packets. This is a failed feasibility test, not evidence
of an accurate deployed quantizer. The useful candidate changes below were
rejected and must not be presented as exported-method gains.

## Execution and controls

Slurm 331792 completed tests and the fixed experiment; 331795 summarized it.
Initial job 331788 stopped before processing prompts because a test used a
float32 identity matrix in a float64 residual comparison. Correcting that test
resolved the failure without changing the selector, bound, metadata rank, or
prompts. All recorded bound and map tests then passed.

Metadata was constructed only from complete FourOverSix q_proj weight matrices
at layers 0, 8, and 15 of pinned Llama-3.2-1B-Instruct. It was frozen before
processing the four prespecified prompts. Each prompt produced one 16-token
packet per matrix. No calibration, observed-token loss, fitting, or domain-based
configuration choice occurred. Inputs came from a W4A16 trajectory; these are
mechanism probes, not end-to-end W4A4 evaluations or generalization benchmarks.

## Proposal quality versus guard acceptance

| Quantity | Result |
|---|---:|
| Packets | 12 |
| Proposed maps reducing exact output error | 5 |
| Proposed maps increasing exact output error | 1 |
| Proposed maps unchanged from baseline | 6 |
| Guarded maps accepted | 0 |
| Largest proposed output-error reduction | 6.44% |
| Proposal beats activation-MSE comparator | 6/12 |
| Observed numerical bound violations | 0 |

Ordinary activation-MSE selection left eleven packets unchanged and harmed one
packet's output error. The output-aware proposal reduced error by 6.44% on the
first-layer prose packet and 5.88% on the first-layer multilingual packet. One
multilingual middle-layer proposal increased error by 0.19%. Thus the proposal
alone is not a guaranteed improvement and must not silently replace the guard.

The exact restricted oracle enumerated sixteen maps per packet over four fixed
input tiles, retaining full output dimensions and all remaining activation
errors. It found sixteen improving maps across four packets; the compact bound
accepted none. These diagnostic maps never selected the runtime output.

## Why the guard is ineffective

For useful proposals, the uncertainty penalty was approximately 76 to 6,104
times the proxy's predicted improvement. This is far beyond a numerical
tolerance issue. The global residual norm bound is valid analytically but too
loose for the actual changes. We used the nuclear norm of the full joint
error-difference matrix; this failure cannot be repaired merely by replacing
independent-tile bounds with a joint calculation, because it already is joint.

Increasing rank or weakening acceptance after seeing these prompts would change
the frozen method. No such tuning occurred. Observing zero violations does not
establish a formal floating-point certificate: the theorem assumes valid
real-arithmetic bounds, and this implementation uses tested float64 numerics.

## Cost and scope

Three float64 metadata objects occupy 835,632 bytes, about 11.81% of the sampled
NVFP4 weight payload plus block-scale storage. Construction took 0.208 seconds
in total. The unfused reference guard averaged 5.29 ms per packet. These figures
do not predict a fused implementation's speed or whole-model overhead; no packed
mixed-format kernel was implemented or benchmarked.

The guarantee concerns aggregate layer-output error across a packet relative
to the same quantized weights with exact activations. It does not guarantee each
token improves, compensate pristine-weight error, or bound downstream NLL.

## Decision

Do not proceed to a fused kernel or claim domain generalization for this guard.
It fails the predeclared acceptance criterion despite available useful changes.
The output-aware proposal provides limited mechanism evidence, while the compact
certification construction requires a substantially different justification to
become practical. This result does not justify another rank/threshold sweep.

- [Frozen protocol and derivation](PROTOCOL.md)
- [Complete packet results](job_331792/REPORT.md)
- [Bound tightness and cost](job_331792/DIAGNOSTICS.md)
- [Raw report, maps, oracles, and hashes](job_331792/report.json)
