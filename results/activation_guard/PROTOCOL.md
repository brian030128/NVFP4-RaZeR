# Online activation error bound: feasibility protocol v0

Declared before executing the first experiment. Date: 2026-09-07.
No calibration text, label, NLL, or configuration election is used. Four fixed
prompts are incoming inputs to a fixed quantizer, not fitting data. Their results
are development evidence and must never be called broad domain generalization.

## Claim and mathematical scope

Fix deployed quantized weights Wq. For a 16-token activation packet X and error
E=Q(X)-X, the added output error relative to Wq with exact activations is
L(E)=||E Wq^T||_F^2=tr(G E^T E), G=Wq^T Wq. It includes interactions across
all changed input tiles. It is not total error relative to pristine weights,
not per-token dominance, and not a downstream loss guarantee.

Build D=diag(sqrt(diag(G))) from weights alone, replacing zero entries by 1.
Let C=D^-1 G D^-1. Decompose C=U Lambda U^T. Fix rank r=16. Center the remaining
eigenvalue interval with alpha=(lambda_min+lambda_(K-r))/2. Store
Gtilde=D(alpha I+F F^T)D, F=U_top diag(sqrt(lambda_top-alpha)).
The exact-eigendecomposition residual radius is half the tail interval width.
The implementation adds conservative terms from full eigendecomposition
reconstruction and orthogonality defects, plus 1e-10 numerical slack.

For A=E_m D, B=E_b D, the joint upper bound is

    L(E_m)-L(E_b)
      <= Ltilde(E_m)-Ltilde(E_b) + epsilon ||A^T A-B^T B||_*.

This follows from spectral/nuclear norm duality. Compute the nuclear norm via
thin QR of [A^T B^T] and an eigendecomposition of the resulting <=32 dimensional
signed Gram matrix. A sum of per-tile acceptances is not valid and is not used.
The theorem assumes a valid residual bound in real arithmetic. Float64 checks
and numerical slack are not a formally verified floating-point certificate.

## Fixed runtime rule

Use canonical FourOverSix E2M1 activations and alpha=1 E0M3 alternatives with
scale groups of 16, shared tensor normalization per 16-token packet and legal
16x64 activation format tiles. Keep full quantized weight matrices fixed.

Start all E2M1. Make one left-to-right pass through all input tiles. For each
tile, compute the exact finite change under Gtilde including the residual from
previous accepted proxy switches; take the switch iff that change is negative.
Incrementally update projected errors. No iterations, retries, rank sweep,
domain-specific threshold or restart. After the pass, apply the joint bound
ONCE. Accept the whole map only if the bound is strictly below minus 1e-10 times
the absolute baseline proxy error; otherwise use baseline for this packet.
The gate uses weights and current activations, never observed model loss.

## First experiment

Pinned local Llama-3.2-1B-Instruct revision
9213176726f574b556790deb65791e0c5aa438b6, native Transformers 4.57.3, eager
attention. Quantize non-head linear weights to canonical FourOverSix and leave
activations unquantized during capture (W4A16 trajectory). Use complete q_proj
matrices of layers 0, 8, 15: no matrix crop or dropped error coordinates.
Construct all metadata before prompts, and record weight/metadata hashes.

Four fixed prose/code/math/multilingual strings are in the hashed runner.
Repeat each string 32 times and take its first 256 tokens including special
tokens. Inspect only positions 240:256. No prompt search. Token hashes are saved.
This yields twelve packets. Probe outputs never update weights or metadata.

For each packet compare the unchanged baseline, activation-MSE map, proposed
map, and guarded map by exact full-output float64 reconstruction. Save proposed
and accepted maps separately; report regressions and per-token changes.

Diagnostic oracle: enumerate the 16 combinations of four prespecified input
tiles [0, floor(n/3), floor(2n/3), n-1], leaving all other tiles at baseline.
Compute true output error and bounds on the full layer, keeping cross terms
with unchanged activation errors. These oracle maps never select the runtime
map. They distinguish absent opportunities, proposal failures, and loose bounds.

Tests cover the residual bound, low-rank nuclear norm identity, joint output
inequality, a tight isotropic case, canonical candidates, incremental proxy
updates, deterministic decisions, zero inputs, and cross-tile interactions.
All execution goes through Slurm H100 allocations. No network downloads needed.

## Feasibility criteria

Report all packets; useful/harmful proposals; improvements missed by the bound;
bound violations; metadata bytes; offline construction time and unfused runtime
time. Count unchanged outputs as fallback, not improvements. Dense exact output
evaluation is a diagnostic and must not be included in a production selector.

Proceed to a fused implementation only if compact metadata actually accepts
useful changes. A zero/near-zero acceptance rate despite exact useful maps is
evidence that this bound is inadequate. Do not repair that by tuning on prompt
loss. Full W4A4 trajectories, fresh model families, task accuracy and end-to-end
latency require a separately frozen transfer protocol after feasibility.
