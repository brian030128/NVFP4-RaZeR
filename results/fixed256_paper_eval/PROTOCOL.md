# Fixed-256-only transfer to the RaZeR evaluation protocol

Freeze all ten math/code-only fixed-256 maps already selected for Qwen3-4B,
Llama-3.1-8B, and Qwen3.8-27B. Use calibration artifacts 333779 (small models)
and 333787 (27B). Run each map on WikiText-2 and C4 alongside one matched
FourOverSix baseline per model: 33 cases, 66 PPL cells. Do not evaluate
adaptive maps, weight-MSE, or new calibration candidates. Do not choose a
winning calibration setting using these results.

Change evaluation to the released 2048-token rule: full nonoverlapping
WikiText-2 raw test windows and 256 seed-0 C4 validation crops from shard
00000. Use the same pinned revisions as the previous baseline audit.
Require exact input token hashes against the released-code run for the
two supported small models and across all policies within each model.
Use the released float32 loss aggregation. The original frozen calibration
remains 512 tokens and math/code only; this is unchanged-map transfer.
Match the released cache setting as well: WikiText uses a fresh cache for
each window; C4 disables caching. No past-key-value state is passed between
windows. A one-window diagnostic proved that disabling WikiText caching
changes SDPA-path numerical results despite identical weights and tokens.

Use tensor-wide activation scaling as in the released quantizer, with
SDPA attention. Quantize every targeted linear input, including Qwen's
o_proj; do not reproduce the historical omission for this full-W4A4 panel.
Weights use the existing FourOverSix E2M1 baseline plus exactly 256 8x64
E0M3 blocks with alpha1. Verify pristine weight hashes and frozen map hashes.
The 27B model uses its native Transformers 5.16.1 text-linear scope; no
published RaZeR row exists for this model.

Validate the small-model baseline against the corrected repository wrapper
on a full window. For Llama, require full baseline PPL equality to the
released-code FourOverSix run under the same Torch/Transformers environment.
Keep the Qwen historical-release values separate because the old wrapper
did not quantize o_proj inputs.

Report each model/setting's WikiText PPL, C4 PPL, arithmetic mean, E0M3 block
count, and signed absolute PPL differences from its matched baseline.
Retain raw per-window losses and descriptive paired NLL diagnostics. Do
not interpret tensor-wide activation-factor results as causal generation
likelihoods: future tokens can influence earlier activation scales.

Use gov113008/taide_h200 with eight H200s and independent one-GPU Slurm
steps, up to eight simultaneously. Every new download and temporary
dependency stays in the worker's job-local /tmp directory.
