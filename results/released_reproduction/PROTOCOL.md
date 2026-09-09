# Direct reproduction using the released RaZeR evaluator

Target: reproduce the displayed Table 3 WikiText-2/C4 numbers for the two
models with disputed baselines: base Llama-3.1-8B and Qwen3-4B. Run BF16 and
NVFP4, FourOverSix, and RaZeR at W4A16 and W4A4: 14 runs, 28 PPL cells.
Success for a cell means its computed PPL rounds to the published two-decimal
value. Report every cell and signed residual; do not select a configuration
by closeness to the table or retune any quantizer.

Extract commit e230099bf8006f571614b6c9a71cfd6556c49ef0 (2026-02-01), the
first portable released evaluator, from Git into worker scratch. Invoke its
actual run_ppl.py through runpy. Its model wrappers, quantizers, seed-0
sampler, default attention backend, 2048-token contexts, and FP32 PPL
aggregation remain unchanged. Only model locations are redirected to pinned
checkpoint snapshots. Hooks record inputs, losses, model/backend identity,
and quantized weight hashes; their return values never change inference.

The initial January release used an author-local cached C4 token tensor at
/home/yc2367/llm/P2-LLM/data_cache. That artifact is unavailable here. The
February evaluator removes the cache and uses the same seed-0 generation
branch. This is a reproducible public-code attempt, not proof that the paper
used those exact cached C4 windows. No calibration is performed.

Important code-history differences from the current tree:

* Original NVFP4 uses explicit midpoint lookup with ties toward smaller
  magnitude. April 2026 changes introduced exponent/mantissa arithmetic with
  half-up rounding; the later local saturation fix is absent from this release.
* Original Qwen attention computes quantized o_proj inputs but passes the
  unquantized tensor. June commit abab3c6 corrects this. Preserve this historical
  behavior in the reproduction and label it, rather than calling it full W4A4.
* Let the released loader choose its attention implementation. The preceding
  hook-based audit explicitly forced eager attention.

The first attempt uses the available Python 3.11/Torch 2.9 environment with
Transformers 4.57.3, recording all relevant versions. The released env.yml
pins Python 3.10.18, Transformers 4.57.3, and Triton 3.3.1 but omits Torch
itself. If environment differences remain material, test a separately labeled
environment reconstruction; do not claim an inferred Torch version was specified.

Follow-up fixed environment diagnostic: retain every case and every evaluation
setting, change to Python 3.10.18 and the listed core package pins, with
Torch 2.7.1/CUDA 12.6 inferred from Triton 3.3.1 and the CUDA dependency pins.
Keep both attempts and compare weights, tokens, losses, and PPL across them.
This is one explicit environment hypothesis, not an environment-version sweep.
The Qwen NVFP4 Wiki weight-only target differs between Table 1 (13.63) and
Table 3 (13.83); Table 3 remains the primary target, and the inconsistency
must be disclosed rather than silently substituting the more favorable value.

Request gov113008/taide as one 8-H100 allocation, or use an available
8-H200 allocation on gov113008/taide_h200 and record the hardware. Launch independent srun
steps with one GPU and four CPUs each; up to eight cases run concurrently,
and remaining cases wait for a free step allocation. HF caches and extracted
source stay in job-local /tmp; reports and original output files persist in
the repository. Compare against the published numbers, not the old 512-token
study. No seed sweep, new calibration, or E0M3 selection is involved.
