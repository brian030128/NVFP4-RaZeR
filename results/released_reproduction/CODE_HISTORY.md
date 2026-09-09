# What differs from the released reproduction path

The reproduction extracts commit `e230099bf8006f571614b6c9a71cfd6556c49ef0`
and calls its actual `run_ppl.py`. Current MixFP4 code is removed from the
Python import path. Model paths are redirected to pinned copies of the same
base model IDs in the release's `model2path.json`. No quantization parameters
are selected using evaluation results.

| Item | Released behavior | Consequence for comparison |
|---|---|---|
| Context | `run_ppl.py`: 2048 tokens | The math/code study's 512-token PPL is a different evaluation protocol. |
| Precision | `utils.py`: loads `torch.bfloat16`, including `--use_fp16` | The CLI and paper label this FP16, but the executable loader uses BF16. |
| Attention | Loader leaves backend choice to Transformers | Both direct runs record the selected backend; the earlier native-hook audit forced eager. |
| NVFP4 element rounding | Original `quant_nvfp4` uses magnitude midpoint lookup, with `<=` selecting the smaller magnitude | Current arithmetic quantization uses half-up ties. The two implementations need not produce identical quantized weights. |
| Qwen attention output | Original wrapper computes `attn_output_quant` but invokes `self.o_proj(attn_output)` | Published-release W4A4 settings omit activation quantization at this projection. Preserve and disclose this when reproducing the release. |
| C4 token source | January release can read an author-local cache; February release always samples | Public code cannot establish the contents of the author's absent cached token tensor. |
| PPL aggregation | Per-window float32 NLL times 2048, float32 sum and exponent | The observer reconstructs and checks the exact original returned PPL. |
| Environment | `env.yml` pins Python 3.10.18 and core packages, but no Torch package | Torch 2.7.1 is inferred from Triton 3.3.1 and CUDA 12.6 dependencies, not a documented author pin. |

Relevant Git history, all available locally:

* `ba19e52c206f63b20e6c5b05a51572bc47ccff6a`: January 31 initial release.
* `e230099bf8006f571614b6c9a71cfd6556c49ef0`: February 1 removes the C4
  cache branch from `run_ppl.py`; this is the archived source used here.
* `c5524e41` and `926d3559`: April changes to arithmetic NVFP4 quantization.
* `abab3c65de95e199c6ab977dd2ac7f43238bf369`: June 17 corrects Qwen's
  output-projection input, changing one line to use `attn_output_quant`.
* `7354f3f`: local September saturation correction in the arithmetic quantizer.

The first release's optional C4 cache path was
`/home/yc2367/llm/P2-LLM/data_cache/testloader_{model_family}_c4_{seq_len}.cache`.
The archive contains neither the cached tokens nor a mapping from individual
paper table cells to a code commit and complete environment.

The paper itself lists Qwen3-4B NVFP4 weight-only WikiText as 13.63 in Table 1
and 13.83 in Table 3. Our primary reproduction retains the Table 3 target;
matching Table 1 is a separately identified observation. This is evidence
of an inconsistency between those cells, not proof that all Qwen residuals
share the same cause.

[Released evaluator](https://github.com/abdelfattah-lab/NVFP4-RaZeR/blob/e230099bf8006f571614b6c9a71cfd6556c49ef0/run_ppl.py) ·
[Qwen fix](https://github.com/abdelfattah-lab/NVFP4-RaZeR/commit/abab3c65de95e199c6ab977dd2ac7f43238bf369) ·
[Paper](https://arxiv.org/html/2501.04052v2).
