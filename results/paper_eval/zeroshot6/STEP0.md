# Part X, step 0: code identity (flipquant main vs NVFP4-RaZeR)

flipquant origin/main `3db0072` (fresh detached worktree, unmodified) against NVFP4-RaZeR `9580a2a` (kernel-opt, the clean main checkout); PROTOCOL.md part X. Overall: **all checks passed**.

## (a) Settings: evaluation/downstream.py vs run_lmeval_deploy.py + experiments/paper/04_downstream.py

| item | NVFP4-RaZeR | flipquant main | verdict |
|---|---|---|---|
| lm-eval | 0.4.11, the env's install (a compat import only below 0.4.10) | 0.4.11, the env's install | same |
| HFLM | `HFLM(pretrained=model, tokenizer=tok, batch_size=B, backend='causal')`; `add_bos_token=False` only with --no-bos, which 04_downstream.py does not pass | `HFLM(pretrained=model, tokenizer=tok, batch_size=batch, backend="causal")` | same (default BOS: 0.4.11 keeps the tokenizer's special tokens; both record it) |
| batch | --batch-size, 16; 04_downstream.py: 16, Qwen3.8-27B 8 (paper_common.LMEVAL_BATCH) | 16, Qwen3.8-27B 8 (BATCH) | same |
| simple_evaluate | limit, log_samples=True, bootstrap_iters=100000, random_seed=0, numpy/torch/fewshot seeds 1234, num_fewshot | the same arguments, plus task_manager=TaskManager(include_path=evaluation/lm_eval_tasks) | differs: the include path adds only mmlu_pro_loglik (no standard task is redefined); no effect on these tasks |
| dataset pins | lm_eval_datasets.py (PINS) | the same module, vendored unchanged, plus EXTRA_PINS (MMLU-Pro, EleutherAI/lambada_openai @ 900124bf) | differs: flipquant pins LAMBADA, NVFP4-RaZeR does not know it |
| few-shot | --num-fewshot (04_downstream.py: mmlu=5 only); the rest lm-eval's defaults | FEWSHOT = {mmlu: 5}; the rest lm-eval's defaults | same (the six tasks are 0-shot) |
| primary metric | PRIMARY: acc_norm (ARC, HellaSwag, OpenBookQA, PIQA), acc (BoolQ, WinoGrande, MMLU), exact_match (GSM8K); no LAMBADA | the same, plus lambada_openai = acc | differs: LAMBADA has no primary metric in run_lmeval_deploy.py (primary() raises), so LAMBADA cannot run there |
| records | examples, loglikelihoods, sample_digest, prompt_hash, metrics, timings, native_gemm, coverage | the same fields and functions (ast-identical, below) | same |
| activation scope | convention (c): per-token scales (fake: four_over_six_rows; native: NativeLinear) | --act-scope row (the default here; the sm120 path accepts only row) | same |
| TF32 | off (matmul and cuDNN) | off (matmul and cuDNN) | same |
| model loading | run_ppl_deploy.load_for_evaluation: AutoModelForCausalLM from the calibration record's snapshot and revision, BF16, attn eager then set_attn_implementation('sdpa'), device cuda, eval; every quantized matrix's sha256 checked against the record (Qwen3.8-27B: Qwen3_5ForConditionalGeneration, sdpa) | models.loader.load: the registry's hf_id at --revision, BF16, attn sdpa, device cuda, eval; no weight hash check | differs in route (the attention is SDPA on both; the parity shows equal outputs on the shared model) |
| tokenizer | AutoTokenizer.from_pretrained(snapshot path, revision) | AutoTokenizer.from_pretrained(hf_id, revision) | same files |
| revisions | the calibration record's (data roots); base Mistral-7B-v0.3, base Llama-3.1-8B, Phi-4, Qwen3.8-27B and Qwen3-4B only (sm120/eval/common.MODELS) | registry pins or --revision (part X passes the main-table revisions for all six models) | differs: NVFP4-RaZeR has none of the Instruct models of the panel |
| policies | exported artifacts (native:<artifact>[:<kernel>]) or fake (c); SM120_BUILD_DIR | models.cli flags; weights packed in memory by NVFP4-RaZeR's packer (models/sm120.py); --kernel-build-dir | differs in route; the same build_V libraries |
| processes | several policies per process (BF16 / fake first, then native), one HFLM per policy | one policy per process | differs; the parity runs both ways |

Checked from the sources:
- The record functions `primary`, `mc_record`, `sample_hashes`, `summarize` and `summarize_group` are AST-identical apart from docstrings: primary yes, mc_record yes, sample_hashes yes, summarize yes, summarize_group **NO**. `summarize_group` differs only in form: NVFP4-RaZeR binds `groups = res.get('groups', {})` and then reads `groups.get(task)`, flipquant inlines it as `res.get("groups", {}).get(task)`; it serves group tasks (MMLU) only, none of the six.
- `lm_eval_datasets.py` sha256: NVFP4-RaZeR 5290f5f65ea4383e..., flipquant 5290f5f65ea4383e... (equal: yes).
- flipquant's extra pins: TIGER-Lab/MMLU-Pro @ b189ec765a, EleutherAI/lambada_openai @ 900124bf3b. The panel checks that every LAMBADA load used 900124bf.
- Primary-metric differences (NVFP4-RaZeR, flipquant): lambada_openai: (None, 'acc').
- **LAMBADA**: run_lmeval_deploy.py has no primary metric for lambada_openai, so it cannot evaluate LAMBADA; only flipquant's dataset pin is checked, as the protocol says.

## (b) Same-session harness parity: Phi-4, --limit 24, batch 16

NVFP4-RaZeR `run_lmeval_deploy.py --model phi4` (one process: bf16, then fo6 = native phi4_fo6 / auto_stock, then fq-16x64 = native phi4_tc_16x64 / auto; SM120_BUILD_DIR=build_V) against flipquant `evaluation.downstream --model phi4-14b --revision 2db69c1c` (one process per policy; build_V, --kernel-set auto; the paper 16x64 map the artifact was exported from). NVFP4-RaZeR's harness knows none of the panel's Instruct models (deviation 3), and base Mistral-7B-v0.3's weights are no longer cached, so the parity runs on Phi-4 (amendment 1).

| policy / task | examples | choices | sample digest | correctness | per-choice LL (bitwise) | metrics | primary |
|---|---:|---:|---|---|---|---|---:|
| bf16/arc_easy | 24 | 96 | yes | yes | yes | yes | 0.5417 |
| bf16/arc_challenge | 24 | 96 | yes | yes | yes | yes | 0.6667 |
| bf16/hellaswag | 24 | 96 | yes | yes | yes | yes | 0.7917 |
| bf16/piqa | 24 | 48 | yes | yes | yes | yes | 0.8333 |
| bf16/winogrande | 24 | 48 | yes | yes | yes | yes | 0.8333 |
| fo6/arc_easy | 24 | 96 | yes | yes | yes | yes | 0.6250 |
| fo6/arc_challenge | 24 | 96 | yes | yes | yes | yes | 0.6250 |
| fo6/hellaswag | 24 | 96 | yes | yes | yes | yes | 0.7500 |
| fo6/piqa | 24 | 48 | yes | yes | yes | yes | 0.8333 |
| fo6/winogrande | 24 | 48 | yes | yes | yes | yes | 0.8333 |
| fq-16x64/arc_easy | 24 | 96 | yes | yes | yes | yes | 0.6250 |
| fq-16x64/arc_challenge | 24 | 96 | yes | yes | yes | yes | 0.6250 |
| fq-16x64/hellaswag | 24 | 96 | yes | yes | yes | yes | 0.7917 |
| fq-16x64/piqa | 24 | 48 | yes | yes | yes | yes | 0.8333 |
| fq-16x64/winogrande | 24 | 48 | yes | yes | yes | yes | 0.8333 |
- fo6: kernels NVFP4-RaZeR KernelSet(stock_ko), flipquant stock_ko; every library from build_V: yes (4 / 4 libraries).
- fq-16x64: kernels NVFP4-RaZeR KernelSet(mixed_ko), flipquant mixed_ko; every library from build_V: yes (4 / 4 libraries).
- Tokenization (both record it): NVFP4-RaZeR {"add_bos_token": null, "bos_token": "<|endoftext|>", "bos_token_id": 100257, "context_starts_with_bos": false, "probe_ids": [14924, 25, 220, 16]}; flipquant {"add_bos_token": null, "bos_token": "<|endoftext|>", "bos_token_id": 100257, "context_starts_with_bos": false, "probe_ids": [14924, 25, 220, 16]}.

## (c) IF4, Zou, FOCUS deploy and GPTQ‡ against the paths behind tab:main-ppl

### IF4 / Zou, layer by layer (Mistral-7B-Instruct-v0.3, GPU)

Reference: NVFP4-RaZeR 44f8cea `quantize/adaptive_formats.py` (sha256 20699b4da3679106...), the code of main-ppl's IF4 / Zou records; flipquant main `flipquant/if4.py`, `flipquant/zou_mixfp4.py`.

| rule | weights bitwise | installed-weight sha256: reference / flipquant / main-ppl record | activations bitwise (one WikiText-2 window) |
|---|---|---|---|
| if4 | 224/224 | 3c4d9b3d7eba / 3c4d9b3d7eba / 3c4d9b3d7eba | 224/224 calls (458752 rows) |
| zou | 224/224 | 7d09817587f2 / 7d09817587f2 / 7d09817587f2 | 224/224 calls (458752 rows) |

### evaluation.ppl spot checks against the main-table records (flipquant main, per-window NLL bitwise)

| model / row | windows (wiki, c4) | NLL equal | other | record |
|---|---|---|---|---|
| llama3.2-1b-instruct/focus | 2, 2 | yes |  | `RUN/llama_instruct/llama3.2-1b-instruct/focus/ppl.json` |
| llama3.2-1b-instruct/gptq | 2, 2 | yes | codes_sha256 952850d7d8cf vs 952850d7d8cf (computed by main); data_sha256 equal yes; gptq_candidates absent in main (opt-in, unused in the record) | `RUN/llama_instruct/llama3.2-1b-instruct/gptqdd/ppl.json` |
| mistral-7b/if4 | 4, 4 | yes | installed weights 3c4d9b3d7eba vs 3c4d9b3d7eba | `RUN/mainrows/mistral-7b/if4.json` |
| mistral-7b/zou | 4, 4 | yes | installed weights 7d09817587f2 vs 7d09817587f2 | `RUN/mainrows/mistral-7b/zou.json` |
| mistral-7b/focus | 2, 2 | yes |  | `RUN/focus/mistral-7b/ppl.json` |
| mistral-7b/gptq | 2, 2 | yes | codes_sha256 b8d0f31c8077 vs b8d0f31c8077 (loaded); data_sha256 equal yes | `RUN/gptqdd/mistral-7b/ppl.json` |
| llama3.1-8b-instruct/focus | 2, 2 | yes |  | `RUN/llama_instruct/llama3.1-8b-instruct/focus/ppl.json` |
| llama3.1-8b-instruct/gptq | 2, 2 | yes | codes_sha256 d3fe98167972 vs d3fe98167972 (computed by main); data_sha256 equal yes; gptq_candidates absent in main (opt-in, unused in the record) | `RUN/llama_instruct/llama3.1-8b-instruct/gptqdd/ppl.json` |
| nemotron-nano-9b-v2/focus | 2, 2 | yes |  | `RUN/focus/nemotron-nano-9b-v2/ppl.json` |
| nemotron-nano-9b-v2/gptq | 2, 2 | yes | codes_sha256 22854fc2fac9 vs 22854fc2fac9 (loaded); data_sha256 equal yes | `RUN/gptqdd/nemotron-nano-9b-v2/ppl.json` |
| phi4-14b/focus | 2, 2 | yes |  | `RUN/focus/phi4-14b/ppl.json` |
| phi4-14b/gptq | 2, 2 | yes | codes_sha256 141a919d520c vs 141a919d520c (loaded); data_sha256 equal yes | `RUN/gptqdd/phi4-14b/ppl.json` |
| qwen3.8-27b/focus | 2, 2 | yes |  | `RUN/focus/qwen3.8-27b/ppl.json` |
| qwen3.8-27b/gptq | 2, 2 | yes | codes_sha256 3c1bc3b2cfd8 vs 3c1bc3b2cfd8 (loaded); data_sha256 equal yes | `RUN/gptqdd/qwen3.8-27b/ppl.json` |

### GPTQ‡ codes files (CPU) and FOCUS states

| model | codes file | code digest = record | key = record |
|---|---|---|---|
| mistral-7b | `RUN/gptqdd/mistral-7b/codes_nvfp4_act-fo6.pt` (3,926,070,201 B) | yes (b8d0f31c8077) | yes |
| phi4-14b | `RUN/gptqdd/phi4-14b/codes_nvfp4_act-fo6.pt` (7,667,859,109 B) | yes (141a919d520c) | yes |
| nemotron-nano-9b-v2 | `RUN/ptq/nemotron-nano-9b-v2/codes_nvfp4_act-fo6.pt` (4,337,895,473 B) | yes (22854fc2fac9) | yes |
| qwen3.8-27b | `RUN/ptq/qwen3.8-27b/codes_nvfp4_act-fo6.pt` (13,697,654,681 B) | yes (3c1bc3b2cfd8) | yes |
| llama3.2-1b-instruct | `RUN/zeroshot6/gptq/llama3.2-1b-instruct/codes_nvfp4_act-fo6.pt` (regenerated in step 0 by main's GPTQ) | see the ppl check | |
| llama3.1-8b-instruct | `RUN/zeroshot6/gptq/llama3.1-8b-instruct/codes_nvfp4_act-fo6.pt` (regenerated in step 0 by main's GPTQ) | see the ppl check | |

| model | FOCUS state | sha256 | recorded sha256 | equal |
|---|---|---|---|---|
| llama3.2-1b-instruct | `RUN/llama_instruct/focus_states/llama3.2-1b-instruct/state/focus.pt` | 418006a09ce958ad... | none (recorded here) | n/a |
| mistral-7b | `RUN/focus/mistral-7b/state/focus.pt` | 7963825abb5a23c9... | 7963825abb5a23c9 | yes |
| llama3.1-8b-instruct | `RUN/llama_instruct/focus_states/llama3.1-8b-instruct/state/focus.pt` | e311a189f43cf916... | e311a189f43cf916 | yes |
| nemotron-nano-9b-v2 | `RUN/focus/nemotron-nano-9b-v2/state/focus.pt` | cb772e159dd8d6a5... | cb772e159dd8d6a5 | yes |
| phi4-14b | `RUN/focus/phi4-14b/state/focus.pt` | be66f52576587c0c... | be66f52576587c0c | yes |
| qwen3.8-27b | `/home/dev/.cache/huggingface/hub/models--edgeai-lab--Qwen3.8-27B-FOCUS-NVFP4/snapshots/b84ff372a7ccab5b8541395db26056c3acd14aff/focus_256/focus.pt` | baf02d9c36551191... | baf02d9c36551191 | yes |

Qwen3.8-27B FOCUS state downloaded again: 18,263,340,091 bytes, sha256 baf02d9c36551191..., equal to part Q's record yes, equal to the Hub's LFS sha256 yes (15 min).

Checks: parity yes, identity_if4_zou yes, ppl_checks yes, codes_files yes, llama_codes_files yes, focus_states yes, qwen27b_download yes.
