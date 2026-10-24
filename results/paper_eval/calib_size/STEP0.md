# Part AA step 0

PROTOCOL.md part AA, step 0, on flipquant calib-size f912a30.

**(a) Code.** `git diff caf69be f912a30` touches calibration/fit_windows.py, calibration/tmopt_launch.py, calibration/train_focus.py, calibration/train_map.py, tests/test_calib_size.py; evaluation/, flipquant/, models/ and kernels/ are byte-identical: **pass**.

**(b) Calibration identity.** Llama-3.2-1B-Instruct FlipQuant 16x64 at the release settings (256 windows, part P's command) on calib-size; part P's trainer map sha256 `00c5e22b3c9045d3…`.

| teacher | run_map_sha256 | equal | tiles equal | teacher device | peak allocated (GiB) | device peak (MiB) | job (s) |
|---|---|---|---|---|---:|---:|---:|
| gpu | `00c5e22b3c9045d3…` | True | True | cuda:0 | 14.40 | 15882 | 82 |
| host | `00c5e22b3c9045d3…` | True | True | cpu | 13.67 | 15112 | 84 |

Result: **pass**.

**(c) Evaluation identity** (Llama-3.2-1B-Instruct FlipQuant 16x64, part P's map, on calib-size):
- zero-shot vs part X's record, per task (sample_digest, per-example correctness, per-choice log-likelihoods, metrics): piqa equal, hellaswag equal, winogrande equal, arc_easy equal, arc_challenge equal, lambada_openai equal;
- GSM8K (date pinned to 15 Oct 2026) vs part Y's samples: 2,638 / 2,638 generations and extracted answers byte-identical;
- PPL vs part P's record: per-window NLL equal on wiki (141 windows): True, c4 (256 windows): True.
Result: **pass**.

**(c2) PPL references** on calib-size, against the records the final table uses (not a stop condition):

| model | row | record | Wiki equal | C4 equal | max abs per-window ΔNLL |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | fo6 | `~/n16k64_campaign/paper_eval/llama_instruct/llama3.2-1b-instruct/ppl/fo6.json` | True | True | wiki 0.00e+00, c4 0.00e+00 |
| Llama-3.2-1B-Instruct | fq16_256 | `~/n16k64_campaign/paper_eval/llama_instruct/llama3.2-1b-instruct/ppl/fq-16x64.json` | True | True | wiki 0.00e+00, c4 0.00e+00 |
| Llama-3.2-1B-Instruct | focus256 | `~/n16k64_campaign/paper_eval/llama_instruct/llama3.2-1b-instruct/focus/ppl.json` | True | True | wiki 0.00e+00, c4 0.00e+00 |
| Phi-4 | fo6 | `~/n16k64_campaign/paper/ppl/phi4/fo6/report.json` | True | True | wiki 0.00e+00, c4 0.00e+00 |
| Phi-4 | fq16_256 | `~/flipquant_release/phi4-14b/ppl/flipquant_16x64.json` | True | True | wiki 0.00e+00, c4 0.00e+00 |
| Phi-4 | focus256 | `~/n16k64_campaign/paper_eval/focus/phi4-14b/ppl.json` | True | True | wiki 0.00e+00, c4 0.00e+00 |

**(d) FOCUS reproducibility.** Llama-3.2-1B-Instruct FOCUS at 256 windows retrained with part P's command: 112 modules; 224 tensors differ from part P's state (not bit-identical: evaluated as the run-to-run reference). State sha256 `7352a7166c0fc5a3…` vs part P's `418006a09ce958ad…`.

How far apart the two states are, against how far training moved part P's state from its initialization (m = 1, q = init_q = 6):

| parameter | elements | differing | max abs difference | mean abs difference | mean abs trained - init |
|---|---:|---:|---:|---:|---:|
| m | 60,817,408 | 60,775,896 | 0.0618 | 0.00647 | 0.0131 |
| q | 121,634,816 | 121,558,936 | 0.00705 | 0.000307 | 0.000675 |

The difference is about half the training drift, so it is not rounding noise. A plausible mechanism, not tested here: train_focus's gradients are nondeterministic, and AdamW's first steps (about lr x sign(g)) amplify sign changes of near-zero gradients. The run-to-run row in REPORT.md shows the evaluation-level effect.

**Step 0: passed.**

## Amendment 5, step 0: GPTQ‡

On flipquant calib-size 8dbe2e4.

**Arguments.** For each model, with the main table's windows and codes files: the PPL job's command equals the main table's logged command (parts J / P; part P ran without a cache file), and the zero-shot / GSM8K arguments equal parts X / Y's GPTQ‡ rows': Llama-3.2-1B-Instruct pass, Phi-4 pass, Mistral-7B-Instruct-v0.3 pass, Llama-3.1-8B-Instruct pass, Nemotron-Nano-9B-v2 pass, Qwen3.8-27B pass, Qwen3-1.7B pass, Qwen3-8B pass. **pass**.

**Codes.** Llama-3.2-1B-Instruct GPTQ‡ at 256 windows with part P's command: codes digest `952850d7d8cf4d3d…`, part P's `952850d7d8cf4d3d…`: **equal** (required). The codes file's sha256 `5f65382fe016aec8…` vs the HF upload's `5f65382fe016aec8…`: equal (reported). GPTQ 122 s, job 141 s, device peak 7152 MiB.

**Evaluations** on calib-size, reading that codes file:
- zero-shot vs part X's GPTQ‡ record, per task (sample_digest, per-example correctness, per-choice log-likelihoods, metrics): piqa equal, hellaswag equal, winogrande equal, arc_easy equal, arc_challenge equal, lambada_openai equal;
- GSM8K (date pinned to 15 Oct 2026) vs part Y's samples: 2,638 / 2,638 generations and extracted answers byte-identical;
- PPL vs part P's record: per-window NLL equal on wiki (141 windows): True, c4 (256 windows): True.

**Amendment 5's step 0: passed.**

## Amendment 6, step 0: FlipQuant 8x64 / 256x64

On flipquant calib-size 394e683; Llama-3.2-1B-Instruct at the release 256 windows.

**8x64.** Recalibrated with part P's command: trainer map sha256 `614f0dc41fcc3356…`, part P's `614f0dc41fcc3356…`: **equal**; tiles equal: True (E0M3 share 2.03 %, job 98 s).

Part P's 8x64 map evaluated on calib-size:
- zero-shot vs part X's record, per task: piqa equal, hellaswag equal, winogrande equal, arc_easy equal, arc_challenge equal, lambada_openai equal;
- GSM8K (date pinned) vs part Y's samples: 2,638 / 2,638 identical;
- PPL vs part P's record: per-window NLL equal on wiki (141 windows): True, c4 (256 windows): True.

**256x64.** Recalibrated with part P's command: trainer map sha256 `648d777cd0acce56…`, part P's `648d777cd0acce56…`: **equal**; tiles equal: True (E0M3 share 11.38 %, job 86 s).

Part P's 256x64 map evaluated on calib-size:
- zero-shot vs part X's record, per task: piqa equal, hellaswag equal, winogrande equal, arc_easy equal, arc_challenge equal, lambada_openai equal;
- GSM8K (date pinned) vs part Y's samples: 2,638 / 2,638 identical;
- PPL vs part P's record: per-window NLL equal on wiki (141 windows): True, c4 (256 windows): True.

**Amendment 6's step 0: passed.**
