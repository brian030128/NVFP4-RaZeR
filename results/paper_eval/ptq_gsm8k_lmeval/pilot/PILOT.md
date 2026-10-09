# Part N0: lm-eval gsm8k_llama smoke and pilot (not a result)

## Nemotron-Nano-9B-v2

Rendered prompt of doc 0 (4641 characters; `rendered_prompt.txt`), off switch present: True; it ends with `'m.\n<SPECIAL_11>Assistant\n<think></think>'`.

| run | batch | problems | strict / flexible (%) | think markers | used budget | mean / max tokens | seconds (generate) | peak GiB |
|---|---:|---:|---|---:|---:|---|---|---:|
| pilot_rtn_fo6_b64 | 64 | 128 | 78.9 / 81.2 | 0 | 0 | 146 / 553 | 50 (46) | 36.2 |
| smoke_rtn_fo6 | 8 | 8 | 87.5 / 87.5 | 0 | 0 | 80 / 132 | 12 (9) | 9.8 |

ETA: 23 s per batch of 64, 0.14 h per configuration, 1.4 h for the 9 (GPTQ loads +2 min, Hadamard +20 % assumed).

## Qwen3.8-27B

Rendered prompt of doc 0 (4843 characters; `rendered_prompt.txt`), off switch present: True; it ends with `'|im_start|>assistant\n<think>\n\n</think>\n\n'`.

| run | batch | problems | strict / flexible (%) | think markers | used budget | mean / max tokens | seconds (generate) | peak GiB |
|---|---:|---:|---|---:|---:|---|---|---:|
| pilot_rtn_fo6_b64 | 64 | 128 | 94.5 / 93.8 | 0 | 4 | 264 / 1024 | 312 (307) | 50.8 |
| smoke_rtn_fo6 | 8 | 8 | 100.0 / 100.0 | 0 | 0 | 174 / 380 | 39 (36) | 22.3 |

ETA: 154 s per batch of 64, 0.91 h per configuration, 8.8 h for the 9 (GPTQ loads +2 min, Hadamard +20 % assumed).

Total ETA for the 18 configurations: 10.2 h.

