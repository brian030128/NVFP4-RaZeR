# Hugging Face upload of the FOCUS states (2026-10-09)

The user's request (relayed by the coordinator, 2026-10-09): the FOCUS states behind the main-ppl FOCUS row (paper-eval dbe3a48, part K) as five **private** repositories under `edgeai-lab`, `focus.pt` as is (no NVFP4 code export). Qwen3.8-27B has no FOCUS state and no repository. The active account was checked first (`edgeai-lab`; no token printed or changed); every repository was created private and checked private before any file went up. Nothing is public.

| model | repository | HF head commit | files |
|---|---|---|---|
| qwen3-1.7b | [edgeai-lab/Qwen3-1.7B-FOCUS-NVFP4](https://huggingface.co/edgeai-lab/Qwen3-1.7B-FOCUS-NVFP4) | `36158bc59ed7903dd4f657a11722624563048b38` | `.gitattributes`, `LICENSE`, `README.md`, `focus.pt`, `ppl_summary.json` |
| qwen3-8b | [edgeai-lab/Qwen3-8B-FOCUS-NVFP4](https://huggingface.co/edgeai-lab/Qwen3-8B-FOCUS-NVFP4) | `b785bd6a79f82c2234ad0cf53370f714a4e21b2e` | `.gitattributes`, `LICENSE`, `README.md`, `focus.pt`, `ppl_summary.json` |
| mistral-7b | [edgeai-lab/Mistral-7B-Instruct-v0.3-FOCUS-NVFP4](https://huggingface.co/edgeai-lab/Mistral-7B-Instruct-v0.3-FOCUS-NVFP4) | `e256ed7832259f70c6b48343a3b6c9f3b3aaf223` | `.gitattributes`, `LICENSE`, `README.md`, `focus.pt`, `ppl_summary.json` |
| nemotron-nano-9b-v2 | [edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FOCUS-NVFP4](https://huggingface.co/edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FOCUS-NVFP4) | `825add4c319d0dde9219b655410534ba2ab2efc1` | `.gitattributes`, `LICENSE`, `NOTICE`, `README.md`, `focus.pt`, `ppl_summary.json` |
| phi4-14b | [edgeai-lab/phi-4-FOCUS-NVFP4](https://huggingface.co/edgeai-lab/phi-4-FOCUS-NVFP4) | `77da31ba0f49c8001720b121a690008fc36b7cf1` | `.gitattributes`, `LICENSE`, `README.md`, `focus.pt`, `ppl_summary.json` |

Upload path: huggingface_hub 1.31 `create_repo(private=True)`, then `upload_large_folder` (the resumable, multi-worker large-file path; Xet storage) with exactly the staged files; each repository has two commits, the empty `initial commit` and `Add files using upload-large-folder tool`.

## Files and sha256

Every file was downloaded again from the Hub at the head commit; its sha256 equals the staged file's (and, for `focus.pt`, the Hub's LFS sha256 too).

| repository | file | bytes | sha256 (staged = remote) | download | LFS |
|---|---|---:|---|---|---|
| Qwen3-1.7B-FOCUS-NVFP4 | `LICENSE` | 11,343 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` | equal | - |
| Qwen3-1.7B-FOCUS-NVFP4 | `README.md` | 7,112 | `3961ca38a27354bb3e7d7e82bf969788bb30c4604509750e105b532e9f854431` | equal | - |
| Qwen3-1.7B-FOCUS-NVFP4 | `focus.pt` | 1,057,104,195 | `5fa877acad86e7b9f64a39936717f8032cab0696d0e657b0525010b4f06a4945` | equal | equal |
| Qwen3-1.7B-FOCUS-NVFP4 | `ppl_summary.json` | 3,865 | `ec585b2d65572e918eb36d08e7c7d78bc3f0d1bb191e70ca20caf6e917e8b245` | equal | - |
| Qwen3-8B-FOCUS-NVFP4 | `LICENSE` | 11,343 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` | equal | - |
| Qwen3-8B-FOCUS-NVFP4 | `README.md` | 7,111 | `3b1cd2f265ff13a51114ebd0f5cece9d2375549fa07282f1bf2afbeed4b190a9` | equal | - |
| Qwen3-8B-FOCUS-NVFP4 | `focus.pt` | 5,209,511,899 | `e4689e64c66fe50afa6022485604efc01c5bf21e369ab54dbc4f7775f4a1b7ae` | equal | equal |
| Qwen3-8B-FOCUS-NVFP4 | `ppl_summary.json` | 3,874 | `b048f42b266ec5ee4e84b9a52ec225c2011a24ebd22f60a40dc49bd5c430f98e` | equal | - |
| Mistral-7B-Instruct-v0.3-FOCUS-NVFP4 | `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | equal | - |
| Mistral-7B-Instruct-v0.3-FOCUS-NVFP4 | `README.md` | 7,094 | `d55702cca7253fb744dbb0ccb54c4bda47ea8fe543b5a1c0ab1b60587a8105aa` | equal | - |
| Mistral-7B-Instruct-v0.3-FOCUS-NVFP4 | `focus.pt` | 5,234,657,055 | `7963825abb5a23c9d38082c76063d477ee249f5b652f0e3edb5393d1dea376aa` | equal | equal |
| Mistral-7B-Instruct-v0.3-FOCUS-NVFP4 | `ppl_summary.json` | 3,891 | `1474ba91417cfccea8ebfaf5e20aaa53ec251a7baf38925f4c74091ae65f4308` | equal | - |
| NVIDIA-Nemotron-Nano-9B-v2-FOCUS-NVFP4 | `LICENSE` | 10,254 | `8cf5f9a673059fc727c03fae31c0d6e365bbf4ef1b7fd9b3099fefe1b49fdda1` | equal | - |
| NVIDIA-Nemotron-Nano-9B-v2-FOCUS-NVFP4 | `NOTICE` | 67 | `5fc60716a9ba57f3792e71ed776b4065671c64d0c2db716698f4d646ad833eb1` | equal | - |
| NVIDIA-Nemotron-Nano-9B-v2-FOCUS-NVFP4 | `README.md` | 7,802 | `70864504ac0d12ec750c21bd7c7a3a25fc9a1700f5aece10a1c15495d6b7f95c` | equal | - |
| NVIDIA-Nemotron-Nano-9B-v2-FOCUS-NVFP4 | `focus.pt` | 5,783,806,155 | `cb772e159dd8d6a5fb72f3ab63c14bb58a481db39bfa4aca48ca0600bd510c3f` | equal | equal |
| NVIDIA-Nemotron-Nano-9B-v2-FOCUS-NVFP4 | `ppl_summary.json` | 3,911 | `ff5702fc4950b6d26a2665be78f4c468a77a159a69509a87de05cf8c47b2fb26` | equal | - |
| phi-4-FOCUS-NVFP4 | `LICENSE` | 1,084 | `fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d` | equal | - |
| phi-4-FOCUS-NVFP4 | `README.md` | 6,961 | `3fac711e59e6ffcd21bde943533619ba61da8d6226f222328ab646c177db55b6` | equal | - |
| phi-4-FOCUS-NVFP4 | `focus.pt` | 10,223,745,627 | `be66f52576587c0c71473e19c378bb71b2a48d89802ffa0d3e24c1c4690cd38f` | equal | equal |
| phi-4-FOCUS-NVFP4 | `ppl_summary.json` | 3,872 | `a6736001dbfc49b613c5b38d9caf1c9b0cefb94db3dbf33934a5230f9a6fba0a` | equal | - |

## focus.pt: the state as is

`focus.pt` is part K's calibration output, copied unchanged (a reflink copy; sha256 of the source = the staged file = the uploaded file). Its metadata needed no change: the scan below found no local path, user name, email, GitHub owner, reference-implementation branch or commit, or "razer" in its `config` and `source`, its layer and field names, its archive record names or its raw bytes.

| model | source (part K) | sha256 (source = uploaded) | layers | m | q | config |
|---|---|---|---:|---:|---:|---|
| qwen3-1.7b | `/home/dev/n16k64_campaign/paper_eval/focus/qwen3-1.7b/state/focus.pt` | `5fa877acad86e7b9f64a39936717f8032cab0696d0e657b0525010b4f06a4945` | 196 | 88,080,384 | 176,160,768 | data release, 256 x 512, batch 32, lr 0.005 / 0.001, top-1000, seed 42 |
| qwen3-8b | `/home/dev/n16k64_campaign/paper_eval/focus/qwen3-8b/state/focus.pt` | `e4689e64c66fe50afa6022485604efc01c5bf21e369ab54dbc4f7775f4a1b7ae` | 252 | 434,110,464 | 868,220,928 | data release, 256 x 512, batch 32, lr 0.005 / 0.001, top-1000, seed 42 |
| mistral-7b | `/home/dev/n16k64_campaign/paper_eval/focus/mistral-7b/state/focus.pt` | `7963825abb5a23c9d38082c76063d477ee249f5b652f0e3edb5393d1dea376aa` | 224 | 436,207,616 | 872,415,232 | data release, 256 x 512, batch 32, lr 0.005 / 0.001, top-1000, seed 42 |
| nemotron-nano-9b-v2 | `/home/dev/n16k64_campaign/paper_eval/focus/nemotron-nano-9b-v2/state/focus.pt` | `cb772e159dd8d6a5fb72f3ab63c14bb58a481db39bfa4aca48ca0600bd510c3f` | 120 | 481,976,320 | 963,952,640 | data release, 256 x 512, batch 32, lr 0.005 / 0.001, top-1000, seed 42 |
| phi4-14b | `/home/dev/n16k64_campaign/paper_eval/focus/phi4-14b/state/focus.pt` | `be66f52576587c0c71473e19c378bb71b2a48d89802ffa0d3e24c1c4690cd38f` | 160 | 851,968,000 | 1,703,936,000 | data release, 256 x 512, batch 32, lr 0.005 / 0.001, top-1000, seed 42 |

## Scans

The FlipQuant upload's patterns (hf_stage.py: local path prefixes, the run directories, "razer", the GitHub owner, the user names, the university domain, HF tokens, email addresses, the host name, the internal branch name), plus `angelslim|tencent|github|gitlab|branch|commit` on the state's metadata and on our own text files (README.md, ppl_summary.json), over every staged file: the state's metadata strings and names, its archive record names and its raw bytes (streamed; the patterns /home/, /tmp/, razer, the owner, HF tokens, the user names, the run directory, angelslim), and every line of the text files.

| repository | hits |
|---|---:|
| Qwen3-1.7B-FOCUS-NVFP4 | 0 |
| Qwen3-8B-FOCUS-NVFP4 | 0 |
| Mistral-7B-Instruct-v0.3-FOCUS-NVFP4 | 0 |
| NVIDIA-Nemotron-Nano-9B-v2-FOCUS-NVFP4 | 0 |
| phi-4-FOCUS-NVFP4 | 0 |

## License files

- Qwen3-1.7B, Qwen3-8B: `LICENSE`, the base snapshot's file at the pinned revision (as staged for the FlipQuant repositories; Apache-2.0).
- Mistral-7B-Instruct-v0.3: the base repository has no license file (its card declares apache-2.0); `LICENSE` is the Apache License 2.0 text from https://www.apache.org/licenses/LICENSE-2.0.txt (retrieved 2026-10-09 07:22 UTC, sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`), added as the request asked ("add Mistral's license").
- NVIDIA-Nemotron-Nano-9B-v2: `LICENSE` (the NVIDIA Open Model License Agreement, verbatim from the page the base card links, retrieved 2026-10-07) and `NOTICE`, as staged for the FlipQuant repository.
- phi-4: `LICENSE`, the base repository's own file (microsoft/phi-4@2db69c1c3e91a05d2c64a3185acfbaf36f744e25: LICENSE, retrieved 2026-10-09 07:22 UTC without a token; MIT).

## The cards

Each `README.md` (copied under `cards/`) has front matter with the base model's license, `base_model`, and the tags quantization / fp4 / nvfp4 / focus, and no draft banner. It states: our reproduction of FOCUS (arXiv:2608.01847) as a baseline for the FlipQuant paper, not an official release by the FOCUS authors; the pinned base revision; the paper's NVFP4 hyperparameters (lr 5e-3 / 1e-3, 2 sub-blocks of 8, KL-Top k = 1000, AdamW with a constant LR, 1 epoch, global batch 32, init q 6, seed 42); the different calibration data (the FlipQuant release set, 256 x 512 math/code windows, 8 steps; the paper: 1,248 x 2,048 WikiText-2, 39 steps); per-token FourOverSix activations in calibration and evaluation; what the file holds (FP32 m per 16-element block, q per 8-element sub-block, s2 per tensor, per linear layer) and how it deploys as standard NVFP4 codes; that it is not a quantized checkpoint (it needs the BF16 base at the pinned revision and the FOCUS implementation; the code will be released with the paper, no link); native WikiText-2 / C4 PPL on the RTX PRO 6000 with BF16 / NVFP4 / FourOverSix and the paired ΔNLL vs FourOverSix (from part K's records and final.json's sources, recomputed and checked equal to final.json); training time and peak GPU memory (report.json). `ppl_summary.json` holds the same numbers and the settings, without local paths.

## Checks after the upload

| repository | private (after upload / after all five) | anonymous API | card metadata (Hub) | card render (cards.json) | problems |
|---|---|---:|---|---|---|
| Qwen3-1.7B-FOCUS-NVFP4 | True / True | HTTP 401 / 401 | license apache-2.0, base_model Qwen/Qwen3-1.7B, tags ['quantization', 'fp4', 'nvfp4', 'focus'] | Hub YAML validation passed; 4/4 tables, 2/2 code blocks rendered; README = staged: True | none |
| Qwen3-8B-FOCUS-NVFP4 | True / True | HTTP 401 / 401 | license apache-2.0, base_model Qwen/Qwen3-8B, tags ['quantization', 'fp4', 'nvfp4', 'focus'] | Hub YAML validation passed; 4/4 tables, 2/2 code blocks rendered; README = staged: True | none |
| Mistral-7B-Instruct-v0.3-FOCUS-NVFP4 | True / True | HTTP 401 / 401 | license apache-2.0, base_model mistralai/Mistral-7B-Instruct-v0.3, tags ['quantization', 'fp4', 'nvfp4', 'focus'] | Hub YAML validation passed; 4/4 tables, 2/2 code blocks rendered; README = staged: True | none |
| NVIDIA-Nemotron-Nano-9B-v2-FOCUS-NVFP4 | True / True | HTTP 401 / 401 | license other, base_model nvidia/NVIDIA-Nemotron-Nano-9B-v2, tags ['quantization', 'fp4', 'nvfp4', 'focus'] | Hub YAML validation passed; 4/4 tables, 2/2 code blocks rendered; README = staged: True | none |
| phi-4-FOCUS-NVFP4 | True / True | HTTP 401 / 401 | license mit, base_model microsoft/phi-4, tags ['quantization', 'fp4', 'nvfp4', 'focus'] | Hub YAML validation passed; 4/4 tables, 2/2 code blocks rendered; README = staged: True | none |

"The card renders": the web page of a private repository cannot be fetched with a token (HTTP 401 for every repository, recorded as `page` in hf_upload.json), so the check is the Hub's own YAML validation of the card it serves (`ModelCard.validate`), the metadata the Hub parsed from it, the README downloaded from the Hub byte-identical to the staged one, and a local CommonMark + tables render (markdown-it-py) in which every table and code fence renders. The user can open the repositories while logged in.

Not uploaded: report.json, ppl.json and every other record (they hold local paths).

Files here: `stage.json` (staging: files, sha256, scans, license sources, the state's metadata), `hf_upload.json` (the upload: commits, per-file checks), `cards/<repo>/{README.md, ppl_summary.json}`; scripts `experiments/paper_eval/focus_hf.py`.
