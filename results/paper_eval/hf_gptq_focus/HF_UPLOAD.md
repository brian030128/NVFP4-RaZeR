# Hugging Face upload: the Llama-3.2-1B-Instruct FOCUS state and FlipQuant maps, and the main-table GPTQ‡ weights (2026-10-15)

The user's requests (relayed by the coordinator, 2026-10-15): nine **private** repositories under `edgeai-lab`, and later the same day a tenth, `edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant`.
- **Account:** the active account was checked first (`edgeai-lab`); no token was printed or changed.
- **Order of operations:** each repository was created private and checked private (anonymous HTTP 401) before any file went up. It was then uploaded and verified.
- **Verification:** every file's sha256 was compared with the staged file's. Each file's method is listed below.
  - The first four repositories (the FOCUS state, Mistral, Nemotron, Phi-4): every file was downloaded again from the Hub at the head, the codes files included.
  - From Qwen3.8-27B on: files up to 1 GiB were downloaded again (and checked against the Hub's LFS sha256 where it has one). Larger files were checked against the Hub's server-side LFS sha256 and size, without a download (the coordinator's rule).
  - The first process downloaded into /dev/shm (8 GiB), which cannot hold Qwen3.8-27B's 13.7 GB codes file, and it stopped there after the upload. The rerun found that repository private, with exactly the staged files and LFS sha256, and verified it as above.
- **Existing repositories:** none was touched.

## Uploaded

Visibility: at 2026-10-16 21:14 UTC, without a token (see Visibility).

| repository | kind | files | sha256 = staged (method) | visibility now | anonymous at upload | head | card YAML (Hub) | README = staged |
|---|---|---:|---|---|---:|---|---|---|
| `edgeai-lab/Llama-3.2-1B-Instruct-FOCUS-NVFP4` | focus | 6 | equal (6 downloaded) | public | 401 | `bf74dbb1dc95` | True | True |
| `edgeai-lab/Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4` | gptq | 4 | equal (4 downloaded) | public | 401 | `2e57b762ec21` | True | True |
| `edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4` | gptq | 5 | equal (5 downloaded) | public | 401 | `fcf9b7c93918` | True | True |
| `edgeai-lab/phi-4-GPTQ-BF16Calib-NVFP4` | gptq | 4 | equal (4 downloaded) | public | 401 | `4fb3389d2f08` | True | True |
| `edgeai-lab/Qwen3.8-27B-GPTQ-BF16Calib-NVFP4` | gptq | 4 | equal (3 downloaded, 1 by Hub LFS sha256) | public | 401 | `6f775089c456` | True | True |
| `edgeai-lab/Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4` | gptq | 6 | equal (6 downloaded) | public | 401 | `bdd8efe0b5f2` | True | True |
| `edgeai-lab/Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4` | gptq | 6 | equal (5 downloaded, 1 by Hub LFS sha256) | public | 401 | `47a528ecd214` | True | True |
| `edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant` | flipquant | 8 | equal (8 downloaded) | public | 401 | `1896c62f9e45` | True | True |
| `edgeai-lab/Qwen3-1.7B-GPTQ-BF16Calib-NVFP4` | gptq | 4 | equal (4 downloaded) | public | 401 | `46641e9e38c6` | True | True |
| `edgeai-lab/Qwen3-8B-GPTQ-BF16Calib-NVFP4` | gptq | 4 | equal (3 downloaded, 1 by Hub LFS sha256) | public | 401 | `3909c3bf0493` | True | True |

## Verification per file

| repository | file | bytes | method | sha256 = staged |
|---|---|---:|---|---|
| `Llama-3.2-1B-Instruct-FOCUS-NVFP4` | `LICENSE` | 7,712 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FOCUS-NVFP4` | `NOTICE` | 116 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FOCUS-NVFP4` | `README.md` | 8,388 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FOCUS-NVFP4` | `USE_POLICY.md` | 6,021 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FOCUS-NVFP4` | `focus.pt` | 729,889,075 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FOCUS-NVFP4` | `ppl_summary.json` | 3,900 | download sha256 | True |
| `Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4` | `LICENSE` | 11,358 | download sha256 | True |
| `Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4` | `README.md` | 6,585 | download sha256 | True |
| `Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4` | `codes_nvfp4_act-fo6.pt` | 3,926,070,201 | download sha256 | True |
| `Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4` | `ppl_summary.json` | 3,964 | download sha256 | True |
| `NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4` | `LICENSE` | 10,254 | download sha256 | True |
| `NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4` | `NOTICE` | 67 | download sha256 | True |
| `NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4` | `README.md` | 7,401 | download sha256 | True |
| `NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4` | `codes_nvfp4_act-fo6.pt` | 4,337,895,473 | download sha256 | True |
| `NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4` | `ppl_summary.json` | 4,005 | download sha256 | True |
| `phi-4-GPTQ-BF16Calib-NVFP4` | `LICENSE` | 1,084 | download sha256 | True |
| `phi-4-GPTQ-BF16Calib-NVFP4` | `README.md` | 6,392 | download sha256 | True |
| `phi-4-GPTQ-BF16Calib-NVFP4` | `codes_nvfp4_act-fo6.pt` | 7,667,859,109 | download sha256 | True |
| `phi-4-GPTQ-BF16Calib-NVFP4` | `ppl_summary.json` | 3,939 | download sha256 | True |
| `Qwen3.8-27B-GPTQ-BF16Calib-NVFP4` | `LICENSE` | 11,358 | download sha256 | True |
| `Qwen3.8-27B-GPTQ-BF16Calib-NVFP4` | `README.md` | 6,818 | download sha256 | True |
| `Qwen3.8-27B-GPTQ-BF16Calib-NVFP4` | `codes_nvfp4_act-fo6.pt` | 13,697,654,681 | hub LFS sha256 | True |
| `Qwen3.8-27B-GPTQ-BF16Calib-NVFP4` | `ppl_summary.json` | 3,946 | download sha256 | True |
| `Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4` | `LICENSE` | 7,712 | download sha256 | True |
| `Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4` | `NOTICE` | 116 | download sha256 | True |
| `Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4` | `README.md` | 7,891 | download sha256 | True |
| `Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4` | `USE_POLICY.md` | 6,021 | download sha256 | True |
| `Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4` | `codes_nvfp4_act-fo6.pt` | 547,457,897 | download sha256 | True |
| `Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4` | `ppl_summary.json` | 3,990 | download sha256 | True |
| `Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4` | `LICENSE` | 7,627 | download sha256 | True |
| `Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4` | `NOTICE` | 116 | download sha256 | True |
| `Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4` | `README.md` | 7,894 | download sha256 | True |
| `Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4` | `USE_POLICY.md` | 4,691 | download sha256 | True |
| `Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4` | `codes_nvfp4_act-fo6.pt` | 3,926,070,201 | hub LFS sha256 | True |
| `Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4` | `ppl_summary.json` | 3,998 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FlipQuant` | `LICENSE` | 7,712 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FlipQuant` | `NOTICE` | 116 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FlipQuant` | `README.md` | 9,160 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FlipQuant` | `USE_POLICY.md` | 6,021 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FlipQuant` | `flipquant_16x64.pt` | 989,903 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FlipQuant` | `flipquant_256x64.pt` | 99,077 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FlipQuant` | `flipquant_8x64.pt` | 1,940,121 | download sha256 | True |
| `Llama-3.2-1B-Instruct-FlipQuant` | `ppl_summary.json` | 7,466 | download sha256 | True |
| `Qwen3-1.7B-GPTQ-BF16Calib-NVFP4` | `LICENSE` | 11,343 | download sha256 | True |
| `Qwen3-1.7B-GPTQ-BF16Calib-NVFP4` | `README.md` | 6,407 | download sha256 | True |
| `Qwen3-1.7B-GPTQ-BF16Calib-NVFP4` | `codes_nvfp4_act-fo6.pt` | 792,899,701 | download sha256 | True |
| `Qwen3-1.7B-GPTQ-BF16Calib-NVFP4` | `ppl_summary.json` | 3,936 | download sha256 | True |
| `Qwen3-8B-GPTQ-BF16Calib-NVFP4` | `LICENSE` | 11,343 | download sha256 | True |
| `Qwen3-8B-GPTQ-BF16Calib-NVFP4` | `README.md` | 6,400 | download sha256 | True |
| `Qwen3-8B-GPTQ-BF16Calib-NVFP4` | `codes_nvfp4_act-fo6.pt` | 3,907,220,925 | hub LFS sha256 | True |
| `Qwen3-8B-GPTQ-BF16Calib-NVFP4` | `ppl_summary.json` | 3,938 | download sha256 | True |

## Naming of the GPTQ repositories

The eight GPTQ repositories were first staged under the requested names `<Base>-GPTQ-NVFP4`. Two of those names already existed under `edgeai-lab`, created by another upload. That upload was not touched; these facts were read without changing anything:

- `edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-GPTQ-NVFP4`: created 2026-10-09 08:57 UTC, public; GPTQ codes with FourOverSix calibration activations (its commit title), not the main table's BF16-calibration codes; no model card.
- `edgeai-lab/Qwen3.8-27B-GPTQ-NVFP4`: created 2026-10-09 08:57 UTC, public; GPTQ codes with FourOverSix calibration activations (its commit title), not the main table's BF16-calibration codes; no model card.

The uploads were held, and no GPTQ repository was created, until the user's naming: <Base>-GPTQ-BF16Calib-NVFP4 for all eight (2026-10-15 09:37 UTC). The cards of the two models with an existing repository say in one sentence that this one is not the same artifact: it is calibrated with BF16 activations (the main table's GPTQ‡), that one with FourOverSix activations. The staged files were kept, only the records were renamed, and each card was regenerated and rescanned.

## The FOCUS state

**`edgeai-lab/Llama-3.2-1B-Instruct-FOCUS-NVFP4`** holds `focus.pt` as is: part P's state behind tab:main-ppl and parts X / Y.
- sha256 `418006a09ce958ad1de91c5483fe28463da85509c09d4ed30e5be8d48f69192c`, equal to the value part X's step 0 recorded. Part P recorded no sha256; part X matched its PPL record bit for bit.
- The layout and card of `edgeai-lab/Llama-3.1-8B-Instruct-FOCUS-NVFP4`: `focus.pt`; the card (our reproduction of FOCUS, arXiv:2608.01847; not an official release); `ppl_summary.json`, `LICENSE`, `USE_POLICY.md`, `NOTICE`.

## The FlipQuant maps (Llama-3.2-1B-Instruct)

**`edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant`** holds part P's three Llama-3.2-1B-Instruct maps (amendment 1, 4a64caf).
- **Before staging:** each map's sha256 was checked against part P's record (`map_sha256` in `records/llama3.2-1b-instruct/<unit>/run.json`); all three are equal.
- **Meta:** cleaned as for `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant`. Local paths are made relative, the GitHub owner is removed from the extension reference, and the reference implementation's branch and commit are removed. The tiles and every tensor record are bit-identical.
- **Layout, file names and card:** those of `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant`. They come from `llama_hf.py`'s staging, summary and card functions, bound to this model; those functions regenerate the 8B card and `ppl_summary.json` byte for byte. Only the model's own facts differ:
  - its records, windows, costs and perplexities;
  - the Llama 3.2 license strings and files, as in the 1B FOCUS card.
- **The windows sentence:** the first 128 fit windows equal the Llama-3.1-8B release's. The 128 extension windows share none with it: the model's own development draws contain all 128 of that release's extension documents (part P's window record).

| map | part P's record = file before staging | uploaded file (meta cleaned) | E0M3 tiles |
|---|---|---|---:|
| `flipquant_8x64.pt` | `4f693ed87bcd4912…` (True) | `36bf7d743cfa4a78…` | 38,587 of 1,900,544 |
| `flipquant_16x64.pt` | `d9dfa5a7d2045af4…` (True) | `1c0f1746eb63a431…` | 26,656 of 950,272 |
| `flipquant_256x64.pt` | `54effb555ff05278…` (True) | `4d98536e354fe70c…` | 6,757 of 59,392 |

## The GPTQ‡ weights

One repository per model, holding `codes_nvfp4_act-fo6.pt` as is: the file behind the main table's GPTQ‡ cell.
- **Checked before staging, against the main-table record:**
  - each file's code digest (FlipQuant's `codes_sha256` formula over its stored entries);
  - its settings key.
- **The key against the main-table variant:**
  - BF16-activation calibration: `propagate` = `hessian_input` = `bf16` (round 3's activation-quantized codes have `quantized`);
  - the release windows (256), block 128, damp 0.01;
  - NVFP4 weights, no map, no rotation.
- **Decoding:** the card's PyTorch snippet reproduces FlipQuant's dequantization bit for bit on each file's first module.

| repository | source file | bytes | file sha256 | rule (4) basis | code digest = record | key = record | main-table variant | written by FlipQuant |
|---|---|---:|---|---|---|---|---|---|
| `edgeai-lab/Qwen3-1.7B-GPTQ-BF16Calib-NVFP4` | `RUN/gptqdd/qwen3-1.7b/codes_nvfp4_act-fo6.pt` | 792,899,701 | `bb50de2383f7690a…` | the code digest equal to the main-table record; the file sha256 is recorded here for the first time | True (`415a788a25da`) | True | True | `c67a5bd` |
| `edgeai-lab/Qwen3-8B-GPTQ-BF16Calib-NVFP4` | `RUN/gptqdd/qwen3-8b/codes_nvfp4_act-fo6.pt` | 3,907,220,925 | `335a69b142c726a9…` | the code digest equal to the main-table record; the file sha256 is recorded here for the first time | True (`4cba489acee8`) | True | True | `c67a5bd` |
| `edgeai-lab/Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4` | `RUN/gptqdd/mistral-7b/codes_nvfp4_act-fo6.pt` | 3,926,070,201 | `7d837f380896b690…` | the file sha256 part X recorded | True (`b8d0f31c8077`) | True | True | `c67a5bd` |
| `edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4` | `RUN/ptq/nemotron-nano-9b-v2/codes_nvfp4_act-fo6.pt` | 4,337,895,473 | `251884a98f6408df…` | the file sha256 part X recorded | True (`22854fc2fac9`) | True | True | `1a2094e` |
| `edgeai-lab/phi-4-GPTQ-BF16Calib-NVFP4` | `RUN/gptqdd/phi4-14b/codes_nvfp4_act-fo6.pt` | 7,667,859,109 | `5c9522f802db3aa7…` | the file sha256 part X recorded | True (`141a919d520c`) | True | True | `c67a5bd` |
| `edgeai-lab/Qwen3.8-27B-GPTQ-BF16Calib-NVFP4` | `RUN/ptq/qwen3.8-27b/codes_nvfp4_act-fo6.pt` | 13,697,654,681 | `f3d7de9ce2bd18a6…` | the file sha256 part X recorded | True (`3c1bc3b2cfd8`) | True | True | `c67a5bd` |
| `edgeai-lab/Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4` | `RUN/zeroshot6/gptq/llama3.2-1b-instruct/codes_nvfp4_act-fo6.pt` | 547,457,897 | `5f65382fe016aec8…` | the file sha256 part X recorded | True (`952850d7d8cf`) | True | True | `3db0072` |
| `edgeai-lab/Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4` | `RUN/zeroshot6/gptq/llama3.1-8b-instruct/codes_nvfp4_act-fo6.pt` | 3,926,070,201 | `06ff8052565c6534…` | the file sha256 part X recorded | True (`d3fe98167972`) | True | True | `3db0072` |

**Where the files come from:**
- The two Llama files were regenerated by FlipQuant main `3db0072` in part X's step 0. Part P computed these codes in-process and kept no file. Their code digests equal part P's records.
- The other six are the main table's own files (parts H / J).

## Scans

The earlier uploads' patterns cover local paths, the host, user names, private repository names, tokens, e-mail addresses and pointers to a reference implementation. They were run over every staged file:
- the codes files' settings strings, module names, archive record names and raw bytes;
- the state's metadata, record names and raw bytes;
- the maps' metadata strings, tile names and raw bytes;
- every line of the text files.

**The rule for binaries** (the coordinator's, 2026-10-15, with its extension). A raw-byte hit in a GPTQ codes file is accepted only if all of these hold:

1. it matches a scan pattern (the token pattern `hf_…`, or another pattern under the extension below);
2. it lies entirely inside a tensor-storage member (`<archive>/data/<n>`) that backs a codes tensor (`q`, the packed 4-bit codes, or `s`, the E4M3 block-scale bytes);
3. the file's pickle, metadata strings and record names have zero hits;
4. the file's sha256 equals its record (for Qwen3-1.7B and Qwen3-8B, which have no earlier file sha256: the code digest equals the main-table record, and the file sha256 is recorded now).

**Extension:** (1) may cover any scan pattern, but only when (2), (3) and the exact-token check all hold. Why the hits occur: the E4M3 scale bytes of these weights fall mostly in 0x58–0x7A, the ASCII letters, so runs of block scales read as lowercase text, and the packed codes occasionally do too. The hits are such runs.

**Other files:** the FOCUS state and the maps may have no hit at all.
- Text files (README, json, LICENSE, NOTICE, USE_POLICY) may have no hit except two accepted kinds, both accepted before in the Llama uploads:
  - Meta's reporting address `LlamaUseReport@meta.com`, inside the official `USE_POLICY.md` files;
  - in the FlipQuant card, the words “commit” / “branch” in its own sentences (the FlipQuant commit is kept; the provenance paragraph), as in `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant`.

**Exact-token check** (the coordinator's condition): every staged file's bytes were searched in-process for the 2 Hugging Face tokens present in this machine's token files, and for each one's first 15 characters. The tokens were never printed, logged or written. Exact-token check: 0 hits on every file (51 files).

| repository | raw-byte hits | inside `s` | inside `q` | elsewhere | metadata hits | exact-token check | within the rule |
|---|---:|---:|---:|---:|---:|---|---|
| `edgeai-lab/Llama-3.2-1B-Instruct-FOCUS-NVFP4` (focus.pt) | 0 | – | – | 0 | 0 | 0 hits | yes |
| `edgeai-lab/Qwen3-1.7B-GPTQ-BF16Calib-NVFP4` | 1,094 | 1,094 | 0 | 0 | 0 | 0 hits | yes |
| `edgeai-lab/Qwen3-8B-GPTQ-BF16Calib-NVFP4` | 1,763 | 1,762 | 1 | 0 | 0 | 0 hits | yes |
| `edgeai-lab/Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4` | 1,357 | 1,356 | 1 | 0 | 0 | 0 hits | yes |
| `edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4` | 895 | 895 | 0 | 0 | 0 | 0 hits | yes |
| `edgeai-lab/phi-4-GPTQ-BF16Calib-NVFP4` | 962 | 961 | 1 | 0 | 0 | 0 hits | yes |
| `edgeai-lab/Qwen3.8-27B-GPTQ-BF16Calib-NVFP4` | 12,583 | 12,582 | 1 | 0 | 0 | 0 hits | yes |
| `edgeai-lab/Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4` | 393 | 393 | 0 | 0 | 0 | 0 hits | yes |
| `edgeai-lab/Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4` | 1,469 | 1,469 | 0 | 0 | 0 | 0 hits | yes |
| `edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant` (map files) | 0 | – | – | 0 | 0 | 0 hits | yes |

Hits of a pattern other than the token pattern (accepted under the extension; the matched bytes are not quoted):

- `Qwen3-8B-GPTQ-BF16Calib-NVFP4`: a project-name pattern (5 bytes) in `model.layers.24.self_attn.o_proj` `q`
- `Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4`: a project-name pattern (5 bytes) in `model.layers.7.self_attn.o_proj` `q`
- `phi-4-GPTQ-BF16Calib-NVFP4`: a project-name pattern (5 bytes) in `model.layers.3.mlp.gate_up_proj` `q`
- `Qwen3.8-27B-GPTQ-BF16Calib-NVFP4`: a path pattern (5 bytes) in `model.language_model.layers.19.mlp.up_proj` `q`
- `Qwen3.8-27B-GPTQ-BF16Calib-NVFP4`: a username pattern (8 bytes) in `model.language_model.layers.29.mlp.gate_proj` `s`

## Licenses

- **Llama-3.2-1B-Instruct** (FOCUS, FlipQuant and GPTQ):
  - `LICENSE` (the base repository's `LICENSE.txt`) and `USE_POLICY.md`, downloaded read-only with the active token;
  - `NOTICE`: the agreement's notice sentence verbatim (section 1.b.iii), checked to appear in `LICENSE`;
  - “Built with Llama” is in all three cards, and all three names start with “Llama”.
- **Llama-3.1-8B-Instruct:** the files of part P's upload (the base repository's `LICENSE` and `USE_POLICY.md`; the base-model release repository's `NOTICE`).
- **Nemotron-Nano-9B-v2:** the NVIDIA Open Model License Agreement and its attribution `NOTICE`, as in the FlipQuant / FOCUS repositories.
- **Qwen3-1.7B, Qwen3-8B:** the base repositories' Apache-2.0 `LICENSE`. **Phi-4:** its MIT `LICENSE`.
- **Mistral-7B-Instruct-v0.3 and Qwen3.8-27B:** the base repositories have no license file, and their cards declare Apache-2.0. `LICENSE` is the Apache License 2.0 text from apache.org (as in the Mistral FOCUS repository).

## Visibility

**The user's decision** (2026-10-15, relayed by the coordinator): make these twelve repositories public, each only after its upload is verified, and change no other repository.
- **Who made the change:** the user. This session did not; its permission system refused the change (“Create Public Surface”), and the coordinator's message said that in that case the user makes it.
- **Visibility now:** read-only checks without a token at 2026-10-16 21:14 UTC; record in `visibility.json`.
- **For a public repository,** the checks are:
  - the model info is readable;
  - the file list equals the staged list;
  - the README downloads and equals the staged one;
  - the head equals the verified upload's.
- **The time of each change** is not recorded by the Hub's API. Each repository was public at the time of the check.

| repository | upload verified | visibility | model info readable | file list = staged | README = staged | head = verified upload |
|---|---|---|---|---|---|---|
| `edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant` | True (this upload) | public | True | True | True | True |
| `edgeai-lab/Llama-3.2-1B-Instruct-FOCUS-NVFP4` | True (this upload) | public | True | True | True | True |
| `edgeai-lab/Qwen3-1.7B-GPTQ-BF16Calib-NVFP4` | True (this upload) | public | True | True | True | True |
| `edgeai-lab/Qwen3-8B-GPTQ-BF16Calib-NVFP4` | True (this upload) | public | True | True | True | True |
| `edgeai-lab/Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4` | True (this upload) | public | True | True | True | True |
| `edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4` | True (this upload) | public | True | True | True | True |
| `edgeai-lab/phi-4-GPTQ-BF16Calib-NVFP4` | True (this upload) | public | True | True | True | True |
| `edgeai-lab/Qwen3.8-27B-GPTQ-BF16Calib-NVFP4` | True (this upload) | public | True | True | True | True |
| `edgeai-lab/Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4` | True (this upload) | public | True | True | True | True |
| `edgeai-lab/Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4` | True (this upload) | public | True | True | True | True |
| `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant` | True (part P's upload) | public | True | True | True | True |
| `edgeai-lab/Llama-3.1-8B-Instruct-FOCUS-NVFP4` | True (part P's upload) | public | True | True | True | True |

## Disk

Staging uses reflinks (`cp --reflink=always`): the staged and source files map the same physical extents. However, /home's XFS project quota (`prjquota`) charges each staged copy its full size, so the staging costs about 37 GiB of quota.
- Nothing was deleted.
- The uploads used an xet cache on /dev/shm.
- The maps' staging is about 3 MB.

