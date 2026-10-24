# Hugging Face upload: the part AA 1,024-window artifacts (part AB, 2026-10-23)

The user's decision (relayed by the coordinator, 2026-10-23): the eight part AA models' 1,024-window calibrations as 24 new **private** repositories under `edgeai-lab`: `<Base>-FlipQuant-Calib1k` (the FQ16-1k map), `<Base>-FOCUS-NVFP4-Calib1k` (the FOCUS-1k state) and `<Base>-GPTQ-BF16Calib-NVFP4-Calib1k` (the GPTQ‡-1k codes). PROTOCOL.md part AB; script `experiments/paper_eval/hf_calib1k.py`.
- **Account:** checked first (`edgeai-lab`); no token was printed or changed. No existing repository was touched.
- **Order of operations:** each repository was created private and checked private (anonymous HTTP 401) before any file went up, then uploaded and verified.
- **Verification:** every file's sha256 against the staged file's: a download into /dev/shm for files up to 1 GiB, the Hub's server-side LFS sha256 and size for larger ones.

## Uploaded

Visibility: at 2026-10-24 08:52 UTC, without a token.

| repository | kind | files | sha256 = staged (method) | visibility now | anonymous at upload | head | card YAML (Hub) | README = staged |
|---|---|---:|---|---|---:|---|---|---|
| `edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant-Calib1k` | flipquant | 6 | equal (6 downloaded) | public | 401 | `9659313a5634` | True | True |
| `edgeai-lab/Llama-3.2-1B-Instruct-FOCUS-NVFP4-Calib1k` | focus | 6 | equal (6 downloaded) | public | 401 | `3ae7de47153e` | True | True |
| `edgeai-lab/Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4-Calib1k` | gptq | 6 | equal (6 downloaded) | public | 401 | `60d9f4012262` | True | True |
| `edgeai-lab/phi-4-FlipQuant-Calib1k` | flipquant | 4 | equal (4 downloaded) | public | 401 | `2268c8372220` | True | True |
| `edgeai-lab/phi-4-FOCUS-NVFP4-Calib1k` | focus | 4 | equal (3 downloaded, 1 by Hub LFS sha256) | public | 401 | `4470e9e32cd5` | True | True |
| `edgeai-lab/phi-4-GPTQ-BF16Calib-NVFP4-Calib1k` | gptq | 4 | equal (3 downloaded, 1 by Hub LFS sha256) | public | 401 | `fc2ccb6fca6e` | True | True |
| `edgeai-lab/Mistral-7B-Instruct-v0.3-FlipQuant-Calib1k` | flipquant | 4 | equal (4 downloaded) | public | 401 | `07b694ac2673` | True | True |
| `edgeai-lab/Mistral-7B-Instruct-v0.3-FOCUS-NVFP4-Calib1k` | focus | 4 | equal (3 downloaded, 1 by Hub LFS sha256) | public | 401 | `1bdc82456d77` | True | True |
| `edgeai-lab/Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4-Calib1k` | gptq | 4 | equal (3 downloaded, 1 by Hub LFS sha256) | public | 401 | `66ff133594d3` | True | True |
| `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant-Calib1k` | flipquant | 6 | equal (6 downloaded) | public | 401 | `c643f64609f3` | True | True |
| `edgeai-lab/Llama-3.1-8B-Instruct-FOCUS-NVFP4-Calib1k` | focus | 6 | equal (5 downloaded, 1 by Hub LFS sha256) | public | 401 | `7bb34c848adb` | True | True |
| `edgeai-lab/Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4-Calib1k` | gptq | 6 | equal (5 downloaded, 1 by Hub LFS sha256) | public | 401 | `5b9a6efefb81` | True | True |
| `edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FlipQuant-Calib1k` | flipquant | 5 | equal (5 downloaded) | public | 401 | `8b64d55bd492` | True | True |
| `edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FOCUS-NVFP4-Calib1k` | focus | 5 | equal (4 downloaded, 1 by Hub LFS sha256) | public | 401 | `c83a6195cedc` | True | True |
| `edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4-Calib1k` | gptq | 5 | equal (4 downloaded, 1 by Hub LFS sha256) | public | 401 | `3dbb35d0b003` | True | True |
| `edgeai-lab/Qwen3.8-27B-FlipQuant-Calib1k` | flipquant | 4 | equal (4 downloaded) | public | 401 | `d3d28f25dbd6` | True | True |
| `edgeai-lab/Qwen3.8-27B-FOCUS-NVFP4-Calib1k` | focus | 4 | equal (3 downloaded, 1 by Hub LFS sha256) | public | 401 | `c79202fd5400` | True | True |
| `edgeai-lab/Qwen3.8-27B-GPTQ-BF16Calib-NVFP4-Calib1k` | gptq | 4 | equal (3 downloaded, 1 by Hub LFS sha256) | public | 401 | `70ef48488562` | True | True |
| `edgeai-lab/Qwen3-1.7B-FlipQuant-Calib1k` | flipquant | 4 | equal (4 downloaded) | public | 401 | `b7681af92adc` | True | True |
| `edgeai-lab/Qwen3-1.7B-FOCUS-NVFP4-Calib1k` | focus | 4 | equal (4 downloaded) | public | 401 | `432e8987e9a6` | True | True |
| `edgeai-lab/Qwen3-1.7B-GPTQ-BF16Calib-NVFP4-Calib1k` | gptq | 4 | equal (4 downloaded) | public | 401 | `033d837844f0` | True | True |
| `edgeai-lab/Qwen3-8B-FlipQuant-Calib1k` | flipquant | 4 | equal (4 downloaded) | public | 401 | `e926ed491f2e` | True | True |
| `edgeai-lab/Qwen3-8B-FOCUS-NVFP4-Calib1k` | focus | 4 | equal (3 downloaded, 1 by Hub LFS sha256) | public | 401 | `bcc75ce33a67` | True | True |
| `edgeai-lab/Qwen3-8B-GPTQ-BF16Calib-NVFP4-Calib1k` | gptq | 4 | equal (3 downloaded, 1 by Hub LFS sha256) | public | 401 | `83df99216e71` | True | True |

## Sources

Each source was verified against its part AA job record by a fresh sha256 of the whole file before staging.
- **Maps:** `fq16_1024/map.pt`, against `map_file_sha256`. The meta was cleaned as for the earlier FlipQuant repositories: the run and calibration-record paths are relative, the GitHub owner is removed from the extension reference, and the reference implementation's branch and commit are removed. The tiles and every tensor record are bit-identical; both sha256 are in each card.
- **FOCUS states:** as is, against `state_sha256` (the pilot's two in RUN, the others on /vault).
- **GPTQ‡ codes:** as is, against `cache_sha256`. The code digest equals the calibration's, the key is the main-table variant at 1,024 windows, and the card's PyTorch snippet reproduces FlipQuant's dequantization bit for bit.
- **Staging:** the cards, summaries, license files and cleaned maps were written to /home/dev/calib1k_hf. The FOCUS states and the codes files were symbolic links to their verified sources (/home's 10 GB floor), so every scan, hash and upload read the source.

## Scans

The earlier uploads' patterns (local paths, the host, user names, private repository names, tokens, e-mail addresses, pointers to a reference implementation) over every staged file: the maps' metadata strings, tile names and raw bytes; the states' metadata, record names and raw bytes; the codes files' settings strings, module names, record names and raw bytes; every line of the text files.
- **Maps and FOCUS states:** no hit.
- **Text files:** only the kinds accepted before (Meta's reporting address inside the official `USE_POLICY.md`; the words “commit” / “branch” in the FlipQuant card's own sentences).
- **Codes files:** the GPTQ upload's binary rule (2026-10-15, with its extension). A raw-byte hit is accepted only if it lies inside a tensor-storage member backing a codes tensor (`q` or `s`), the file's pickle, metadata strings and record names have no hit, the file's sha256 equals its record, and the exact-token check is clean.

| repository | raw-byte hits | inside `s` | inside `q` | metadata hits | within the rule |
|---|---:|---:|---:|---:|---|
| `Llama-3.2-1B-Instruct-GPTQ-BF16Calib-NVFP4-Calib1k` | 393 | 393 | 0 | 0 | yes |
| `phi-4-GPTQ-BF16Calib-NVFP4-Calib1k` | 961 | 961 | 0 | 0 | yes |
| `Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4-Calib1k` | 1,357 | 1,356 | 1 | 0 | yes |
| `Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4-Calib1k` | 1,470 | 1,469 | 1 | 0 | yes |
| `NVIDIA-Nemotron-Nano-9B-v2-GPTQ-BF16Calib-NVFP4-Calib1k` | 895 | 895 | 0 | 0 | yes |
| `Qwen3.8-27B-GPTQ-BF16Calib-NVFP4-Calib1k` | 12,582 | 12,582 | 0 | 0 | yes |
| `Qwen3-1.7B-GPTQ-BF16Calib-NVFP4-Calib1k` | 1,094 | 1,094 | 0 | 0 | yes |
| `Qwen3-8B-GPTQ-BF16Calib-NVFP4-Calib1k` | 1,763 | 1,762 | 1 | 0 | yes |

Hits of a pattern other than the token pattern (accepted under the extension; the matched bytes are not quoted):

- `Mistral-7B-Instruct-v0.3-GPTQ-BF16Calib-NVFP4-Calib1k`: a project-name pattern (5 bytes) in `model.layers.7.self_attn.o_proj` `q`
- `Llama-3.1-8B-Instruct-GPTQ-BF16Calib-NVFP4-Calib1k`: a project-name pattern (5 bytes) in `model.layers.29.mlp.gate_proj` `q`
- `Qwen3.8-27B-GPTQ-BF16Calib-NVFP4-Calib1k`: a username pattern (8 bytes) in `model.language_model.layers.29.mlp.gate_proj` `s`
- `Qwen3-8B-GPTQ-BF16Calib-NVFP4-Calib1k`: a project-name pattern (5 bytes) in `model.layers.24.self_attn.o_proj` `q`

**Exact-token check:** every staged file's bytes were searched in-process for the 2 Hugging Face tokens in this machine's token files and their 15-character prefixes; the tokens were never printed, logged or written: 0 hits on all 111 files.

## Events

- **Commit rate limit.** The 24th repository's commit was refused with HTTP 429 ("repository commit creation, 128 per hour") after 46 commits by this upload; the limit is treated as account-wide. upload_large_folder retried about twice a second. The coordinator authorized stopping the uploader (12:02 UTC), then one commit attempt at least 65 minutes later, at most one per 10 minutes and six in all.
- **Private storage limit.** The first attempt (13:08 UTC) was refused with HTTP 400: "Private storage limit reached for user edgeai-lab" (the 23 verified repositories hold 86.7 GB). No further attempt was made; the decision went to the user. After the 23 were made public (below), the 24th was uploaded privately with the same throttled single-commit retry.
- **Commit attempts of the 24th** (hf_upload.json `gentle_attempts`): 2026-10-23 13:08 UTC: HTTP 400; 2026-10-24 08:49 UTC: committed.

## Licenses

As in the earlier repositories of each model: the Llama 3.2 / 3.1 Community License files (`LICENSE`, `USE_POLICY.md`, `NOTICE`; “Built with Llama”; every name starts with “Llama”), the NVIDIA Open Model License and its `NOTICE` (Nemotron), Apache-2.0 (Qwen3; Mistral-7B-Instruct-v0.3 and Qwen3.8-27B with the apache.org text, as their base repositories have no license file) and MIT (Phi-4).

## Visibility

**The user's request:** make the uploaded repositories public. **Who made the change:** the coordinator, in its own session, at the user's direct request (`update_repo_settings(private=False)`), first for the 23 verified repositories and later for the 24th once its upload was verified. This session made nothing public and changed no visibility.
- **Visibility now:** this session's read-only check without a token at 2026-10-24 08:52 UTC (`visibility.json`); the table above lists each repository's state.
- **For a public repository** the check also compares, without a token, its file list with the staged list, its README with the staged one and its head with the verified upload's.

Files here: `stage.json` (sanitized), `hf_upload.json`, `cards.json`, `rule_check.json`, `token_check.json`, `nonhf_hits.json`, `visibility.json`, `cards/<repo>/{README.md, ppl_summary.json}`.
