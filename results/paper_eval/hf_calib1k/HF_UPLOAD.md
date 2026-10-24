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

## Addendum (2026-10-24): the 8x64 / 256x64 maps (part AB amendment 1)

The user's request (relayed by the coordinator, 2026-10-24): part AA amendment 6's maps (FlipQuant 8x64 and 256x64 at 1,024 windows) added to the eight public `<Base>-FlipQuant-Calib1k` repositories, in the layout of the 256-window `<Base>-FlipQuant` repositories. The repositories were public before and after; every commit is public. No visibility was changed and no other repository was touched.
- **Authorization:** this session's permission system refused the first commit ("Create Public Surface"); the part stopped and nothing was uploaded. The user then authorized it directly in this session (2026-10-24): only these eight repositories; per repository the two maps added and README.md and ppl_summary.json updated, exactly as staged; no other repository, file or visibility change. The uploads ran after that authorization.
- **Sources:** each map's sha256 equals its amendment 6 job record. The meta is cleaned as for the 16x64 map; tiles and every tensor record are bit-identical. The 16x64 map, re-staged the same way, reproduces the uploaded file byte for byte.
- **Cards:** README.md and ppl_summary.json regenerated for the three tiles, with numbers from part AA's record at paper-eval 5dcb1ff: the three maps and their E0M3 shares (and the 256-window maps'), the settings, the results against each map's 256-window version, FourOverSix and the 16x64 map, and a note on 256x64.
- **Gates:** no hit in the maps' metadata, tile names or raw bytes; text files only the accepted kinds; the exact-token check: 0 hits on all 53 staged files.
- **Upload:** one commit per repository (`Add the 8x64 and 256x64 maps (1,024 windows); update the card`), adding the two maps and updating README.md and ppl_summary.json.
- **Verification, without a token:** the file list, every file's sha256 by download, the card metadata, the head and its commit; the Hub's YAML validation, README = staged and a local render.

Visibility at 2026-10-24 14:19 UTC, without a token: public.

| repository | head before → after | files (downloaded, = staged) | card YAML (Hub) | README = staged | attempts |
|---|---|---:|---|---|---|
| `edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant-Calib1k` | `9659313a5634` → `f447d81c6fdf` | 8 of 8 | True | True | 2026-10-24 14:16 UTC committed |
| `edgeai-lab/phi-4-FlipQuant-Calib1k` | `2268c8372220` → `a477cc860e39` | 6 of 6 | True | True | 2026-10-24 14:16 UTC committed |
| `edgeai-lab/Mistral-7B-Instruct-v0.3-FlipQuant-Calib1k` | `07b694ac2673` → `2922fffdf530` | 6 of 6 | True | True | 2026-10-24 14:17 UTC committed |
| `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant-Calib1k` | `c643f64609f3` → `ab08ef72c40a` | 8 of 8 | True | True | 2026-10-24 14:17 UTC committed |
| `edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FlipQuant-Calib1k` | `8b64d55bd492` → `2a28a31bb8dc` | 7 of 7 | True | True | 2026-10-24 14:17 UTC committed |
| `edgeai-lab/Qwen3.8-27B-FlipQuant-Calib1k` | `d3d28f25dbd6` → `06b068a3b68b` | 6 of 6 | True | True | 2026-10-24 14:17 UTC committed |
| `edgeai-lab/Qwen3-1.7B-FlipQuant-Calib1k` | `b7681af92adc` → `faed670920dd` | 6 of 6 | True | True | 2026-10-24 14:18 UTC committed |
| `edgeai-lab/Qwen3-8B-FlipQuant-Calib1k` | `e926ed491f2e` → `28cbd2bf9189` | 6 of 6 | True | True | 2026-10-24 14:18 UTC committed |

| repository | map | source = job record | E0M3 share | uploaded sha256 |
|---|---|---|---:|---|
| `Llama-3.2-1B-Instruct-FlipQuant-Calib1k` | `flipquant_8x64.pt` | True | 11.28 % | `c45684a0abb34d48…` |
| `Llama-3.2-1B-Instruct-FlipQuant-Calib1k` | `flipquant_256x64.pt` | True | 39.69 % | `309f115a041cdcdc…` |
| `phi-4-FlipQuant-Calib1k` | `flipquant_8x64.pt` | True | 2.85 % | `657e80d319daefda…` |
| `phi-4-FlipQuant-Calib1k` | `flipquant_256x64.pt` | True | 10.32 % | `d292e9e7f6601d94…` |
| `Mistral-7B-Instruct-v0.3-FlipQuant-Calib1k` | `flipquant_8x64.pt` | True | 4.09 % | `e3fa4365e2f30b3a…` |
| `Mistral-7B-Instruct-v0.3-FlipQuant-Calib1k` | `flipquant_256x64.pt` | True | 14.81 % | `dbec75c2fc97dafc…` |
| `Llama-3.1-8B-Instruct-FlipQuant-Calib1k` | `flipquant_8x64.pt` | True | 4.30 % | `989ef5ec3d992101…` |
| `Llama-3.1-8B-Instruct-FlipQuant-Calib1k` | `flipquant_256x64.pt` | True | 15.05 % | `8c2fb731fdf5e3fc…` |
| `NVIDIA-Nemotron-Nano-9B-v2-FlipQuant-Calib1k` | `flipquant_8x64.pt` | True | 4.29 % | `bbfba43713237770…` |
| `NVIDIA-Nemotron-Nano-9B-v2-FlipQuant-Calib1k` | `flipquant_256x64.pt` | True | 15.98 % | `8738a1c478ee3995…` |
| `Qwen3.8-27B-FlipQuant-Calib1k` | `flipquant_8x64.pt` | True | 2.26 % | `1a07a2f3ec96f46b…` |
| `Qwen3.8-27B-FlipQuant-Calib1k` | `flipquant_256x64.pt` | True | 7.05 % | `73b703adf5a1fb14…` |
| `Qwen3-1.7B-FlipQuant-Calib1k` | `flipquant_8x64.pt` | True | 6.66 % | `7d0e2218a36d90fd…` |
| `Qwen3-1.7B-FlipQuant-Calib1k` | `flipquant_256x64.pt` | True | 21.70 % | `de69f6c867bc5105…` |
| `Qwen3-8B-FlipQuant-Calib1k` | `flipquant_8x64.pt` | True | 3.63 % | `262f1569777a66d7…` |
| `Qwen3-8B-FlipQuant-Calib1k` | `flipquant_256x64.pt` | True | 9.96 % | `ec48d0ecf9b16c91…` |

Files here: `stage_a1.json` (sanitized), `upload_a1.json`, `cards_a1.json`, `token_check_a1.json`; `cards/<repo>/` now holds the updated cards of the FlipQuant repositories.

