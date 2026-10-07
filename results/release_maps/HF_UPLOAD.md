# Hugging Face upload of the release maps (2026-10-07)

The 24 release maps (8 models × 8x64 / 16x64 / 256x64) are on the Hugging Face Hub, one **private** repository per model under `edgeai-lab`, uploaded with that account's active token on the user's decision. No repository is public, and no token or account setting was changed.

| model | repository | HF commit | files |
|---|---|---|---|
| qwen3-1.7b | [edgeai-lab/Qwen3-1.7B-FlipQuant](https://huggingface.co/edgeai-lab/Qwen3-1.7B-FlipQuant) | `f863da0d890d4d5925670d33399d0da630eb7884` | `.gitattributes`, `LICENSE`, `README.md`, `flipquant_16x64.pt`, `flipquant_256x64.pt`, `flipquant_8x64.pt`, `ppl_summary.json` |
| qwen3-8b | [edgeai-lab/Qwen3-8B-FlipQuant](https://huggingface.co/edgeai-lab/Qwen3-8B-FlipQuant) | `adac60c4eedf68127e220ae9fe6b1e8e34ffdbbc` | `.gitattributes`, `LICENSE`, `README.md`, `flipquant_16x64.pt`, `flipquant_256x64.pt`, `flipquant_8x64.pt`, `ppl_summary.json` |
| mistral-7b | [edgeai-lab/Mistral-7B-Instruct-v0.3-FlipQuant](https://huggingface.co/edgeai-lab/Mistral-7B-Instruct-v0.3-FlipQuant) | `62a727be895efc77293a1b14453d02f9b90e6a67` | `.gitattributes`, `README.md`, `flipquant_16x64.pt`, `flipquant_256x64.pt`, `flipquant_8x64.pt`, `ppl_summary.json` |
| mistral-7b-base | [edgeai-lab/Mistral-7B-v0.3-FlipQuant](https://huggingface.co/edgeai-lab/Mistral-7B-v0.3-FlipQuant) | `dad31eb63db243d3afb60e88df8a438876d07496` | `.gitattributes`, `README.md`, `flipquant_16x64.pt`, `flipquant_256x64.pt`, `flipquant_8x64.pt`, `ppl_summary.json` |
| llama3.1-8b | [edgeai-lab/Llama-3.1-8B-FlipQuant](https://huggingface.co/edgeai-lab/Llama-3.1-8B-FlipQuant) | `89ec75d6b0e825bade8cb9527a892d517dd08766` | `.gitattributes`, `LICENSE`, `NOTICE`, `README.md`, `USE_POLICY.md`, `flipquant_16x64.pt`, `flipquant_256x64.pt`, `flipquant_8x64.pt`, `ppl_summary.json` |
| nemotron-nano-9b-v2 | [edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FlipQuant](https://huggingface.co/edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FlipQuant) | `24c446d656e0c86e3bb947efff0484c870e87267` | `.gitattributes`, `LICENSE`, `NOTICE`, `README.md`, `flipquant_16x64.pt`, `flipquant_256x64.pt`, `flipquant_8x64.pt`, `ppl_summary.json` |
| phi4-14b | [edgeai-lab/phi-4-FlipQuant](https://huggingface.co/edgeai-lab/phi-4-FlipQuant) | `548b81db1128e81344ae5a5a3c7cc8cbaa48dc11` | `.gitattributes`, `README.md`, `flipquant_16x64.pt`, `flipquant_256x64.pt`, `flipquant_8x64.pt`, `ppl_summary.json` |
| qwen3.8-27b | [edgeai-lab/Qwen3.8-27B-FlipQuant](https://huggingface.co/edgeai-lab/Qwen3.8-27B-FlipQuant) | `74c52705f65e3b12a45270755f4fc44d42a15a78` | `.gitattributes`, `README.md`, `flipquant_16x64.pt`, `flipquant_256x64.pt`, `flipquant_8x64.pt`, `ppl_summary.json` |

## What was uploaded

- `flipquant_{8x64,16x64,256x64}.pt`: the release maps with only their `meta` changed (below); the tiles are bit-identical.
- `README.md`: the final model card (from the release directory's draft; below).
- `ppl_summary.json`: PPL per unit with BF16 and FourOverSix, the paired ΔNLL ± 2 SE against FourOverSix with verdicts, and the evaluation settings (native sm_120, the paper convention, datasets, revisions, windows); no local paths, host or user names.
- License files: Qwen3-1.7B and Qwen3-8B `LICENSE`; Llama-3.1-8B `LICENSE`, `USE_POLICY.md`, `NOTICE`; Nemotron `LICENSE`, `NOTICE`. Mistral (both), Phi-4 and Qwen3.8-27B have no license file in their base repositories, so their cards' license field carries it.
- **Not uploaded:** the release directory's `records/` (they hold local paths) and `ppl/` (the per-window reports).

## The maps' meta changes (all 24; the user's decisions)

| field | release file | uploaded file |
|---|---|---|
| `source.run` | the local run directory | relative, e.g. `runs/llama3.1-8b_16x64/run` |
| `source.calibration_record` | the local record path | relative, e.g. `tmopt_data/llama8b/calibration/report.json` (its sha256 field kept) |
| `fit_extension.flipquant_reference` | `brian030128/flipquant flipquant/data.py calibration_sets_extended @ b1c4123` | `flipquant flipquant/data.py calibration_sets_extended @ b1c4123` |
| `branch`, `commit` | the reference implementation's branch and commit | removed (the FlipQuant commit `flipquant.commit` = 120173a and the neutral `repo` / `trainer` strings kept) |

Each uploaded map was written with `flipquant.maps.save` and checked: torch.equal per module, every tensor record of the archive byte-identical to the release file's, and `flipquant.maps.load` reads it. sha256, release file → uploaded file (also in each card's provenance lines and in `hf/stage.json`):

| model | file | release file | uploaded file |
|---|---|---|---|
| qwen3-1.7b | `flipquant_8x64.pt` | `4ddfc7361e60b1773f030add26e69a409620fbb0988de3130da0d3d3c90a5361` | `a6a13d610d3319537dc503f36899483cf6c2e2512c5d25dd4c278940299e241e` |
| qwen3-1.7b | `flipquant_16x64.pt` | `98ca73091a4e33d840d4762c75fefbaf7c13fad915448eda9f2e329bed39eda9` | `d8d231fa5ce527f1a6608d6eb77abcb8fc741a6197c2aea4721e0daa750a787c` |
| qwen3-1.7b | `flipquant_256x64.pt` | `91df20cfb8ca39056d7166215afadba71f3c6b449adccc48deb0cf6f3a3e3abe` | `0f0c82d7b82315d954fd898c6dfb97d252e6a64e5b7866b207ce68b92d8d7ab8` |
| qwen3-8b | `flipquant_8x64.pt` | `f0760449f1aba9665b042d2b44c710c6e3bf4a0a6d458983bd59723b2d82847e` | `2bc06672c955f900d1088758a64264db5f79b94efbd7c6ace08e46f65ca2a659` |
| qwen3-8b | `flipquant_16x64.pt` | `3a363d75ea973a521e1fa4411c7ee85c1f1b8f88e3ed0360c0d14272c2151fc7` | `e727b2a7f497ad7602121b6ca0d35972d59b3d904d742e128b94d8b6a921bb76` |
| qwen3-8b | `flipquant_256x64.pt` | `340fb09949a1f7980231c8a6b5bb35480d90a89634ae96c4ed777bd325e4e89d` | `c388ed5de3f1f5a9887e091786f383be87f21aadf0d1ae1aa9ee685763912137` |
| mistral-7b | `flipquant_8x64.pt` | `765729fb1586c02264b0c8907d7be5d19734a9d19d3e23dc687a878f1079ab14` | `c986fea02b128abe50308f389a9b81c59e2420e0c591627870d1fd1ada106674` |
| mistral-7b | `flipquant_16x64.pt` | `4ddbdd11c0fb79eb7051412a0879345acecb56d583cc98c02a2f9a6a388c071e` | `f19fa89c47fcdf3fd4fddae3ab6df3c93658ce3c25f52e5ff5c788afb8b2a4f5` |
| mistral-7b | `flipquant_256x64.pt` | `a9e3732846fc82e7cdf23889387cc0a4e9627a431cb49252344b8caaeeda9aec` | `5dc243b3385cfa4876f520d5d6cfafa96a68fdfd3e3e4a6ee36bdba1552405e4` |
| mistral-7b-base | `flipquant_8x64.pt` | `9c2472331a1ec8bfd140f42f4ed430f5f25ac64d24dab53147a5dabb8d8d8153` | `c1c55bc56021696e0529b8a0bd91ddc5c626460a12b2152bf34b09e8caa6ebc4` |
| mistral-7b-base | `flipquant_16x64.pt` | `f8e4421784ed7985b9e4d8a1ba0650c5d02962cce0715a9b8b94c5e1447a19a1` | `24c67f3598c8f5b4541135e1eb4750a071070af5f076efadffde35d6e2b245de` |
| mistral-7b-base | `flipquant_256x64.pt` | `d8ea6a24aca0c1bb8544d54b0cbe0352cbbd73206e6f28f2c398f06e2cd71870` | `4d5927b0325df9e624ff5545e292ace202ad74903a331e19a6d0b40ed033a99b` |
| llama3.1-8b | `flipquant_8x64.pt` | `946fd3c4b4017086a8d2cc8bfea1445ceda87a217e257ee923a2f516f8ba666c` | `ad40f1827204a3166375954774b4b3676790e036b09c1ff46a18e6d03c9b1613` |
| llama3.1-8b | `flipquant_16x64.pt` | `d4ccc18b61a6db781d7799723c47a905ca69407d72a7df009e04929ea877e150` | `c9d10f3035691492fb4da24880510fb511d51f2f56081cddba0b47d15be08d41` |
| llama3.1-8b | `flipquant_256x64.pt` | `fcff3f58c18e786580b9c408dd666a0dc22cc2c547b43d17b9ac6051f38aa49c` | `260fe3463792c20a9ffb38dcdc6485f6b1d24eba8e71a5327eebca78d521c670` |
| nemotron-nano-9b-v2 | `flipquant_8x64.pt` | `9ecc2ded8d5c1a8f7b677d556a245dd5515c5bea20b313889a11b0f32e20aad5` | `6c37619cef87411e53a1e06b10518b6ef4c11bad7e141b2a16c52c452321dabf` |
| nemotron-nano-9b-v2 | `flipquant_16x64.pt` | `6bae7360fc0083bb3e2bd57c1f1839640845b5188a58adb850ba33883cbd2d1e` | `8eed5296d693a6031c50554bd0ee5837b15cd88c65b992b118fc5838c48c9df1` |
| nemotron-nano-9b-v2 | `flipquant_256x64.pt` | `53c6d91fc6be656ad4196cbe7586d9ee75a5007b9f693ff0a0d43e13e0447e76` | `4ab8f3137a96466cc0ea9e1762232b25bedd3825e63a004432faaa88b4502433` |
| phi4-14b | `flipquant_8x64.pt` | `fc45a7acffe95ac4bd3bbeb3b7eaa5fff513f1029b00965b51b991bec5b00dc6` | `9131bff768fce81dbb5e4ad559d1c3dc95c356145a5c705f301aacb9c2a58e94` |
| phi4-14b | `flipquant_16x64.pt` | `79c81a945a6e9e252e72581b4abb6f2d3c06938aa45a16e10edb831bd88c075d` | `8ea582050b40943a5abca4c44dc3edd0f86922f422fe77991ba7b04d75cb08fb` |
| phi4-14b | `flipquant_256x64.pt` | `53d9f46fedeb579141a5a1575e69e92ba6a084f97247ee68328f65eefd871e89` | `145f060338481ac66aff4e9f73842c1f329ca731d23db88cbf2e2d194c8e38f7` |
| qwen3.8-27b | `flipquant_8x64.pt` | `17c8e5c79c9c3900d448765ba7324c3b8149cc5883f208e0a18e15ae4ef88ad0` | `f9e0dec72ecd4278a7f22f773390b2205293119275285cbdcf773eef7ceb4a5a` |
| qwen3.8-27b | `flipquant_16x64.pt` | `9b051a0c76796d12236bbdd3c80bb0815db7c00f5f370110c5e7dc5d04cc710e` | `050f0eead1bb2199bdcdb36b489618ef7177b0e325eefe5e412b067965d43e8d` |
| qwen3.8-27b | `flipquant_256x64.pt` | `914e843b1ce4aa494ff0a093188a4435f45dac81f9cb56fe6147c13901188e5a` | `59b38ae511226745a401be7616ed9a4049841fb797918badcddd3911c99fd30f` |

## License files

- **Llama-3.1-8B:** `LICENSE` and `USE_POLICY.md` are meta-llama/Llama-3.1-8B's own files at the calibration revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` (sha256 `64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369` and `a568f2ebc73cec3fd74ba2afd992d4e945a8c7a9d851f9b66163aac834b7b859`). The local snapshot holds neither, so they were downloaded read-only with a stored read token chosen by name (the user's decision): passed to that download only, never printed, the active account unchanged.
  - Against the text first extracted from the base card's gated prompt (after normalizing whitespace, quotes and headings): the official `LICENSE` has one more sentence, “By clicking "I Accept" below or by using or distributing any portion or element of the Llama Materials, you agree to be bound by this Agreement.”; the official `USE_POLICY.md` says “unlicensed uses of Llama 3.1” where the card says “unlicensed uses of Meta Llama 3”. Nothing else differs. The official files are the ones uploaded.
  - `NOTICE`: “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.” The card shows “Built with Llama” under its title, and the repository name starts with “Llama”.
- **NVIDIA-Nemotron-Nano-9B-v2:** the base repository has no license file. `LICENSE` is the NVIDIA Open Model License Agreement (Last Modified: October 24, 2025) from the page the base card links, https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/, fetched 2026-10-07 05:26 UTC (page sha256 `624c40d8cc6da3e6636a7a05bf34cf6e925a4f9f23666f7095468865917013f2`), rendered to text by `hf_licenses.py render`: verbatim paragraphs, the section numbers the page displays (CSS counters), bullets as “•” (sha256 `8cf5f9a673059fc727c03fae31c0d6e365bbf4ef1b7fd9b3099fefe1b49fdda1`; re-rendering reproduces it byte for byte).
  - Cross-check against NVIDIA's PDF of the same agreement (`nvidia-open-model-license-agreements-24-10-2025.pdf`, sha256 `4d2fb590aa9b30c47f2058bff17291df7fa2aa0c1bd775a20703da9bb267cfab`; text decoded through its ToUnicode maps, some glyphs undecodable): the decoded text matches the page except formatting and punctuation. The PDF numbers subsections “1.1.” where the page shows “1.1”, ends definitions 1.4 and 1.5 with a period, and uses curly quotes where the page has straight ones in some definitions. The page, which the card links, is what was uploaded.
  - `NOTICE`: “Licensed by NVIDIA Corporation under the NVIDIA Open Model License” (the agreement's section 3.1). The card's license section records the source URL and retrieval time.
- **Qwen3-1.7B, Qwen3-8B:** `LICENSE` is the base snapshot's own file (Apache 2.0).

## Model cards

From the release directory's drafts:
- removed: the draft banner, the uploader placeholders, the flipquant link placeholder, and references to `records/`, `ppl/` and internal tool names;
- added: Hub front matter (license; `license_name` / `license_link` for Nemotron; `base_model`; tags `quantization`, `fp4`, `nvfp4`, `mixfp4`, `flipquant`), “The FlipQuant code will be released with the paper.” (no link), “released under the same license as the base model”, the license files' sources, the uploaded files' sha256 and the provenance lines (release → uploaded sha256);
- kept: the usage section, the settings (flagged as not the paper's 20-epoch setting), the PPL table against FourOverSix and BF16 with the paper-setting rows, the per-unit time and memory tables, and Qwen3.8-27B's note that the simulated path does not fit one 96 GB GPU;
- **corrected:** the WikiText-2 window count. Every draft said 146; the evaluation records give:

| model | WikiText-2 windows (2048 tokens) | C4 windows |
|---|---:|---:|
| qwen3-1.7b | 146 | 256 |
| qwen3-8b | 146 | 256 |
| mistral-7b | 163 | 256 |
| mistral-7b-base | 163 | 256 |
| llama3.1-8b | 141 | 256 |
| nemotron-nano-9b-v2 | 147 | 256 |
| phi4-14b | 141 | 256 |
| qwen3.8-27b | 145 | 256 |

## Checks

- **Before upload:** every staged file was scanned for local paths, the host name, user names, the private repositories and branch names, e-mail addresses and tokens (in the maps: every meta string, the module names and the raw bytes). The only hit is Meta's reporting address `LlamaUseReport@meta.com` inside the official `USE_POLICY.md`.
- **Creation:** `create_repo(private=True, exist_ok=False)`; none of the names existed. Each repo reported `private` and answered an anonymous API request with HTTP 401 before any file went up.
- **After upload** (`hf/hf_upload.json`):

| repository | private | anonymous API | files as staged | card metadata parsed by the Hub | every file downloaded again, sha256 equal |
|---|---|---:|---|---|---|
| edgeai-lab/Qwen3-1.7B-FlipQuant | True | 401 | yes | license `apache-2.0`, base_model `Qwen/Qwen3-1.7B` | yes |
| edgeai-lab/Qwen3-8B-FlipQuant | True | 401 | yes | license `apache-2.0`, base_model `Qwen/Qwen3-8B` | yes |
| edgeai-lab/Mistral-7B-Instruct-v0.3-FlipQuant | True | 401 | yes | license `apache-2.0`, base_model `mistralai/Mistral-7B-Instruct-v0.3` | yes |
| edgeai-lab/Mistral-7B-v0.3-FlipQuant | True | 401 | yes | license `apache-2.0`, base_model `mistralai/Mistral-7B-v0.3` | yes |
| edgeai-lab/Llama-3.1-8B-FlipQuant | True | 401 | yes | license `llama3.1`, base_model `meta-llama/Llama-3.1-8B` | yes |
| edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FlipQuant | True | 401 | yes | license `other` (nvidia-open-model-license), base_model `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | yes |
| edgeai-lab/phi-4-FlipQuant | True | 401 | yes | license `mit`, base_model `microsoft/phi-4` | yes |
| edgeai-lab/Qwen3.8-27B-FlipQuant | True | 401 | yes | license `apache-2.0`, base_model `Qwen/Qwen3.8-27B` | yes |

## Files here

- `hf/stage.json`: the staged files (sha256, bytes), the maps' meta changes (old → new) and sha256 (release → uploaded), the license sources and the scans.
- `hf/hf_upload.json`: per repository the URL, commit, privacy checks, Hub file list, parsed card metadata and the download check.
- Scripts (`experiments/release_maps/`): `hf_stage.py` (staging and scan), `hf_licenses.py` (the license texts), `hf_upload.py` (create, upload, verify), `hf_record.py` (this record).
