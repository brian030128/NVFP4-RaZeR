# Hugging Face upload of Llama-3.1-8B-Instruct's FlipQuant maps and FOCUS state (2026-10-10)

The user's request (relayed by the coordinator, 2026-10-10): part P's Llama-3.1-8B-Instruct artifacts as two **private** repositories under `edgeai-lab`, in the formats of the release-map repositories and the FOCUS repositories, before the HF identity is switched. The active account was checked first (`edgeai-lab`; no token printed or changed). Each repository was created private and checked private (anonymous 401) before any file went up; then uploaded and verified.

| repository | files | private | anonymous | head | every file's sha256 (download) | card YAML (Hub) | README = staged |
|---|---:|---|---:|---|---|---|---|
| `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant` | 8 | True | 401 | `8ab5916e1cb1` | equal | True | True |
| `edgeai-lab/Llama-3.1-8B-Instruct-FOCUS-NVFP4` | 6 | True | 401 | `228257c43bea` | equal | True | True |

## What was uploaded

**`edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant`**: the three part-P maps with their meta cleaned as for the release (the run directory and calibration-record paths relative, the extension reference without its GitHub owner, the reference implementation's branch and commit removed; tiles and every tensor record bit-identical), the card, `ppl_summary.json`, `LICENSE`, `USE_POLICY.md`, `NOTICE`. sha256 calibration output → uploaded:

- `flipquant_8x64.pt`: `3bf00ea63e0a07371b85fed1d6ffba813d3c23540a494010626ceeb46832b9bf` → `db896cece0fdef908ba1622b23a3c21b8f867e474ceb41439f94821c476f33ce` (83,771 E0M3 tiles of 13,631,488)
- `flipquant_16x64.pt`: `502ec6cfee834782192ed10f8a0d79a8bfddac9fe6cc4b5a8cb3d3fe0dd3ca86` → `3d527e1191f6e8bef3010016875c12419a81cec75e2ba231edf6943f8cd6f131` (52,719 E0M3 tiles of 6,815,744)
- `flipquant_256x64.pt`: `7582c4dad38c93b6ba004ad1c47502144576386e4dc43ae34715d758920a5b08` → `84b009d76e27f883ab53d7c5ca2ed7bf71b7930affacc3de1d13da4aa56732c4` (9,447 E0M3 tiles of 425,984)

**`edgeai-lab/Llama-3.1-8B-Instruct-FOCUS-NVFP4`**: `focus.pt` as is (sha256 `e311a189f43cf9161f837a855346683c1cd7225bb1984cd27eb796751c80c82a`, equal to part P's state; metadata, record names and raw bytes scanned, nothing found), the card (our reproduction of FOCUS, arXiv:2608.01847; not an official release; hyperparameters, the calibration-data difference, per-token FourOverSix activations, what m / q / s2 are; not a quantized checkpoint; PPL, training time and memory), `ppl_summary.json`, `LICENSE`, `USE_POLICY.md`, `NOTICE`.

## License files

`LICENSE` and `USE_POLICY.md` are meta-llama/Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659's own files, downloaded read-only with the active token on 2026-10-10 22:08 UTC (sha256 LICENSE `64e1b2889b7892e6`, USE_POLICY.md `a568f2ebc73cec3f`; LICENSE equal to the base model's: True, USE_POLICY.md equal to the base model's: True). `NOTICE` is the base-model release repository's (the agreement's notice). Both cards say “Built with Llama”; both repository names start with “Llama”.

## Scans

Every staged file, with the release / FOCUS patterns (local paths, the host, user names, private repository names, tokens, e-mail addresses, pointers to a reference implementation): `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant` 5 hits, 0 not accepted, `edgeai-lab/Llama-3.1-8B-Instruct-FOCUS-NVFP4` 1 hit, 0 not accepted. Accepted, as in the release-map upload: Meta's reporting address `LlamaUseReport@meta.com` inside the official `USE_POLICY.md` (byte-identical to the base model's), and the words “commit” / “branch” in the release card's own sentences (the FlipQuant commit is kept; the provenance paragraph).

## Addendum (2026-10-15): visibility

The user's decision (2026-10-15, relayed by the coordinator) was to make both repositories public. The user made the change. The implementation session did not: its permission system refused to.

Read-only checks without a token, at 2026-10-16 21:14 UTC; record in `../hf_gptq_focus/visibility.json`:

| repository | visibility | model info readable | file list = staged | README = staged | head = verified upload |
|---|---|---|---|---|---|
| `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant` | public | True | True | True | True |
| `edgeai-lab/Llama-3.1-8B-Instruct-FOCUS-NVFP4` | public | True | True | True | True |
