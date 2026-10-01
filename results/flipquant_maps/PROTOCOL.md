# FlipQuant maps for the collaborators' repository (`~/flipquant`)

Written 2026-10-01 at 15:20 UTC, before any GPU run of this task. Branch `flipquant-maps`, cut from `tm-opt` 74058af.
The calibration code there is identical to `kernel-opt`'s.

**The request**, relayed by the coordinator as the user's new top priority:
- Produce FlipQuant maps at 8x64, 16x64 and 256x64 for the models of the paper's new main table.
- Place them in `~/flipquant` (github brian030128/flipquant), where the collaborators can use them directly.
- No PPL, no lm-eval, no kernel work. The 8x64 kernel plan stays on hold.

## Method

FlipQuant = TM-OPT+TC with exactly the paper's settings (`docs/FLIPQUANT_CALIBRATION_zh.md`).
- **Command:** `run_train_map.py --model <key> --data-root <root> --unit <unit>` with the paper's flags
  (`experiments/paper/paper_common.TM_OPT_TC`): `--tm-opt --tile-grad-tc --param ste --lr 0.02 --init-logit -1
  --epochs 20 --eval-every 2 --no-dev --no-eval`.
- **Batch:** 8, with no accumulation, for every new model.
- **Fixed:** seed 0; Adam with betas (0.9, 0.999) and eps 1e-12; a constant schedule; the last-epoch map.
- **Environment:** the paper runner's (`paper_common.env` + `ACCURACY_ENV`): `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, deterministic. Per-token FourOverSix activations.
- **Trainer:** our pipeline for every new model. If a model cannot be supported in reasonable time, the coordinator is
  told before switching to `~/flipquant`'s `calibration/train_map.py`.

## Models

| `~/flipquant` registry key | our key | model @ revision | maps |
|---|---|---|---|
| qwen3-1.7b | qwen3_1p7b | Qwen/Qwen3-1.7B @ 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | trained |
| qwen3-8b | qwen3_8b | Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218 | trained |
| mistral-7b | mistral7b_ins | mistralai/Mistral-7B-Instruct-v0.3 @ c170c708c41dac9275d15a8fff4eca08d52bab71 | trained |
| nemotron-nano-9b-v2 | nemotron9b | nvidia/NVIDIA-Nemotron-Nano-9B-v2 @ 6533e8de2c68e4536bf7c411d7a3ce5734111476 | trained |
| phi4-14b | phi4 | microsoft/phi-4 @ 2db69c1c3e91a05d2c64a3185acfbaf36f744e25 | the paper's, if the rule below holds |
| qwen3.8-27b | qwen27b | Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 | the paper's, if the rule below holds |

- **Revisions:**
  - The new models are pinned to the Hub's `main` as resolved on 2026-10-01 at about 15:05 UTC.
  - `~/flipquant`'s registry pins Nemotron and Qwen3.8-27B; it resolves `main` for the others.
  - `microsoft/phi-4` `main` resolved to 2db69c1c, the paper's revision.
- **Our existing Mistral maps are for the base model and are not used.**

## Fit sets

Each new model's 128 math/code windows are drawn the way our Phi-4 fit set was, with `prepare_model_data.py` and no
prior record:
- the rule is `campaign.data.builder_seed0`, the rule behind Llama's fit set: file order, skip documents shorter than
  512 tokens, offset `random.Random(20260926).randrange`, at the pinned dataset revisions;
- before it is used, it must reproduce the Llama, Qwen3-4B and Mistral records exactly: document, offset and token
  hash.
- `math_code_data` then reproduces all 128 windows and checks every token hash.
- The matrices (shape and weight sha256) are computed from the pinned snapshot.
- Development draws are not made (`--fit-only`, new), because calibration runs with `--no-dev`.

**Code added:** the four models in `prepare_model_data.py` and `run_multiround.py`, so that `data_paths` resolves
them; `--fit-only`; and a model loader for Nemotron if transformers' native NemotronH needs one.

## Gates and checks

- **Gate R (reproduction), approved by the coordinator.** Re-train Mistral-7B-v0.3 8x64 with the paper's command into
  a scratch directory. The map must be bitwise equal to the committed paper map (sha256 8aacdd77…, `maps.sha256.json`).
  - It runs while the GPU is otherwise idle during preparation.
  - If it fails, stop and report before any new model is trained.
- **The fit-set builder rule check:** as above.
- **Nemotron smoke** (disclosed, not used): a short run to check that the trainer handles NemotronH (Mamba-2 on
  transformers' torch path, since mamba_ssm is not installed), and to measure time and memory.
- **Per trained map:** the run's `report.json` has status complete, 20 epochs, and the registered settings.

## Reuse rule (phi4-14b, qwen3.8-27b)

A paper map is reused only if two conditions hold:
- the registry's revision equals the paper's;
- the map's module names and tile shapes equal `~/flipquant`'s `quantizable_linears` on a meta-device model.

Otherwise the map is regenerated with the method above. Reused maps are copied from `/home/dev/n16k64_campaign/paper/maps`
and never moved.

## Delivery

- **File:** `~/flipquant/maps/<registry key>/flipquant_<unit>.pt`, in `flipquant-map/1` (`flipquant.maps.save`'s
  layout). Existing files are never overwritten.
- **meta:** trainer, repository and branch and commit, all settings, model id and revision, unit, E0M3 tile share,
  calibration time, and, for reused maps, the source path and sha256.
- **Verification:** each file is loaded with `~/flipquant`'s `maps.load`. Its module names and tile shapes are checked
  against `quantizable_linears` on a meta-device model built by their loader (CPU only, no evaluation).
- **README:** rows are added to `~/flipquant/maps/README.md` locally. Nothing is committed or pushed to their repository.
- **Record:** a delivery manifest (file, sha256, source) and the run records go to `results/flipquant_maps/` on this
  branch.
