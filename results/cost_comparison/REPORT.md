# Calibration cost vs. QAT (BF16-distillation) cost: KL-only multi-round MixFP4 election

**NOT FINAL.** The study was paused by user decision on 2026-09-24 before B-8x64-opt finished; see [STATUS.md](STATUS.md) for the completed arms and their numbers. This draft is kept only as a skeleton.

Llama-3.1-8B, W4A4 with FourOverSix activations. Every arm ran on the same single
NVIDIA RTX PRO 6000 Blackwell (96 GB). The protocol, [PROTOCOL.md](PROTOCOL.md), was
written and hashed before any arm ran (sha256 c70c59c0…, `registration.json`). Its
append-only section 10 records the deviations. Raw run records are in `runs/`.

## Answer

(to be written when all runs are complete)

## Cost vs. accuracy

(table from `tables.md`, completed at the end)

## Setup (all arms, same machine)

- 1× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB; 94.97 GiB usable by
  PyTorch), driver 595.71.05, 241 GB RAM, 128 CPUs.
- conda env `n16k64`: Python 3.11.11, torch 2.9.0+cu128 (CUDA 12.8), transformers 5.16.1,
  datasets 4.8.5, triton 3.5.0, psutil 7.2.2. torchao 0.14.1 (arm C optimizers only) is
  installed outside the env.
- The calibration used transformers 4.57.3; every report records both versions
  (`--transformers-deviation`).
- Llama-3.1-8B @ d04e592b, BF16, SDPA. All 224 text Linear weights are checked against
  their recorded sha256 in every run.

**Data (Step 1).** The cluster inputs of `run_multiround.py` were rebuilt locally by
`prepare_multiround_data.py`:
- The three development `fresh.pt` files are regenerated from the published manifests. They
  are **bitwise identical** to the cluster files: sha256 2d2c4940…, 5c5a1bb2…,
  eb913e15…, checked again by `run_multiround.py` on every load.
- All 128 fit token hashes, all 141 + 256 evaluation-window hashes and all 224 weight
  hashes match.
- The calibration report (sha256 47e0d54c…) is not available. The fields the script
  reads come from `results/math_code_adaptive/calibration_333779_llama8b`. That this is
  the same fit set is shown by 64 fit/election records of a job citing 47e0d54c….
