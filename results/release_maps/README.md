# FlipQuant release maps: records

The records of the release maps calibrated on 2026-10-05/06 with flipquant's `calibration.train_map` (branch
`optimized-defaults`, github brian030128/flipquant) for eight models at three tile units. The maps themselves are not
here: they are in `/home/dev/flipquant_release/<model>/flipquant_<unit>.pt` on the RTX PRO 6000 host, outside git. Their
sha256 values are in `REPORT.md`. Since 2026-10-07 they are also on the Hugging Face Hub as eight private repositories
under `edgeai-lab` (`HF_UPLOAD.md`).

- **`REPORT.md`**: E0M3 tiles, time and memory per calibration, the one-time data preparation per model, PPL against
  FourOverSix and BF16 with 2 SE verdicts, the paper-setting maps' PPL for reference, the provenance and checks, and
  every map's sha256.
- **`summary.json`**: the same, machine-readable.
- **`<model>/MODEL_CARD.md`**: the draft model card of each model (the release directory's `README.md`). `NOTICE` is
  the Llama 3.1 attribution notice.
- **`<model>/records/<unit>/`**:
  - `run.json`: the release driver's record (command, flipquant commit, extension checks, time, memory);
  - `reproduction.json`: train_map's;
  - `trainer_report.json`: the trainer's own report (PhaseMonitor phases and epochs, the fit extension's windows).
- **`<model>/ppl/`**: the evaluation reports (per-window NLLs) and `significance.json` (evaluation.significance against
  FourOverSix).
- **`HF_UPLOAD.md`**: the Hugging Face upload: the eight private repositories, their commits and checks, the maps'
  meta changes (release → uploaded sha256), the license files' sources and the card changes; `hf/` holds `stage.json`
  and `hf_upload.json`. Scripts: `hf_stage.py`, `hf_licenses.py`, `hf_upload.py`, `hf_record.py`.
- **`dev_kl/`**: development-set KL along the calibration, Llama-3.1-8B, Phi-4 and Qwen3-1.7B at 16x64. It compares
  the release setting, run for 10 epochs, with the paper setting, using the trainer's own development evaluation, and
  measures the WikiText-2 / C4 perplexity of the epoch-10 maps (and Phi-4's epoch 7). `NOTE.md` has the tables, paired
  tests and findings; `dev_kl.json`, `paired.json` and `ppl.json` the data; `dev_kl.png` the plot. Scripts:
  `devkl_run.py`, `devkl.sh`, `devkl2.sh`, `devkl3.sh`, `devkl_report.py`.
- **`../../experiments/release_maps/`**: the scripts that produced all of this.
  - `release.py`: the driver.
  - `chain2.sh` and `chain3.sh`: its queues.
  - `watch_tree.py`: live process-tree measurement.
  - `remeasure.py`: the measured re-runs; `measure.py` is topk-cal's, byte for byte.
  - `measure_vmhwm.py`, `prep_run.py` and `compare_prep.py`: the data-preparation measurement.
  - `report.py`: REPORT.md and the cards.
  - `publish_records.sh`: copies them here.

**Settings:** `--fit-windows 256 --epochs 5 --teacher-topk 1000`. Everything else is the paper's: learning rate 0.02,
initial logit −1, seed 0, the per-model batch (2 × 4 for Nemotron-Nano-9B-v2 and Qwen3.8-27B). The maps' `meta` records
the non-default settings, the base model's revision and the flipquant commit.
