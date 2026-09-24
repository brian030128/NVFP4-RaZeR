# Native shadow verification of the development evaluation

**Verdict: FAIL.** Running the development evaluation on the native mixfp4 kernel instead of
fake quantization changes **11 of 72** backtracking decisions of one 256×64 KL-only run. **2 of
the 7 accepted steps** would have been rejected. By the pre-registered criterion this is FAIL,
so we cannot claim that native evaluation has "no effect on the selection path".

Every disagreement is on a step whose fake development-KL change is smaller than 0.0025 in
magnitude, and all but one are within 2 SE (per document) of zero. Every try with a larger margin
agrees (45 of 45), including the large steps of rounds 0–1, which carry most of the gain.

The protocol, [PROTOCOL.md](PROTOCOL.md), was written and hashed before any run
(`registration.json`, sha256 e4c0a7a9…).

## What ran

- **Machine and settings:** 1× RTX PRO 6000 Blackwell, Llama-3.1-8B, the same data and settings
  as the cost study's B-256-opt: `--unit 256x64 --objective kl --eval-batch 16 --score-batch 8`.
- **Decisions:** the fake development evaluation makes every decision.
- **Shadow:** at every evaluation the native evaluation runs on the same state and is logged.
  - Native kernel: `libb8x64.so` (sha256 0e237ada…), mixfp4 @7b3ab34, SASS-patched.
  - Weights: operand B, format granule 8×64.
  - Activations: per-document tensor-wide FourOverSix, one epilogue scale per document.
  - Eval batch 16 (8,192 tokens per forward).
- **Native "current":** it follows fake's accepted states.
- **Code:** `run_multiround.py --shadow-native`, with `repro_local/realquant/native_dev.py` and the
  fused activation quantizer's opt-in per-document mode (`fused_quant.quantize(gs=...,
  signed_zero=True)`).

**Implementation checks.** All passed at the start of every run. Each is bitwise, signed zeros
included:
- For all 224 matrices, the packed E2M1 and E0M3 candidates decode to what `decode_base` /
  `decode_alt` return, i.e. what `apply()` installs.
- The packed map weight decodes to the installed weight. This holds at the start map, at a random
  mixed map (212,624 of 425,984 tiles E0M3) and after restoring the start map (0 mismatches each),
  and at the B-256-opt map (pre-check b).
- The first 64 native activation calls equal `quant_per_document`.
- The unit gate (`repro_local/realquant/test_native_dev.py`) gives native vs fake F.linear relative
  error 3.4–3.7e-3. That is BF16 output rounding: native rounds D to bf16, then applies the document
  scale.
- The fused quantizer's default path is unchanged: `test_fused_quant.py` passes bitwise again on
  448 real activation inputs (5.1e9 elements, 0 differences in codes, scale bytes and global
  scales).
- The first pre-check attempt aborted on one of these checks, before any measurement: the E0M3
  zero sign did not match `decode_alt`'s offset-binary storage. It was fixed; see PROTOCOL.md,
  deviation 1.

## Pre-check: native vs fake at fixed maps (192 documents)

| map | fake KL | native KL | native − fake | per-document abs ΔKL, mean / max | fake CE | native CE | native − fake | per-document abs ΔCE, mean / max |
|---|---:|---:|---:|---|---:|---:|---:|---|
| FourOverSix start | 0.106448 | 0.108106 | +0.001658 (+1.6%) | 0.00897 / 0.08504 | 1.480964 | 1.483542 | +0.00258 | 0.0163 / 0.0845 |
| B-256-opt final (7,625 tiles) | 0.095233 | 0.096248 | +0.001015 (+1.1%) | 0.00602 / 0.05298 | 1.471465 | 1.471706 | +0.00024 | 0.0126 / 0.0736 |

- Both are far below the pre-registered bug thresholds (10% relative in mean KL; 0.01 nats in mean
  CE), so no implementation bug.
- Loading the B-256-opt map reproduces that run's final fake development KL bitwise, per document.
- The map's gain survives on native: −0.01186 native vs −0.01121 fake.
- Per document the two evaluators differ by up to 0.085 in KL. This is the same W4A4 sensitivity
  to BF16 numerics that the cost study measured for batch size.

## Shadow run

The run stopped after 8 rounds ("no step lowers the development objective"), with 72 tries,
7 accepted steps and 7,648 E0M3 tiles.

| | fake decides | native (logged) |
|---|---|---|
| decisions that agree | — | 61 / 72 |
| accepted steps native would also accept | — | 5 / 7 |
| rejected steps native would accept | — | 9 / 65 |
| development-evaluation seconds per try (mean) | 30.8 | **11.5** (+0.03 repacking) |

### Disagreements

Δ values are mean development KL changes. The ± after Δfake is 2 SE over the 192 documents.

| round | step (tiles) | Δfake | Δnative | margin abs(Δfake) | discrepancy abs(Δnative − Δfake) | fake | native |
|---:|---:|---|---:|---:|---:|---|---|
| 1 | 838 | +0.002380 ± 0.002502 | −0.000960 | 0.002380 | 0.003340 | reject | accept |
| 3 | 138 | −0.001043 ± 0.001539 | +0.000215 | 0.001043 | 0.001258 | **accept** | reject |
| 4 | 138 | +0.001625 ± 0.001944 | −0.002140 | 0.001625 | 0.003764 | reject | accept |
| 5 | 66 | −0.000282 ± 0.001844 | +0.001301 | 0.000282 | 0.001583 | **accept** | reject |
| 6 | 129 | +0.002126 ± 0.002052 | −0.000878 | 0.002126 | 0.003004 | reject | accept |
| 6 | 64 | +0.000303 ± 0.001822 | −0.001375 | 0.000303 | 0.001678 | reject | accept |
| 6 | 32 | +0.000218 ± 0.002077 | −0.001356 | 0.000218 | 0.001574 | reject | accept |
| 6 | 16 | +0.001028 ± 0.002383 | −0.000142 | 0.001028 | 0.001170 | reject | accept |
| 6 | 8 | +0.001035 ± 0.002379 | −0.000085 | 0.001035 | 0.001120 | reject | accept |
| 6 | 4 | +0.000497 ± 0.002071 | −0.001510 | 0.000497 | 0.002006 | reject | accept |
| 7 | 33 | +0.000062 ± 0.001846 | −0.000637 | 0.000062 | 0.000699 | reject | accept |

### Discrepancy vs margin

| margin abs(Δfake) | tries | disagreements |
|---|---:|---:|
| < 0.0005 | 11 | 5 |
| 0.0005 – 0.001 | 4 | 0 |
| 0.001 – 0.0025 | 12 | 6 |
| 0.0025 – 0.005 | 5 | 0 |
| ≥ 0.005 | 40 | 0 |

- **Size of the disagreement:** the largest discrepancy in the run is 0.00376. Fifteen tries have
  discrepancy > margin, and 11 of those flip.
- **Link to significance:** 49 tries have a fake step significant over documents (abs(Δfake) >
  2 SE). Only one of them disagrees: round 6, 129 tiles, +0.00213 ± 0.00205.
- **Noise level:** per step, the native-minus-fake difference of the per-document deltas has
  2 SE of about 0.0023–0.0034. For small steps the fake and native per-document deltas are only
  weakly correlated (−0.3 to +0.6).
- **Smallest margins:**
  - smallest accepted margin: 3.2e-5 (round 6, 2 tiles; native agreed);
  - smallest margin of any try: the same;
  - both disagreements on accepted steps have margins of 0.0010 and 0.0003.
- **Accepted steps:**

  | round | step (tiles) | margin | native |
  |---:|---:|---:|---|
  | 0 | 7,571 | 0.00127 | agrees |
  | 1 | 419 | 0.00684 | agrees |
  | 2 | 149 | 0.00193 | agrees |
  | 3 | 138 | 0.00104 | **rejects** |
  | 4 | 69 | 0.00036 | agrees |
  | 5 | 66 | 0.00028 | **rejects** |
  | 6 | 2 | 0.00003 | agrees |

  Rounds 0–1 take development KL from 0.10645 to 0.09833 of the final 0.09469. From round 2 on,
  the accepted steps sit at or within 2 SE of zero on the development set: round 2 is
  −0.00193 ± 0.00188, and rounds 3–6 are within.

### Context (not part of the criterion)

- **Final map (fake):** WikiText 6.836477 and C4 9.780264. Per window that is −0.0074 ± 0.0017 and
  −0.0050 ± 0.0017 vs the cost study's FourOverSix arm A. Against B-256-opt's map the difference is
  −0.0005 ± 0.0014 and −0.0003 ± 0.0018, i.e. indistinguishable.
- **Cost:**
  - The run took 8 scoring passes and 73 development evaluations.
  - Setup was 133 s and selection 3,961 s. That includes 73 native evaluations totalling 842 s,
    which exist only because of the shadow.
  - Peak GPU was 74.0 / 80.4 GiB, against 62.8 / 68.9 GiB for B-256-opt: the native candidate
    packs add about 11 GiB. Peak host RAM was 43.9 GiB.

## Conclusion

- **Native evaluation does change the path of the election.** It flips 11 of 72 decisions,
  including 2 accepted steps.
- **The flips are confined to steps whose effect is at the level of evaluator numerics.** They
  occur only at margins below 0.0025, almost always within 2 SE of zero over the 192 documents.
  The large, clearly significant steps (all of rounds 0–1 and every margin ≥ 0.0025) agree
  without exception.
- **What this implies:**
  - Late-round acceptances depend on BF16 and FP4 numerical details as much as on the data.
  - This matches the cost study's findings: non-deterministic scoring, batch sensitivity, and the
    B vs B' divergence to maps of equal quality.
- **What it doesn't show:** the final quality of a native-decided run is not measured here. The
  shadow cannot replay a trajectory that native would have taken, and fake and native final maps
  of this kind were statistically indistinguishable in the cost study.
- **Speed:** native development evaluation is 2.7× faster (11.5 s vs 30.8 s per try).

## Reproduction

```
DATA=/home/dev/n16k64_campaign/cost_comparison/data
R=/home/dev/n16k64_campaign/native_dev_shadow
export HF_HUB_OFFLINE=0 PYTHONPATH=$PWD
python repro_local/realquant/test_native_dev.py $DATA        # unit gate
python run_multiround.py --unit 256x64 --objective kl --eval-batch 16 --score-batch 8 --budget-hours 12 --check-start \
    --shadow-native --data-root $DATA --transformers-deviation --out $R/precheck_start_map
python run_multiround.py ... --check-start --shadow-native \
    --init-map /home/dev/n16k64_campaign/cost_comparison/runs/B_256x64_opt/map.pt ... --out $R/precheck_b256opt_map
python run_multiround.py --unit 256x64 --objective kl --eval-batch 16 --score-batch 8 --budget-hours 12 \
    --shadow-native --data-root $DATA --transformers-deviation --out $R/shadow_256x64
python results/native_dev_shadow/analyze_shadow.py $R        # summary.json, tables.md
```

Environment: torch 2.9.0+cu128, transformers 5.16.1 (the calibration's 4.57.3 is recorded as a
deviation), triton 3.5.0, driver 595.71.05. Script hashes at registration are in
`registration.json`. Every run record is in `runs/`, and the per-try table with all 72 tries is in
`tables.md`.
