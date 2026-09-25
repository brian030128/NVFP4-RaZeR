# Qwen3-4B: the N16K64 one-shot k=3 maps under the Part C evaluation protocol

Protocol: [PROTOCOL.md](PROTOCOL.md), registered 2026-09-25 09:49:28 UTC (sha256 6920c6cf…), before
any check or evaluation. There is one deviation (check 3 moved to the GPU; see PROTOCOL.md).
No map was recalibrated. The only evaluation convention used is (a): one tensor-wide FourOverSix
activation scale per 2048-token window, as in Part C.

**Result.** Under Part C's own evaluation, the campaign's one-shot CE+KL k=3 maps beat every
multi-round KL-only map by a wide margin, on both corpora and both backends.
- **N16K64-n8-k3** (8,149 8x64 tiles):
  - native PPL **11.719 / 15.717**, against FourOverSix's 14.214 / 17.306 (−0.1930 / −0.0963 nats
    per window);
  - better than DET-FAKE-8x64 (187,147 tiles; 12.628 / 16.718) by **−0.0746 ± 0.0029 / −0.0617 ±
    0.0027**;
  - better than DET-NATIVE-8x64 (10,786 tiles; 13.045 / 17.058) by **−0.1072 ± 0.0038 /
    −0.0819 ± 0.0033**.
- **N16K64-n16-k3** (4,399 16x64 tiles = 8,798 8x64 tiles): native 12.128 / 15.963. It is also
  better than every multi-round map, and worse than n8_k3.

All differences below are significant; the fake backend agrees on every sign and significance.

## Evaluation (WikiText-2 146 windows, C4 256 windows; paired per-window ΔNLL ± 2 SE)

| backend | map | E0M3 tiles | WikiText-2 PPL | C4 PPL | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |
|---|---|---:|---:|---:|---|---|
| **native** | FourOverSix | 0 | 14.2140 | 17.3055 | — | — |
| **native** | **N16K64-n8-k3** | 8,149 | **11.7192** | **15.7169** | −0.19299 ± 0.00709 (better) | −0.09629 ± 0.00400 (better) |
| **native** | N16K64-n16-k3 | 4,399 (16x64) | 12.1275 | 15.9630 | −0.15875 ± 0.00650 (better) | −0.08075 ± 0.00343 (better) |
| **native** | DET-NATIVE-8x64 | 10,786 | 13.0453 | 17.0581 | −0.08579 ± 0.00557 (better) | −0.01440 ± 0.00258 (better) |
| **native** | DET-NATIVE-256x64 | 6,673 (256x64) | 13.9906 | 17.4348 | −0.01584 ± 0.00399 (better) | +0.00744 ± 0.00211 (worse) |
| **native** | DET-FAKE-8x64 | 187,147 | 12.6275 | 16.7179 | −0.11835 ± 0.00616 (better) | −0.03454 ± 0.00311 (better) |
| **native** | DET-FAKE-256x64 | 6,720 (256x64) | 14.0251 | 17.4551 | −0.01337 ± 0.00442 (better) | +0.00861 ± 0.00201 (worse) |
| fake | FourOverSix | 0 | 14.2442 | 17.3219 | — | — |
| fake | N16K64-n8-k3 | 8,149 | 11.6899 | 15.7152 | −0.19763 ± 0.00823 (better) | −0.09734 ± 0.00399 (better) |
| fake | N16K64-n16-k3 | 4,399 (16x64) | 12.1184 | 15.9781 | −0.16162 ± 0.00744 (better) | −0.08075 ± 0.00331 (better) |
| fake | DET-NATIVE-8x64 | 10,786 | 13.0396 | 17.0706 | −0.08836 ± 0.00647 (better) | −0.01461 ± 0.00255 (better) |
| fake | DET-NATIVE-256x64 | 6,673 (256x64) | 14.0873 | 17.4644 | −0.01108 ± 0.00498 (better) | +0.00820 ± 0.00202 (worse) |
| fake | DET-FAKE-8x64 | 187,147 | 12.6095 | 16.7192 | −0.12190 ± 0.00713 (better) | −0.03541 ± 0.00279 (better) |
| fake | DET-FAKE-256x64 | 6,720 (256x64) | 13.9997 | 17.4526 | −0.01731 ± 0.00500 (better) | +0.00752 ± 0.00188 (worse) |

BF16 (fake, Part C): 13.6588 / 16.6409.

| backend | one-shot map minus | ΔWiki | ΔC4 |
|---|---|---|---|
| native | n8-k3 − DET-NATIVE-8x64 | −0.10720 ± 0.00383 | −0.08188 ± 0.00329 |
| native | n8-k3 − DET-NATIVE-256x64 | −0.17715 ± 0.00616 | −0.10373 ± 0.00413 |
| native | n8-k3 − DET-FAKE-8x64 | −0.07464 ± 0.00293 | −0.06174 ± 0.00273 |
| native | n8-k3 − DET-FAKE-256x64 | −0.17962 ± 0.00633 | −0.10490 ± 0.00405 |
| native | n16-k3 − DET-NATIVE-8x64 | −0.07296 ± 0.00326 | −0.06635 ± 0.00287 |
| native | n16-k3 − DET-NATIVE-256x64 | −0.14291 ± 0.00527 | −0.08820 ± 0.00360 |
| native | n16-k3 − DET-FAKE-8x64 | −0.04040 ± 0.00262 | −0.04621 ± 0.00231 |
| native | n16-k3 − DET-FAKE-256x64 | −0.14538 ± 0.00557 | −0.08936 ± 0.00350 |

The fake-evaluation versions of these 16 pairings agree in sign and significance (`tables.md`).

**For reference only, a different protocol:** the campaign's own records under per-token
activation scales (convention (c)).

| map | fake WikiText-2 / C4 | native WikiText-2 / C4 |
|---|---|---|
| n8_k3 | 11.7066 / 15.7092 | 11.6925 / 15.6958 |
| n16_k3 | 12.1095 / 15.9714 | 12.0888 / 15.9623 (12.0933 / 15.9515 after the weights-on-A fix) |
| FourOverSix | 14.2183 / 17.3175 | 14.2106 / 17.3062 |

For these maps the activation convention moves PPL by less than 0.04: at most 0.039, for n16_k3 on native WikiText-2.

## Checks

1. **The maps are what their provenance says** (`checks.json`):
   - The file sha256 (n8 1ce7a7d7…, n16 d8e3e0fa…) and the mask payload hash match each
     `.provenance.json`.
   - The header names Qwen/Qwen3-4B at `1cfa9a72…` for model and tokenizer, type block [8, 64] /
     [16, 64], and the 252 module names and shapes of the calibration record, in order.
   - The election rule is `max(mean_CE+3SE, mean_KL+3SE)<0`.
2. **The conversion is exact.** For every module, the element mask expands identically before and
   after conversion. The tile counts are 8,149 → 8,149 (n8) and 4,399 → 8,798 8x64 tiles (n16),
   per module and in total.
3. **The weight candidates are the same functions, and the installed weights are the same values.**
   Checked on the GPU, on all 252 modules:
   - The campaign's `quant.four_over_six` / `quant.e0m3` equal our `quant_nvfp4_4over6` /
     `quant_mix_4_6(..., clip='a1', elect='always')` bitwise.
   - For both maps, the weight the campaign's installer rule (`tiles.apply_mask`) puts in place
     equals ours by value in every module.
   - Bitwise, 8 / 20 modules are identical. The remaining differences (264,471 / 289,760 elements)
     are all zeros of opposite sign inside E0M3 tiles: `decode_alt` has no −0, `quant_mix_4_6`
     does.
   - On CPU, check 3 could not run (deviation 1). There our encoder and `quant_mix_4_6` disagree
     because of CPU division semantics; neither evaluator runs on CPU.
4. **Validity of the evaluation.**
   - FourOverSix's per-window NLL equals Part C's evaluation bitwise on both backends.
   - Every native map had 0 packed-weight mismatches, and the first 64 activation calls of every map
     were bitwise.
   - The 256x64 maps were expanded exactly to 8x64 tiles in the process. The table lists their own
     tile counts; that is 213,536 / 215,040 in 8x64 units.

## Observations (not criteria)

- **Objective and search, not activation convention.** The one-shot CE+KL maps keep their
  per-token-convention advantage under convention (a).
  - Relative to the KL-only multi-round maps, the gap is 0.04–0.18 nats per window on Qwen3-4B.
  - n8_k3 elects fewer tiles (8,149) than DET-NATIVE-8x64 (10,786), yet gains 2.3× more on
    WikiText-2.
- **PPL below BF16 is not fidelity to BF16.** Every 8x64 map here is below BF16 on WikiText-2
  (13.66). The one-shot maps are also far below BF16 on C4 (15.72 vs 16.64).
  - For this post-trained model, lower PPL on raw text does not mean closer to BF16. The one-shot
    rule's CE term rewards lower NLL directly, while the KL-only multi-round objective rewards
    fidelity to BF16.
  - The development-set KL of the one-shot maps was not measured here, so their fidelity is not
    compared.
- **The multi-round search's path dependence** at 8x64 (`../det_fake/REPORT.md`) is part of this
  gap. DET-FAKE's single 176,728-flip step brought it 0.02–0.03 nats closer, but not to n8_k3.

## Reproduction

```
python results/multiround_models/qwen4b/oneshot_k3_eval/convert_check.py /home/dev/n16k64_campaign/multimodel/data \
    /home/dev/n16k64_campaign/multimodel/runs/qwen4b_oneshot/maps          # GPU; checks.json
# evaluation: runs/chain_oneshot_phi4.sh (its final Phi-4 step was replaced by a no-op when Part C was paused)
python results/multiround_models/qwen4b/oneshot_k3_eval/analyze_oneshot.py  # summary.json, tables.md
```

- **Converted maps:** n8 1e495ac7…, n16 ed8d82ee… (in
  `/home/dev/n16k64_campaign/multimodel/runs/qwen4b_oneshot/maps`).
- **Run records:** in `runs/`.
