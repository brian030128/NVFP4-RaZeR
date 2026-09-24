# Native-decision calibration: a safe-transfer test

**Verdict: SAFE**, under the pre-registered PPL-only criterion ([PROTOCOL.md](PROTOCOL.md),
registered 10:59 UTC with sha256 ad688841…, before any measurement).

The native evaluation is the primary one. On it, the map whose backtracking decisions were all
made on the native kernel (DET-NATIVE) is never significantly worse than the fake-decided maps:

- **vs DET-FAKE:** WikiText-2 is inconclusive (+0.00075 ± 0.00146 per window); C4 is significantly
  better (−0.00210 ± 0.00108).
- **vs the other four fake-decided maps:** significantly better than B-256-opt on both corpora
  and than SHADOW on C4, and inconclusive in every other comparison.
- **Secondary, fake evaluation:** DET-NATIVE vs DET-FAKE is inconclusive on both corpora.
- **Speed:** native decisions cut optimization time by 1.32×, from 31.2 to 23.7 min, even
  though the native run needed 3 more rounds. A development evaluation takes 14.4 s per try
  instead of 34.7 s.

## Runs

Both runs are deterministic: `--deterministic`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`--unit 256x64 --objective kl --skip-ce-backward --eval-batch 16 --score-batch 8`, with the same
data. They differ only in `--dev-backend`.

| | DET-FAKE | DET-NATIVE |
|---|---:|---:|
| development KL (own backend), start → end | 0.106448 → 0.094804 | 0.108106 → 0.093990 |
| rounds (scoring passes) / development evaluations | 5 / 45 | 8 / 64 |
| final E0M3 tiles | 7,617 | 8,385 |
| setup / optimization | 94 s / **1,875 s** | 121 s / **1,419 s** |
| scoring per pass / development evaluation per try | 69.9 s / 34.7 s | 69.6 s / 14.4 s |
| peak GPU allocated / reserved | 60.9 / 63.4 GiB | 72.1 / 74.6 GiB (native weight packs add about 11 GiB) |
| peak host RSS | 43.7 GiB | 43.9 GiB |

**Where the paths diverge.**
- Rounds 0–2 are identical in both runs: candidate counts (15,136 / 29,087 / 18,970), every try
  size, and every accepted step (7,568, then 454, then 148). This confirms deterministic scoring.
- Round 3 has the same 18,957 candidates and the same tries down to 37 tiles. At the 37-tile try,
  **fake rejects (ΔKL +0.00053) and native accepts (−0.00028)**. Fake goes on to accept 9 tiles.
- From round 4 the states differ:
  - DET-FAKE stops at round 4.
  - DET-NATIVE takes a 1,061-tile step in round 4, then 119 and 58, and stops at round 7.

## Evaluation of the maps (WikiText-2 / C4, released windows, per-window paired ΔNLL ± 2 SE)

### Native kernel (primary)

| map | E0M3 tiles | WikiText | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs DET-FAKE | ΔC4 vs DET-FAKE |
|---|---:|---:|---:|---|---|---|---|
| FourOverSix | 0 | 6.875681 | 9.825390 | — | — | | |
| DET-FAKE | 7,617 | 6.831718 | 9.779919 | −0.00641 ± 0.00169 | −0.00464 ± 0.00157 | — | — |
| **DET-NATIVE** | 8,385 | 6.836861 | **9.759441** | −0.00566 ± 0.00171 | −0.00673 ± 0.00169 | +0.00075 ± 0.00146 | **−0.00210 ± 0.00108** |
| B-256-opt | 7,625 | 6.848610 | 9.789530 | −0.00394 ± 0.00163 | −0.00366 ± 0.00201 | +0.00247 ± 0.00150 (worse) | +0.00098 ± 0.00157 |
| B'-256-opt | 4,694 | 6.838040 | 9.765437 | −0.00549 ± 0.00178 | −0.00612 ± 0.00185 | +0.00093 ± 0.00161 | −0.00148 ± 0.00172 |
| SHADOW | 7,648 | 6.845790 | 9.783512 | −0.00436 ± 0.00171 | −0.00427 ± 0.00182 | +0.00206 ± 0.00141 (worse) | +0.00037 ± 0.00151 |
| B-256-ref | 7,724 | 6.844188 | 9.773750 | −0.00459 ± 0.00175 | −0.00527 ± 0.00158 | +0.00182 ± 0.00149 (worse) | −0.00063 ± 0.00143 |

Every map is significantly better than FourOverSix on both corpora.

**The SAFE criterion** (DET-NATIVE minus each map, native evaluation):

| DET-NATIVE minus | ΔWiki | ΔC4 |
|---|---|---|
| DET-FAKE | +0.00075 ± 0.00146 (inconclusive) | −0.00210 ± 0.00108 (**better**) |
| B-256-opt | −0.00172 ± 0.00146 (**better**) | −0.00308 ± 0.00155 (**better**) |
| B'-256-opt | −0.00017 ± 0.00152 (inconclusive) | −0.00061 ± 0.00160 (inconclusive) |
| SHADOW | −0.00131 ± 0.00153 (inconclusive) | −0.00246 ± 0.00164 (**better**) |
| B-256-ref | −0.00107 ± 0.00144 (inconclusive) | −0.00147 ± 0.00160 (inconclusive) |

No comparison is significantly worse, so both conditions hold: **SAFE**.

### Fake quantization (secondary)

| map | WikiText | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs DET-FAKE | ΔC4 vs DET-FAKE |
|---|---:|---:|---|---|---|---|
| FourOverSix | 6.887180 | 9.829412 | — | — | | |
| DET-FAKE | 6.833892 | 9.772741 | −0.00777 ± 0.00177 | −0.00578 ± 0.00187 | — | — |
| DET-NATIVE | 6.831542 | 9.773669 | −0.00811 ± 0.00172 | −0.00569 ± 0.00148 | −0.00034 ± 0.00139 | +0.00010 ± 0.00149 |
| B-256-opt | 6.839694 | 9.782815 | −0.00692 ± 0.00189 | −0.00475 ± 0.00204 | +0.00085 ± 0.00152 | +0.00103 ± 0.00183 |
| B'-256-opt | 6.831254 | 9.776279 | −0.00815 ± 0.00182 | −0.00542 ± 0.00183 | −0.00039 ± 0.00156 | +0.00036 ± 0.00148 |
| SHADOW | 6.836477 | 9.780264 | −0.00739 ± 0.00174 | −0.00501 ± 0.00171 | +0.00038 ± 0.00157 | +0.00077 ± 0.00130 |
| B-256-ref | 6.844625 | 9.775727 | −0.00620 ± 0.00187 | −0.00548 ± 0.00177 | +0.00157 ± 0.00133 (worse) | +0.00031 ± 0.00178 |
| BF16 (unquantized) | 6.240271 | 8.957924 | | | | |

## Verification (all before the measurements)

- **Bitwise checks**, signed zeros included, all passed:
  - both packed candidates of all 224 matrices;
  - the map weight at the start, a random mixed and the restored map;
  - every one of the seven evaluated maps (0 mismatches each);
  - the first 64 activation calls of every map.
- **The fake `--evaluate-map` path reproduces every recorded fake final evaluation bitwise, per
  window:** arm A, B-256-opt, B'-256-opt, B-256-ref and SHADOW. Each DET run's built-in fake
  evaluation, done in deterministic mode, is also bitwise identical to its default-mode map
  evaluation, so deterministic mode changes no forward numerics here.
- **Native vs fake final NLL at fixed maps** (verification step 3; the bug threshold was 0.01
  nats):
  - FourOverSix: −0.0017 ± 0.0018 (WikiText), −0.0004 ± 0.0015 (C4);
  - B-256-opt: +0.0013 ± 0.0014, +0.0007 ± 0.0023.
- On native, FourOverSix WikiText is 6.875681. The published H200 fake value is 6.875525; the
  local fake value is 6.887180.

## Interpretation and caveats

- **The native backend moves only decisions within noise.** It changes decisions on steps whose
  development effect is within evaluator noise (the shadow study localised these below a 0.0025
  KL margin). Here that led to a different path: 8,385 tiles against 7,617. The result was no
  worse on the native kernel, and better on C4 against DET-FAKE.
- **Selection variance is as large as the backend effect.** Among the fake-decided maps, the
  native evaluation already separates them. B-256-opt, SHADOW and B-256-ref are significantly
  worse than DET-FAKE on WikiText, by 0.0018–0.0025. The paired ±2 SE does not include this
  variance, and each backend has one selection.
- **The native evaluation is the relevant one for deployment.** Ranking maps by fake PPL is not
  the same as ranking them by native PPL. For example, DET-NATIVE vs DET-FAKE on C4 is −0.0021
  (significant) on native but +0.0001 on fake. B-256-opt vs DET-FAKE on WikiText is worse on
  native but inconclusive on fake.
- **The development documents are not held out.** The 192 documents decide every step.
- **Nothing was tuned on test data.** Nothing was selected or tuned on WikiText or C4. Zero-shot
  was deferred by user decision and is not part of this verdict
  (`results/multiround_local/PROTOCOL.md`, deviation 1).

## Reproduction

```
DATA=/home/dev/n16k64_campaign/cost_comparison/data
R=/home/dev/n16k64_campaign/native_decision
CR=/home/dev/n16k64_campaign/cost_comparison/runs
export HF_HUB_OFFLINE=0 PYTHONPATH=$PWD
COMMON="--unit 256x64 --objective kl --data-root $DATA --transformers-deviation"
DET="--skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --deterministic"
CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_multiround.py $COMMON $DET --dev-backend fake   --out $R/det_fake
CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_multiround.py $COMMON $DET --dev-backend native --out $R/det_native
MAPS="--evaluate-map FourOverSix=fourover6 --evaluate-map DET-FAKE=$R/det_fake/map.pt --evaluate-map DET-NATIVE=$R/det_native/map.pt \
      --evaluate-map B-256-opt=$CR/B_256x64_opt/map.pt --evaluate-map Bprime-256-opt=$CR/Bprime_256x64_opt/map.pt \
      --evaluate-map SHADOW=/home/dev/n16k64_campaign/native_dev_shadow/shadow_256x64/map.pt --evaluate-map B-256-ref=$CR/B_256x64_ref/map.pt"
python run_multiround.py $COMMON --eval-backend native $MAPS --out $R/eval_native
python run_multiround.py $COMMON --eval-backend fake $MAPS --evaluate-map BF16=bf16 --out $R/eval_fake
python results/native_decision/analyze_decision.py $R        # summary.json, tables.md
```

- Environment: torch 2.9.0+cu128, transformers 5.16.1 (a recorded deviation from 4.57.3), triton
  3.5.0, driver 595.71.05, native kernel `libb8x64.so` (sha256 0e237ada…).
- Map hashes: DET-FAKE 067c3971…, DET-NATIVE 6e9704f5….
- Run records: `runs/`. Hashes at registration: `registration.json`.
