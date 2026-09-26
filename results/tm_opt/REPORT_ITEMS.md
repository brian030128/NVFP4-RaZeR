# TM-OPT: seed variance, three models, and the GEMM cost of E0M3-dense maps — report

**Status (2026-09-26 09:50 UTC).**
- **#2 (seed variance):** done (this section).
- **#1 (three models):** running.
- **#3 (GEMM and prefill cost):** waiting for #1. Its kernels are built.

**Protocol.** [PROTOCOL_ITEMS.md](PROTOCOL_ITEMS.md), registered 2026-09-26 08:57:46 UTC (sha256
97cacaf8…), before any run.
- **The study is descriptive.** Nothing is selected on WikiText-2, C4 or zero-shot.
- **#3's scripts** were recorded before any #3 step (`registration_items_bench.json`).

## #2 Seed variance (Llama-3.1-8B, TM-OPT STE 8x64)

**Setup.** Seeds 1 and 2 are identical to the verified group-2 run (seed 0) except the calibration
batch order. The evaluation is native (primary) and fake, with FourOverSix, the three seeds and
MR-OPT 8x64 in one native process.
- **Checks passed:** the FourOverSix and MR-OPT 8x64 window NLLs repeated the committed evaluations
  bitwise (native and fake), and seed 0 repeated its own group-2 evaluation.

| seed | E0M3 tiles | final dev KL | native WikiText-2 | native C4 | fake WikiText-2 | fake C4 | native ΔWiki vs MR-OPT 8x64 | native ΔC4 vs MR-OPT 8x64 | Jaccard with MR-OPT |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 0 | 306,968 | 0.08288 | 6.7822 | 9.6754 | 6.7792 | 9.6734 | −0.00458 ± 0.00158 (better) | −0.00916 ± 0.00323 (better) | 0.004 |
| 1 | 307,600 | 0.08411 | 6.7791 | 9.6719 | 6.7806 | 9.6759 | −0.00505 ± 0.00173 (better) | −0.00952 ± 0.00351 (better) | 0.004 |
| 2 | 308,688 | 0.08388 | 6.7748 | 9.6755 | 6.7760 | 9.6771 | −0.00567 ± 0.00154 (better) | −0.00915 ± 0.00338 (better) | 0.004 |

For reference, MR-OPT 8x64 scores 6.8134 / 9.7644 and FourOverSix 6.8757 / 9.8254 (native).

- **Every seed is significantly better than MR-OPT 8x64 on both corpora**, native and fake. In the
  fake evaluation the ΔNLL is −0.0055 to −0.0062 on WikiText-2 and −0.0086 to −0.0090 on C4.
- **Every seed is significantly better than FourOverSix:** native −0.0137 to −0.0148 on WikiText-2
  and −0.0154 to −0.0158 on C4.
- **The two methods pick mostly different tiles.** Each TM-OPT map contains only about 1,170 of
  MR-OPT's 3,801 tiles (31 %). The Jaccard index is tiny because TM-OPT elects 81× more tiles.
- **Time and memory per seed:** selection 13.9 min; 35.0–35.2 s per epoch; setup 1.3–2.0 min; peak
  GPU 40.8 / 42.0 GiB; host 42.2 GiB.

**Seed vs seed** (native paired ΔNLL, first minus second):

| seeds | ΔWiki | ΔC4 | Jaccard | tiles shared / only first / only second |
|---|---|---|---:|---|
| 0 − 1 | +0.00046 ± 0.00157 (n.s.) | +0.00036 ± 0.00108 (n.s.) | 0.446 | 189,414 / 117,554 / 118,186 |
| 0 − 2 | +0.00109 ± 0.00138 (n.s.) | −0.00001 ± 0.00107 (n.s.) | 0.438 | 187,586 / 119,382 / 121,102 |
| 1 − 2 | +0.00062 ± 0.00155 (n.s.) | −0.00038 ± 0.00097 (n.s.) | 0.441 | 188,705 / 118,895 / 119,983 |

- **The seeds' maps overlap only by ~44 %** (Jaccard), and about 118k–121k tiles differ per pair.
  Yet no pair differs significantly on either corpus.
- **Many tile sets of about 307k tiles do equally well.** The tile identity is not stable across
  seeds; the quality is.

**Is the seed spread small relative to the TM-OPT − MR-OPT gap?** Yes, on both corpora, by the
registered rule (spread < |gap| / 2):

| corpus | gap: mean over seeds of ΔNLL vs MR-OPT | seed spread (range of the three means) | small | every seed significantly better |
|---|---:|---:|---|---|
| WikiText-2 | −0.00510 | 0.00109 | **yes** | yes |
| C4 | −0.00927 | 0.00038 | **yes** | yes |

**Is the local-vs-H200 tile-count difference within the seed spread? Only through epoch 6; from
epoch 8 on it is not.**

| epoch | H200 E0M3 tiles | local seeds 0 / 1 / 2 | H200 inside the local range | relative seed spread | seed 0 vs H200 |
|---:|---:|---|---|---:|---:|
| 4 | 2,072 | 1,894 / 1,805 / 2,123 | yes | 16.4 % | −8.6 % |
| 6 | 24,585 | 22,876 / 21,767 / 26,053 | yes | 18.2 % | −7.0 % |
| 8 | 83,110 | 72,715 / 70,347 / 78,151 | no | 10.6 % | −12.5 % |
| 12 | 183,041 | 167,760 / 164,601 / 169,079 | no | 2.7 % | −8.3 % |
| 16 | 260,629 | 241,822 / 238,717 / 242,855 | no | 1.7 % | −7.2 % |
| 20 | 329,837 | 306,968 / 307,600 / 308,688 | no | 0.6 % | −6.9 % |

- **The difference is systematic, not seed noise.** The three local seeds converge to within 0.6 %
  of each other by epoch 20, while all stay about 7 % below the H200 run.
- **Probable causes, not verified here:**
  - the H200 run used the calibration record's transformers version, while ours runs 5.16.1
    (a recorded deviation);
  - the GPU's floating-point reductions differ.
- **The PPL gap is tiny:** local fake 6.776–6.781 / 9.673–9.677 against H200 6.7847 / 9.6754.

**Development KL (native monitor) per seed:**

| epoch | seed 0 | seed 1 | seed 2 |
|---:|---:|---:|---:|
| 0 | 0.10811 | 0.10811 | 0.10811 |
| 4 | 0.10024 | 0.09828 | 0.09918 |
| 8 | 0.08850 | 0.08897 | 0.08825 |
| 12 | 0.08661 | 0.08573 | 0.08562 |
| 16 | 0.08434 | 0.08537 | 0.08432 |
| 20 | 0.08288 | 0.08411 | 0.08388 |

The table is in `items_seeds.md`; everything is in `items_seeds.json`.

## #1 TM-OPT vs MR-OPT on three models

Running (queue `tm_opt/queue_items.sh`).

## #3 GEMM and latency cost of E0M3-dense maps

Waiting for #1. The kernels are built (see the #3 section when done).

## Reproduction

```
/home/dev/n16k64_campaign/tm_opt/queue_items.sh            # #2, #1 (copy in runs/queue_items.sh; log runs/commands_items.log)
python results/tm_opt/analyze_items.py seeds               # items_seeds.{json,md}
python results/tm_opt/analyze_items.py models              # items_models.{json,md}
```
