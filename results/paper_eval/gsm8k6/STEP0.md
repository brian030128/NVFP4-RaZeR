# Part Y, step 0: the harness (flipquant paper-sm120-runs 4bbe0b1)

PROTOCOL.md part Y. Overall: **all checks passed**.

## (a) The generation path: f2a56f1 (part N's harness) vs 4bbe0b1

`evaluation/downstream.py` and `evaluation/lm_eval_datasets.py`: byte-identical. Everything changed since f2a56f1 (code, outside results / docs / tests):  11 files changed, 661 insertions(+), 26 deletions(-) -- calibration, the opt-in `--fo6-tensor-qmax` and `--gptq-candidates`, and the registry; checked numerically by (b) and (c').

## (b) Default paths: 4bbe0b1 vs main 3db0072

Code that differs (evaluation-relevant directories):

```
evaluation/accuracy.py   |   3 +-
 evaluation/downstream.py | 115 +++----------------------------------
 evaluation/ref_nll.py    | 145 +++++++++++++++++++++++++++++++++++++++++++++++
 flipquant/data.py        |   7 ++-
 flipquant/formats.py     |  27 ++-------
 flipquant/gptq.py        |  93 +-----------------------------
 flipquant/linear.py      |  19 +------
 models/cli.py            |  30 +---------
 models/registry.py       |  13 +----
 models/sm120.py          |  81 +-------------------------
 10 files changed, 178 insertions(+), 355 deletions(-)
```

Numeric parity, Llama-3.2-1B-Instruct, `evaluation.ppl --limit 2` (2 WikiText-2 + 2 C4 windows) on both commits, the same session and env:

| row | per-window NLL bitwise (wiki, c4) | other |
|---|---|---|
| bf16 | yes, yes |  |
| nvfp4 | yes, yes |  |
| fo6 | yes, yes |  |
| focus | yes, yes |  |
| gptq | yes, yes | codes_sha256 952850d7d8cf / 952850d7d8cf |
| if4 | yes, yes | installed weights ab7e135e9407 / ab7e135e9407 |
| zou | yes, yes | installed weights 0d424a91679b / 0d424a91679b |
| fq-8x64 | yes, yes |  |
| fq-16x64 | yes, yes |  |
| fq-256x64 | yes, yes |  |

## (c') Nemotron-Nano-9B-v2 FourOverSix / FlipQuant 16x64 vs part N's samples

All 1,319 problems with part N's exact command, env and batch 64 (these runs are the panel's cells). Per (problem, filter): the generation (`resps`) and the extracted answer (`filtered_resps`).

| row | samples | generations identical | answers identical | first difference |
|---|---:|---:|---:|---|
| fo6 | 2638 | 2638 | 2638 | none |
| fq-16x64 | 2638 | 2638 | 2638 | none |

Checks: (a) yes, (b) yes, (c) yes.
