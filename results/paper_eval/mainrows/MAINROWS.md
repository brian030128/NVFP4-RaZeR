# IF4 and MixFP4 (Zou), simulated, the methods' own rules (1x16, own E2M1, per-token activation scales)

flipquant paper-sm120-runs `evaluation.ppl --mode fake --weight if4 / zou_mixfp4 --act-method own --unit 1x16`; the hybrid models in n16k64-fast. Spot check (non-hybrid models): the main-ppl 44f8cea records (the NVFP4-RaZeR implementation), per-window NLL and installed weight sha256, bit for bit.

| model | IF4 WikiText-2 / C4 | MixFP4 (Zou) WikiText-2 / C4 | spot check |
|---|---:|---:|---|
| Qwen3-1.7B | 19.6007 / 20.9536 | 19.5027 / 21.5547 | if4: NLL =, weights =; zou: NLL =, weights = |
| Qwen3-8B | 9.8852 / 13.6027 | 9.8615 / 13.5916 | if4: NLL =, weights =; zou: NLL =, weights = |
| Mistral-7B-Instruct-v0.3 | 5.6703 / 8.3621 | 5.6629 / 8.3689 | if4: NLL =, weights =; zou: NLL =, weights = |
| Nemotron-Nano-9B-v2 | 8.3911 / 11.4502 | 8.4088 / 11.4578 | n/a (hybrid: n16k64-fast) |
| Phi-4 | 6.6346 / 10.5108 | 6.6392 / 10.5066 | if4: NLL =, weights =; zou: NLL =, weights = |
| Qwen3.8-27B | 7.3343 / 10.1657 | 7.3476 / 10.1639 | n/a (hybrid: n16k64-fast) |
