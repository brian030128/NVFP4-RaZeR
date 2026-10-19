# Part AA: the calibration windows

The release rule, extended (PROTOCOL.md part AA): the calibration record's 128 windows, then N - 128 by the vendored extension rule (seed 20260930; OpenWebMath train-00000 @ fde8ef8 and CodeParrot-clean file-000000000001 @ 35a59fb; (N - 128) / 2 per source; skipping the record's fit documents, the three development records and the published C4 evaluation documents). Drawn on the CPU by calibration.fit_windows (calib-size f912a30); every window recorded by document sha256, offset and token sha256.

| model | N | extension windows | skipped (fit / dev / C4) | contains | nesting checks | token-list sha256 | drawn in (s) |
|---|---:|---:|---|---:|---|---|---:|
| Llama-3.2-1B-Instruct | 1,024 | 896 | 128 / 192 / 0 | 256 | all pass | `d73b2bb2f3179c9a…` | 27 |
| Llama-3.2-1B-Instruct | 2,048 | 1,920 | 128 / 192 / 0 | 1,024 | all pass | `ed177316f06aa935…` | 35 |
| Phi-4 | 1,024 | 896 | 128 / 192 / 0 | 256 | all pass | `a78f561bb8561686…` | 29 |
| Phi-4 | 2,048 | 1,920 | 128 / 192 / 0 | 1,024 | all pass | `18dfa61a5184b5cb…` | 37 |
| Qwen3.8-27B | 2,048 | 1,920 | 128 / 192 / 233 | 256 | all pass | `68bbb9e7e840635d…` | 34 |

Nesting: each source's first (n - 128) / 2 extension windows equal the smaller set's (the release 16x64 trainer report for 256), record by record; the skip counts equal the release run's; every token hash is distinct. The full token-list sha256 values are in calib_size.json.

## 13-gram audit

Word 13-grams (lowercase; characters outside [a-z0-9] separate words; word 13-grams within each text) of every window of each model's 2,048 set, decoded with its tokenizer, against the evaluation texts (supersets of what is scored):

| evaluation set | items | words | 13-grams |
|---|---:|---:|---:|
| wikitext2_test | 1 | 206,718 | 206,706 |
| c4_validation_eval | 297 | 794,955 | 791,391 |
| piqa | 1,838 | 82,035 | 60,385 |
| hellaswag | 10,042 | 1,926,856 | 1,806,352 |
| winogrande | 1,267 | 27,030 | 11,826 |
| arc_easy | 2,376 | 101,046 | 72,534 |
| arc_challenge | 1,172 | 59,593 | 45,529 |
| lambada_openai | 5,153 | 316,427 | 254,591 |
| gsm8k_test | 1,319 | 143,399 | 127,571 |
| gsm8k_llama_fewshot | 8 | 425 | 329 |

Windows with a shared 13-gram, by evaluation set and membership (release = the 256 set; 1024 / 2048 = the windows each larger set adds):

| model | windows | hits |
|---|---:|---|
| Llama-3.2-1B-Instruct | 2,048 | none |
| Phi-4 | 2,048 | none |
| Qwen3.8-27B | 2,048 | none |
| Mistral-7B-Instruct-v0.3 | 2,048 | none |
| Llama-3.1-8B-Instruct | 2,048 | none |
| Nemotron-Nano-9B-v2 | 2,048 | none |
| Qwen3-1.7B | 2,048 | none |
| Qwen3-8B | 2,048 | none |

**Pilot gate (no hit among the 1,024 sets' new windows of the pilot models): passed.**

**Positive control** (calib_size_audit_control.py). For each model, 30 windows were chosen with a fixed seed and re-scanned as they are, then with one evaluation item spliced into the middle of the decoded text: gsm8k_test item 663 (72 words), hellaswag item 2471 (112 words), lambada_openai item 3234 (53 words), arc_challenge item 98 (33 words), wikitext2_test item test (78 words). A splice counts as detected when the audit attributes it to its own set and item.

| model | windows | 13-grams scanned | plain windows with a hit | splices detected (of 30 each) |
|---|---:|---:|---:|---|
| Llama-3.2-1B-Instruct | 2,048 | 588,798 | 0 | gsm8k_test 30, hellaswag 30, lambada_openai 30, arc_challenge 30, wikitext2_test 30 |
| Phi-4 | 2,048 | 586,983 | 0 | gsm8k_test 30, hellaswag 30, lambada_openai 30, arc_challenge 30, wikitext2_test 30 |
| Qwen3.8-27B | 2,048 | 548,025 | 0 | gsm8k_test 30, hellaswag 30, lambada_openai 30, arc_challenge 30, wikitext2_test 30 |
| Mistral-7B-Instruct-v0.3 | 2,048 | 474,361 | 0 | gsm8k_test 30, hellaswag 30, lambada_openai 30, arc_challenge 30, wikitext2_test 30 |
| Llama-3.1-8B-Instruct | 2,048 | 587,039 | 0 | gsm8k_test 30, hellaswag 30, lambada_openai 30, arc_challenge 30, wikitext2_test 30 |
| Nemotron-Nano-9B-v2 | 2,048 | 560,529 | 0 | gsm8k_test 30, hellaswag 30, lambada_openai 30, arc_challenge 30, wikitext2_test 30 |
| Qwen3-1.7B | 2,048 | 572,132 | 0 | gsm8k_test 30, hellaswag 30, lambada_openai 30, arc_challenge 30, wikitext2_test 30 |
| Qwen3-8B | 2,048 | 572,132 | 0 | gsm8k_test 30, hellaswag 30, lambada_openai 30, arc_challenge 30, wikitext2_test 30 |

Control: **pass**. So the audit's result is not an artifact of the scan: it detects evaluation text that is there.

