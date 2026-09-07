# Uncalibrated domain stress tests

Declared while job 329471 is loading the model, before any new fitting or
held-out results. After that job freezes its maps, evaluate all four candidates
from both seeds on 128 GSM8K test problems with reference solutions and 128
MBPP test programming problems with reference code. Sample without replacement
using seed 20260917, pin dataset revisions, and record row IDs and token hashes.
Use plain `Question: ...\nAnswer: ...` and `Problem: ...\nCode:\n...` text,
at most 2048 tokens. Evaluate mean full-text next-token NLL per example,
with paired mean/SE and exponentiated mean NLL. This is a language-modeling
stress test of math and code text, not generated-answer accuracy or pass@k.
Variable-length examples receive equal weight; separately report token-weighted
perplexity. No math/code training examples are used to choose maps or gates.

Report candidates and baseline fallback for rejected exports. No stress-test
measurement selects a map or changes the rule. Two broad text calibration
domains and these two unseen task-text domains still do not establish
universality or causal decoding performance. Fake-quantization scope and
activation formulas remain identical to the primary experiment.
