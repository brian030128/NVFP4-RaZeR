# Confirmation reserved before development results

This plan is written while job332316 is running, before its evaluation
results. Execute it only if the frozen development screen passes. A failed
development method is not rescued by selecting this panel instead.

Keep the entire algorithm, training corpus, seeds,256 updates, candidate
formats,8x64 granularity and quantization scope unchanged. Export the final
iterate. Add EleutherAI/pythia-410m and EleutherAI/pythia-1.4b, resolving and
recording revisions before scoring. They add one architecture family at two
sizes. Reuse the already-frozen maps for Llama1B, OPT350M and Qwen0.6B.

New evaluation domains, not used to choose a method in this continuation:

- Literature: emozilla/pg19-test, test split, text field.
- Scientific articles: ccdv/arxiv-summarization, test split, article field.
- Government reports: ccdv/govreport-summarization, test split, report field
  (verify dataset schema before execution; an unavailable field is an
  implementation issue to resolve from the dataset card, not a reason to
  choose a favorable replacement dataset).

Resolve dataset revisions and record them before loading examples. For each
model/domain use the first64 distinct documents with at least512 tokens,
one random512-token window per document, independent RNG seed20260925 for
each model/domain. Record document and token hashes. Do not use loss to
exclude examples. Check document hashes against the64 C4 training documents;
exclude exact overlap before evaluating any policy and record exclusions.
No claim of absence from the source models' original pretraining data follows.

Evaluate FourOverSix, fixed-candidate weight MSE, stale256, fixed-budget and
relinearized policies on identical examples. The15-cell confirmation screen
requires >=0.01PPL gain vsFourOverSix in>=12cells, no supported harm, and
lower point NLL than stale256 and weight MSE in>=9cells each. Both fitting
objectives must improve on each of the two newly trained models. A failure
is preserved; no choosing another seed, subset, checkpoint or model size.

Report paired document-level mean NLL and descriptive2SE, raw losses, map
counts, reversals, fitting and optimization time. Current dynamic activation
tensor scales depend on the entire teacher-forced window, as in the prior
development studies. These are matched fake-quantized reference-text scores,
not a certified causal sequence likelihood or autoregressive generation
benchmark. Confirmation under this protocol does not establish kernel
performance, decoding accuracy, calibration-free selection or novelty.

Model and data sources:
https://huggingface.co/EleutherAI/pythia-1.4b
https://huggingface.co/datasets/emozilla/pg19-test
https://huggingface.co/datasets/ccdv/arxiv-summarization
https://huggingface.co/datasets/ccdv/govreport-summarization
