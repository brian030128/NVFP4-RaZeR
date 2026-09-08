# Attribution and claim boundary

[AND-mask](https://arxiv.org/abs/2009.00329) explicitly uses gradient agreement
across environments as a route to invariance. Our source-consensus score is
closely related in motivation. Calling gradient agreement itself novel would
be incorrect.

[GAQAT](https://arxiv.org/abs/2412.05551) studies gradient conflicts in
quantization-aware domain generalization, including selective scale-gradient
freezing. [QT-DoG](https://arxiv.org/abs/2410.06020) studies quantization as
regularization for domain generalization. They are relevant prior art even
though this experiment changes only fixed-candidate FP4 tile-format bits.

The useful empirical question here is narrower: does a fixed source-consensus
rule recover transferable gains at a legal8x64 granularity when an entire
source family is withheld from its scoring table? A positive answer would
still require comparison with appropriate prior algorithms, larger models,
causal/decode evaluation, and untouched confirmation before a strong paper
claim. This document does not assert a positive answer in advance.
