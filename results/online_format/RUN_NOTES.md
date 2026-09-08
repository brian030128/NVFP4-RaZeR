# Execution notes

Job332256 failed the all-zero correctness case before loading models or
evaluation data. The canonical tensor quantizer divides by its global maximum,
which is zero in this case. Add an explicit zero-input result in the new
selector. No algorithm setting or evaluation-dependent decision changed.
