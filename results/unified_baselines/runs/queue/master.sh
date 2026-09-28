#!/bin/bash
# The whole study in order (PROTOCOL.md): phase 1, then 2, then 3. A phase that stops ends the queue.
Q=/home/dev/n16k64_campaign/unified_baselines
$Q/phase1_llama.sh && $Q/phase2_mistral.sh && $Q/phase3_probes.sh
