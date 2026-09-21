# Final submission risk audit

Recommended scope: **N16/N8 engineering trade-off study with deployment benefit unmeasured**.

This is a software/fake-quant quality campaign. Native FP4/E0M3 Tensor Core execution, the external N8 ~13% / N16 ~1.5% overhead estimates, latency/speedup, area, and power are not claims of this work.

| risk | disposition | evidence/reason |
|---|---|---|
| R01 | mitigated | k2 is explicitly labeled post-hoc; only the later held-out winner is prospective |
| R02 | closed | matched selected-weight controls completed; interpret by G3 |
| R03 | closed_for_reporting | Qwen-outlier-excluded summaries are retained |
| R04 | open_negative_result | P51 exact-tile G6 result |
| R05 | closed_by_claim_removal | no CE/KL independence or per-tile significance claim |
| R06 | mitigated | exact hashes and five-draw/aggregation evidence; map identity is not overclaimed |
| R07 | open_blocks_SOTA | full baseline table retained; G9 controls framing |
| R08 | closed_as_evidence | natural and matched-density N16/N8 contrasts reported under G4 |
| R09 | closed_as_quality_screen | format-preserving activation candidates screened; incompatible unfused work excluded |
| R10 | open_blocks_general_quality_claim | accuracy/GSM8K/PG19 follow frozen gate |
| R11 | closed | five draws report within-evaluation and between-draw uncertainty separately |
| R12 | closed | two reserved new families evaluated only after winner hash lock |
| R13 | open_out_of_scope | no native GPU, overhead, speedup, area, or power claim is made |
| R14 | closed | G7 excludes hidden metadata/unfused/high-precision scope drift |
| R15 | closed | all development candidates and staged selection rules are retained |
| R16 | pending_P81 | redacted chain, Windows-safe archive, and clean extraction are independently verified in P81 |
| R17 | closed | GPU ownership/concurrency audit |
| R18 | closed_as_preserved_evidence | all Qwen OOM/failure attempts remain in the append-only inventory |

## Claim classifications

| claim | classification | limitation |
|---|---|---|
| C01 | supported_post_hoc | k2 was selected from prior results; this is a post-hoc robustness extension |
| C02 | unsupported_practically_equivalent | development/breadth gate; no per-model threshold tuning |
| C03 | supported | requires both global and per-module selected-weight controls |
| C04 | unsupported_tradeoff_only | native deployment overhead was not measured |
| C05 | unsupported_old_selector_fallback | tile interventions are the mechanistic inference unit |
| C06 | unsupported | software/fake-quant PPL only |
| C07 | unsupported_or_stopped_by_gate | GSM8K and PG19 remain secondary/exploratory transfers |
| C08 | unsupported | broad SOTA requires a corrected, scope-matched full-panel superiority family; held-out component alone is insufficient |
| C09 | unsupported | quality portability only; no native performance implication |
| C10 | unsupported | old exact CE sign was 3/9; aggregate map CIs are conditional on the stored map |
| C11 | unsupported | CE/KL dependence invalidates the independence argument |
| C12 | out_of_scope_unsupported | all experiments use BF16-dequantized software/fake quantization |

All adverse baselines, negative results, OOM/crash/invalid attempts, and stopped-by-gate cells remain in `FAILED_OR_SKIPPED_RUNS.json` and the append-only registry. P81 changes R16 from pending to closed only after an independent clean-extraction verification passes.
