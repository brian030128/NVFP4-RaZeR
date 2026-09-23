# Claims ledger

Workspace base commit is `db63419cc33b2bbbda2117aad636435a2956532d`.
New analysis scripts are **uncommitted** and identified by hashes in
`ARTIFACT_MANIFEST.sha256` and `RUN_LOG.jsonl`, not claimed to be present in
that commit. Historical outcomes retain their per-run source manifests and
artifact hashes; an exact historical Git commit for all campaign code is not
independently established. See `results/SOURCE_DRIFT.json`,
`results/EVALUATION_REUSE.json`, and `evidence_ledger.csv`. A shared directory
name is not proof of identical executable source.

| ID | Status | Claim | Artifact and row/filter; qualification |
|---|---|---|---|
| C01 | Supported, observed | Natural joint identity varies but exceeds module-random overlap | `results/map_pairs.csv`, `ranking=natural_joint`, all 30 model/pair rows; compare observed intersection to module null, not a population stability probability |
| C02 | Supported, observed | All five joint draws have favorable PPL on both corpora for these three models | `results/quality_results.csv`, `policy=joint`, all 30 cells; `results/draw_variation.csv`, joint rows. Point estimates, not universal accuracy safety |
| C03 | Contradicted as universal claim | Joint always outperforms both matched objectives | `results/quality_contrasts.csv`, complete three-model/five-draw primary PPL panel. CE/KL matched is more favorable at Llama 4/10 and 7/10, Qwen 10/10 and 0/10, Mistral 3/10 and 9/10 points. Observed counterexamples, not a population dominance test |
| C04 | Inconclusive | Joint has lower population cross-draw risk than single objectives | All 150 primary PPL cells are present. `results/draw_variation.csv` retains model/corpus-specific SD/ranges; Qwen CE-matched SD is smaller than joint on both corpora. Five draws do not estimate population tail risk; secondary accuracy remains incomplete |
| C05 | Supported, mathematical | Exact parent SE requires covariance; all same-k passing children imply parent pass | `scripts/test_math.py`, correlated-parent and all-children tests; `results/granularity_results.csv`, Qwen rows with exact parent SE. Not finite-effect calibration |
| C06 | Not universally supported; incomplete across models | Actual NLL degrades monotonically as N increases | `results/GRANULARITY_TABLE.md`: Qwen N8–N256 measured, WikiText worsens pointwise but C4 N128 slightly improves over N64. No adjacent-contrast significance claim. 16 Llama/Mistral coarse cells remain missing; analytical regret is not actual NLL |
| C07 | Not supported | Reliable individual-tile sign/magnitude calibration | `results/effect_diagnostics.csv`, all model/objective/stratum rows; source/hash columns resolve raw fidelity records. Same calibration sample, stratified tiles, baseline restoration not exact |
| C08 | Historical supportive analysis | Aggregate ranking can be useful | [Mechanism verdict](../mechanism/MECHANISM_VERDICT.md) and [follow-up verdict](../followup/FOLLOWUP_VERDICT.md); corresponding source/hash rows in `evidence_ledger.csv`. Read and hashed, not newly executed group experiments |
| C09 | Power-limited historical support | Boundary/corruption pattern | [Boundary verdict](../boundary/BOUNDARY_CORRUPTION_VERDICT.md), formal terminal classification; preserve endpoint labels and unresolved hour-20 deviation, no strict-compliance promotion |
| C10 | Not established | Low overlap proves a connected flat optimum or reorder efficacy | C01 set comparisons supply no connectivity/reorder intervention. Historical reorder context is separately scoped in `results/reorder_ratio_audit.csv` |
| C11 | Failed prefix reproduction gate for all three tested models | Current Ada evaluation exactly reproduces historical A6000 windows | `results/anchor_windows.csv`, Llama `pilot_llama_attempt1/attempt2`, and `results/ADA_ACCURACY_RESOURCE_RESULT.json` for Mistral/Qwen. Token/weight identities match, but maximum window-NLL differences exceed `1e-10`; Mistral 0.00745024 and Qwen 0.04593504. No Ada accuracy promotion. These checks do not isolate the physical cause or establish native execution |
| C12 | Not tested | Native FP4/E0M3 quality, speedup, memory saving, power | Frozen execution scope and run commands: floating-point fake-quant evaluation only; no native-performance endpoint in this campaign |
| C13 | Supported, observed | Favorable PPL does not imply favorable accuracy on every task | `results/accuracy_sample_metrics.csv`, `policy=joint`, fixed seed0: negative point changes on Llama 3/8, Qwen 1/8, Mistral 2/8 tasks. Raw samples independently checked; no new accuracy CI or population-risk conclusion |
| C14 | Supported, observed | Qwen CE-matched is more favorable than joint at all ten five-draw/corpus points; KL-matched is less favorable | `results/quality_contrasts.csv`, `model=qwen4b`, primary contrasts; `results/draw_variation.csv` has all five draws per policy. Across-draw SD is smaller for CE-matched than joint on each corpus. Not universal model or population robustness evidence |
| C15 | Supported, observed | Llama KL natural regresses against baseline at all ten five-draw/corpus points; KL matched-quota is favorable at all ten | `results/draw_variation.csv`, `model=llama8b`, five draws per corpus and policy; independently recomputed in `results/OBJECTIVE_TABLE_AUDIT.json`. Budget and identity both differ; not a uniquely identified density mechanism or population-risk estimate |
| C16 | Contradicted by this attempt | Identical reported calibration forward losses suffice to reconstruct exact historical scores/maps | `results/MISTRAL_CALIBRATION_IDENTITY_DIAGNOSTIC.json`: all 128 teacher/CE/KL losses match, but only 6/224 module score digests match; N8/N16 masks differ. This attempt's parents are rejected. Backward nondeterminism/allocation is an unproven cause. The final repair failed at import before model loading; its CPU-only path fix does not establish numerical recovery or authorize another calibration attempt |
| C17 | Verified reproduction, scoped | All three models' N8/N16 historical full-window PPL is exactly reproduced on A6000 | `results/N8_REPRODUCTION.json` and `results/N16_REPRODUCTION.json`: installed weight/runtime/window identities match, maximum NLL difference 0 for Llama/Qwen/Mistral WikiText 141/146/163 and each C4 256 windows. This is stored-map reproduction, not calibration-score reproduction or native execution evidence |
| C18 | Supported, fixed-seed0 points | Qwen matched-quota accuracy is objective-dependent, not universally preserved | `results/ACCURACY_NEW_AUDIT.json` and `results/SECONDARY_ACCURACY_TABLE_AUDIT.json`: exact-baseline admission; CE matched +1.1689 pp macro, 0/8 negative task changes; KL matched +0.2448 pp macro, 3/8 negative task changes. No new CI, significance, non-inferiority or five-draw accuracy robustness claim |

The explicitly authorized Mistral calibration attempt4 supersedes C16's last
operational-state sentence, not its scientific limitation: it completed after
the CPU import repair, but still matched only 6/224 score streams, with 58/34
N8/N16 mask differences despite all 128 forward CE/KL losses matching.
Historical quant hash and original raw/subset options were restored; this did
not establish score identity. `results/AUTHORIZED_CALIBRATION_AUDIT.json`
retains exact sources, hashes and the rejection. No fifth attempt is authorized.
Llama's authorized attempt4 also failed exact identity: all 128 forward CE/KL
losses match, but only 6/224 score digests match and the N8/N16 masks differ by
71/39 tiles. Its parent moments are equally rejected; source restoration is
not evidence of historical backward equivalence.

Mistral's newly admitted fixed-seed0 secondary accuracy likewise does not
support universal task safety: CE matched macro +0.0032 pp with 3/8 declining
tasks; KL matched +0.3684 pp with 2/8 declining tasks. These are array-checked
points from `results/SECONDARY_ACCURACY_TABLE_AUDIT.json`, not new confidence
intervals, equivalence, non-inferiority or across-draw accuracy evidence.

All new statistical intervals are pointwise exploratory. No new confirmatory
significance, independent five-draw replication or per-tile simultaneous
confidence statement is claimed. Missing results remain missing.
