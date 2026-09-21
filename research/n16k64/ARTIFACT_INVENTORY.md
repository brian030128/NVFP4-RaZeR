# N16K64 artifact inventory

This snapshot is a review layer, not a bulk upload of the compute workspace. It
captures the scientific definitions, final tables, negative results, audits,
figures, and the code versions needed to understand the work. Exact lineage for
every copied file is in [`SNAPSHOT_SOURCE_INDEX.json`](SNAPSHOT_SOURCE_INDEX.json);
the machine-readable inclusion/exclusion record is
[`ARTIFACT_INVENTORY.json`](ARTIFACT_INVENTORY.json).

## What is included

| Class | Location | Treatment |
|---|---|---|
| Current repository implementation and already-published research | repository root, `quantize/`, `native/`, `results/`, `scripts/`, `slurm/` | Inherited unchanged from recorded `main` commit `2d3e8f397d57009ba843c83b2a75a176262b026c`. |
| Five N16K64 campaigns | [`campaigns/`](campaigns/) | Protocols, reports, compact result JSON/CSV, audits, failure summaries, and figures. Text machine paths and account/device identifiers are pseudonymized; numerical fields are unchanged. |
| Historical campaign code | [`software/`](software/) | Primary, PPL-extension, and mechanism/follow-up/boundary snapshots. Only path plumbing and synthetic GPU identifiers in tests were made portable. Original/public hashes are both recorded. |
| Review metadata | this directory | Research index, claim-to-evidence matrix, validation report, inventories, redaction log, and checksums. |

The copied campaign layer originally contained 512 source artifacts. The reverse
completeness audit added omitted producers, plans, tests, dependencies, calibration
manifests and compact paired arrays; current membership and checksums are in the
source index and [addition ledger](validation/COMPLETENESS_ADDITIONS.json).
The complete review tree is about 32 MiB, suitable for ordinary Git.
CRLF CSVs and legacy end-of-line whitespace in copied campaign/source evidence
are preserved; the subtree `.gitattributes` suppresses whitespace warnings only
for those immutable copies. Newly authored review files remain subject to the
normal Git whitespace checks. Worktree modes were normalized independently of
content: only files with a shebang are executable, and every evidence/data file
is non-executable. This changes no recorded SHA-256.

## What is intentionally excluded

- Model weights, Hugging Face caches, and complete third-party datasets.
- Exact binary maps, calibration moments, score shards, per-token arrays, and
  most paired-cluster NPZ arrays. The six small boundary paired-cluster arrays
  are now included byte-identically; they total less than 1 MiB.
- Full run/container logs, environment trees, caches, and complete append-only
  registries.
- The reviewer archives themselves: the primary bundle is roughly 39 GiB over
  seven ZIPs, the PPL-extension handoff is about 1.37 GiB, and the mechanism
  handoff is about 2.89 GiB.
- Machine-private GPU process details. Co-tenancy failures and attempt counts
  remain visible in redacted summaries.

These exclusions prevent exact-map GPU replay and raw analyses for other
campaigns using the Git branch alone. Boundary paired-cluster arrays are sufficient
for independent recomputation of its reported statistical endpoints. This review
recomputed their mean p=1 effects only, not a new bootstrap. Exclusions do not remove compact outcomes, protocols,
failure classifications, or audit verdicts. The full SHA-256 values and sizes
of all local reviewer archives are recorded in the JSON inventory. They are
internal-workspace artifacts and are **not** claimed to be publicly accessible.
No Git LFS, paid storage, or external upload was enabled.

The source-to-final mapping, original tracked draft disposition, ignored-file
audit, restoration instructions and package limitations are in
[INTEGRATION_AUDIT.md](INTEGRATION_AUDIT.md). Daily development is now in the
original `mixfp4` checkout; the delivery clone is retained only as temporary backup.

## Campaign seals

| Campaign | Manifest SHA-256 | Entries / attempts | Public review directory |
|---|---|---:|---|
| Primary full validation | `432d922a49751603740622a084a665983fb337295048112e68311cd8110e19ee` | 78,705 entries; 338 run directories | [`primary/`](campaigns/primary/) |
| PPL-improvement extension | `6a214e94b5bd04b92f6f7179aff021016201d12d0317ff42e643c5e03e320406` | 441 entries; 169 attempts | [`ppl_improvement/`](campaigns/ppl_improvement/) |
| Mechanism | `1cb942724aeacc2c74862377bfbae8899c76ac6caa824a3f79cfd6c8e78033c4` | 3,154 entries; 10 run directories | [`mechanism/`](campaigns/mechanism/) |
| Follow-up | `b658a422d6dd53ab2f2442cf8ffcf11c3a33d9beea0d9bf6047f228c0677ccef` | 3,213 entries; 10 run directories | [`followup/`](campaigns/followup/) |
| Boundary/corruption | `bfbc8fe52204b0806f25757cc0c19d54a807af86e9f9c85ca221de612328d1fc` | 4,241 entries; 17 run directories | [`boundary/`](campaigns/boundary/) |

The seals above describe the internal immutable campaigns. The Git copies may
have different hashes because text-only redaction is deliberate. Both hashes
are available per file in `SNAPSHOT_SOURCE_INDEX.json`.
