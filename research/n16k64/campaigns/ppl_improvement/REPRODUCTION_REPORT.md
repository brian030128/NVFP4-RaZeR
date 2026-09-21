# Reproduction report

Extension protocol: `freeze/PROTOCOL_EXTENSION.json`, SHA-256 `bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1`.
Parent authoritative artifact manifest: `432d922a49751603740622a084a665983fb337295048112e68311cd8110e19ee`; parent campaign remains read-only.

This campaign is append-only and uses exact stored maps. Every accepted GPU run includes host and in-container before/during/phase/post ownership evidence, UUID-scoped device isolation, a sidecar no longer than 60 seconds, and a run-level checksum manifest.

Attempts audited before P80 finalization: 169; GPU-hours (including failed/invalid attempts): 89.022122560; maximum concurrent campaign GPUs: 3.

No-GPU reproduction from the extracted reviewer ZIP:

```bash
python tools/verify_bundle.py .
python tools/recompute_tables.py .
```

These commands verify packaged hashes, parse JSON/JSONL/gzip members, validate exact-map binaries, and independently rebuild/check the four machine-readable result tables. Model caches and raw selector tensor workspaces are excluded from the core ZIP and are not needed for these checks.

Optional GPU reproduction uses the frozen model revisions, calibration/evaluation manifests, exact maps, and the launcher policy in the bundle. It requires one homogeneous A6000 or RTX 6000 Ada allocation; portability uses the same map separately on each family. No native FP4/E0M3 performance result is reproduced or implied.
