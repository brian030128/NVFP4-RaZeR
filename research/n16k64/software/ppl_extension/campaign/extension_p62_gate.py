"""Record the predeclared P62 K64 local-transform stopping decision."""
from __future__ import annotations

import json
import os
from pathlib import Path

from campaign import runtime


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"


def load(path):
    return json.loads(Path(path).read_text())


def resolution(spec):
    p62 = spec["P62_local_transform"]
    return {
        "schema_version": "1.0", "matrix_id": "P62_K64_LOCAL_TRANSFORM",
        "status": "stopped_by_gate", "protocol_sha256": PROTOCOL,
        "gate": "G7", "attempts_launched": 0,
        "proof_requirement": p62["required_pre_result_proof"],
        "evidence_at_gate": p62["current_evidence"],
        "reason": p62["fallback"],
        "claim_disposition": "No local-transform quality or compatibility claim; no hidden transform or high-precision state was introduced.",
    }


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    spec_path = CR / "provenance/P60_P62_QUANTIZER_SPEC.json"
    out = resolution(load(spec_path))
    out["quantizer_spec_sha256"] = runtime.sha256_file(spec_path)
    path = CR / "provenance/P62_GATE_RESOLUTION.json"
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, out); path.chmod(0o444)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P62-G7-gate", "model_revision": runtime.sha256_file(path),
                   "tokenizer_revision": "not_applicable", "model_class": "gate",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None,
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(path)], "summary": out,
                    "uncertainty": {}, "attempted_endpoints": ["P62_G7_GATE"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
