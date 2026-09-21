"""One-device clean-allocation probe used before the exact-map anchors."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from campaign import runtime


def main() -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"preflight probe requires exactly one visible GPU, got {torch.cuda.device_count()}")
    runtime.phase("cuda_probe_start")
    prop = torch.cuda.get_device_properties(0)
    x = torch.arange(1024 * 1024, dtype=torch.float32, device="cuda").reshape(1024, 1024) / 1048576.0
    y = x @ x.T
    torch.cuda.synchronize()
    observed = {
        "index": 0,
        "name": torch.cuda.get_device_name(0),
        "uuid": str(prop.uuid),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": prop.total_memory,
        "checksum": float(y.sum().cpu()),
    }
    # Ensure the host sidecar obtains at least one steady-state sample in addition to its
    # launch/final samples.  This is monitoring evidence, not a performance measurement.
    time.sleep(22)
    runtime.phase("cuda_probe_finalize")
    path = runtime.out_dir("gpu_probe") / "GPU_PREFLIGHT_PROBE.json"
    runtime.atomic_json(path, {"schema_version": "1.0", "status": "complete", "device": observed,
                               "quality_values_accessed": False, "performance_measurement": False})
    launch = json.loads((runtime.run_dir / "launch_record.json").read_text())
    campaign_root = Path(os.environ["CAMPAIGN_ROOT"])
    protocol_sha = (campaign_root / "freeze/PROTOCOL_EXTENSION.sha256").read_text().split()[0]
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": protocol_sha,
        "source": {"model_id": "gpu-preflight-probe", "model_revision": "not_applicable",
                   "tokenizer_revision": "not_applicable", "model_class": "not_applicable",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(attention_backend=None, activation_quantizer=None),
        "data": {"calibration_manifest_sha256": None, "evaluation_manifest_sha256": None,
                 "token_hashes": {}, "overlap_audit": None},
        "policies": [],
        "results": {"raw_outputs": [str(path)], "summary": {"device": observed,
                    "quality_values_accessed": False, "performance_measurement": False},
                    "uncertainty": {}, "attempted_endpoints": ["P00_GPU_PREFLIGHT"],
                    "missing_endpoints": []},
        "logs": [], "failures": [],
    })
    print(json.dumps(observed, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
