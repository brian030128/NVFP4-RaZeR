"""Resolve G5 and either enqueue P32 or record its frozen stopping decision."""
from __future__ import annotations

import json
import os
from pathlib import Path

from campaign import runtime


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
MODELS = ("llama8b", "qwen4b", "mistral7b")


def load(path):
    return json.loads(Path(path).read_text())


def write_new(path, value, readonly=False):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    if readonly:
        path.chmod(0o444)


def common_command(model, stage):
    return ["-m", "campaign.extension_calibration_size", "--model", model,
            "--stage", stage, "--freeze", str(CR / "freeze/PROTOCOL_EXTENSION.json"),
            "--freeze-sha256", PROTOCOL, "--attn", "sdpa"]


def quality_command(model, stage):
    return ["-m", "campaign.evaluate_ppl", "--model", model,
            "--plan", str(CR / "plans" / f"{stage.upper()}_calibration_size_{model}.json"),
            "--domains", "wiki,c4", "--length", "2048",
            "--protocol-id", "ppl-improvement-extension-v1",
            "--freeze", str(CR / "freeze/PROTOCOL_EXTENSION.json"),
            "--freeze-sha256", PROTOCOL, "--teacher", "none", "--attn", "sdpa",
            "--no-token-arrays"]


def resolution_reason(continue_to_p32):
    return ("P31 met the predeclared cross-model marginal-gain rule"
            if continue_to_p32 else
            "P31 did not meet the predeclared cross-model marginal-gain rule")


def specs():
    out = []
    previous = "P32_gate"
    for model in MODELS:
        jid = f"P32_calib_{model}"
        out.append({"job_id": jid, "matrix_id": "P32_CAL_SIZE_256",
                    "protocol_id": "ppl-improvement-extension-v1", "gpus": 1,
                    "gpu_model": "a6000", "env": "main", "cpus": 16,
                    "memory": "170g", "priority": 93, "max_invalid_retries": 3,
                    "depends_on": [previous], "command": common_command(model, "p32")})
        previous = jid
    out.append({"job_id": "P32_plan_freeze", "matrix_id": "P32_CAL_SIZE_256",
                "protocol_id": "ppl-improvement-extension-v1", "gpus": 0,
                "gpu_model": "any", "env": "main", "cpus": 8, "memory": "16g",
                "priority": 94, "max_invalid_retries": 0,
                "depends_on": [previous],
                "command": ["-m", "campaign.extension_plans", "--stage", "p32"]})
    previous = "P32_plan_freeze"
    for model in MODELS:
        jid = f"P32_quality_{model}"
        out.append({"job_id": jid, "matrix_id": "P32_CAL_SIZE_256",
                    "protocol_id": "ppl-improvement-extension-v1", "gpus": 1,
                    "gpu_model": "a6000", "env": "main", "cpus": 16,
                    "memory": "150g", "priority": 95, "max_invalid_retries": 3,
                    "depends_on": [previous], "command": quality_command(model, "p32")})
        previous = jid
    out.append({"job_id": "P32_analysis", "matrix_id": "P32_CAL_SIZE_256",
                "protocol_id": "ppl-improvement-extension-v1", "gpus": 0,
                "gpu_model": "any", "env": "main", "cpus": 16, "memory": "32g",
                "priority": 96, "max_invalid_retries": 0,
                "depends_on": [previous],
                "command": ["-m", "campaign.extension_p32_analysis"]})
    out.append({"job_id": "P32_terminal", "matrix_id": "P32_CAL_SIZE_256",
                "protocol_id": "ppl-improvement-extension-v1", "gpus": 0,
                "gpu_model": "any", "env": "main", "cpus": 2, "memory": "4g",
                "priority": 97, "max_invalid_retries": 0,
                "depends_on": ["P32_analysis"],
                "command": ["-m", "campaign.extension_terminal", "--stage", "p32"]})
    return out


def main():
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    gate_path = CR / "provenance/P31_G5_DECISION.json"
    gate = load(gate_path)
    generated = []
    if gate.get("continue_to_p32"):
        for spec in specs():
            path = CR / "queue/jobs" / f"{spec['job_id']}.json"
            write_new(path, spec)
            generated.append({"path": str(path), "sha256": runtime.sha256_file(path)})
        status = "authorized_enqueued"
    else:
        status = "stopped_by_gate"
        spec = {"job_id": "P32_terminal", "matrix_id": "P32_CAL_SIZE_256",
                "protocol_id": "ppl-improvement-extension-v1", "gpus": 0,
                "gpu_model": "any", "env": "main", "cpus": 2, "memory": "4g",
                "priority": 97, "max_invalid_retries": 0,
                "depends_on": ["P32_gate"],
                "command": ["-m", "campaign.extension_terminal", "--stage", "p32"]}
        path = CR / "queue/jobs/P32_terminal.json"
        write_new(path, spec)
        generated.append({"path": str(path), "sha256": runtime.sha256_file(path)})
    resolution = {"schema_version": "1.0", "matrix_id": "P32_CAL_SIZE_256",
                  "status": status, "protocol_sha256": PROTOCOL,
                  "G5_decision_path": str(gate_path),
                  "G5_decision_sha256": runtime.sha256_file(gate_path),
                  "continue_to_p32": bool(gate.get("continue_to_p32")),
                  "generated_queue_specs": generated,
                  "reason": resolution_reason(bool(gate.get("continue_to_p32")))}
    resolution_path = CR / "provenance/P32_GATE_RESOLUTION.json"
    write_new(resolution_path, resolution, readonly=True)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": "P32-gate", "model_revision": runtime.sha256_file(resolution_path),
                   "tokenizer_revision": "not_applicable", "model_class": "gate",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None,
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(resolution_path)] + [x["path"] for x in generated],
                    "summary": {"P32_status": status}, "uncertainty": {},
                    "attempted_endpoints": ["P32_G5_GATE"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps(resolution, sort_keys=True))


if __name__ == "__main__":
    main()
