"""Resolve and enqueue the three frozen conditional breadth stages.

P21, P42, and P52 are successful terminal rows when stopped by their declared
gate.  A stopped gate creates no GPU work and records the reason immutably.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from campaign import runtime


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
MODELS = ("qwen27b", "phi4", "olmo2_13b")
OP = CR / "provenance/P21_P42_P52_BREADTH_OPERATIONALIZATION.json"
RESOURCES = {
    "qwen27b": {"gpus": 2, "memory": "230g", "max_memory_gib": 44, "reserve": True},
    "phi4": {"gpus": 1, "memory": "150g"},
    "olmo2_13b": {"gpus": 1, "memory": "150g"},
}


def load(path):
    return json.loads(Path(path).read_text())


def latest_complete(prefix):
    accepted = []
    for run in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(run.name.rsplit("_attempt", 1)[1])
            launch, status, valid = (load(run / name) for name in
                                     ("launch_record.json", "job_status.json",
                                      "run_record_validation.json"))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if (launch.get("status") == status.get("status") == "complete"
                and valid.get("valid")):
            accepted.append((attempt, run))
    if not accepted:
        raise FileNotFoundError(prefix)
    return max(accepted)[1]


def write_new(path, value, readonly=False):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime.atomic_json(path, value)
    if readonly:
        path.chmod(0o444)


def gpu_shape(spec, model):
    resource = RESOURCES[model]
    spec.update(gpus=resource["gpus"], gpu_model="a6000", env="main",
                cpus=16, memory=resource["memory"], max_invalid_retries=3)
    if resource.get("reserve"):
        spec["reserve"] = True
    return spec


def selector_command(model):
    command = ["-m", "campaign.extension_selector_stats", "--model", model,
               "--freeze", str(CR / "freeze/PROTOCOL_EXTENSION.json"),
               "--freeze-sha256", PROTOCOL, "--attn", "sdpa"]
    if "max_memory_gib" in RESOURCES[model]:
        command.extend(["--max-memory-gib", str(RESOURCES[model]["max_memory_gib"])])
    return command


def quality_command(stage, model):
    command = ["-m", "campaign.evaluate_ppl", "--model", model,
               "--plan", str(CR / "plans" / f"{stage.upper()}_"
                              f"{'density' if stage == 'p21' else 'aggregation' if stage == 'p42' else 'selector'}_breadth_{model}.json"),
               "--domains", "wiki,c4", "--length", "2048",
               "--protocol-id", "ppl-improvement-extension-v1",
               "--freeze", str(CR / "freeze/PROTOCOL_EXTENSION.json"),
               "--freeze-sha256", PROTOCOL, "--teacher", "none",
               "--attn", "sdpa", "--no-token-arrays"]
    if "max_memory_gib" in RESOURCES[model]:
        command.extend(["--max-memory-gib", str(RESOURCES[model]["max_memory_gib"])])
    return command


def job_base(jid, matrix, priority, depends, command, gpus=0):
    spec = {"job_id": jid, "matrix_id": matrix,
            "protocol_id": "ppl-improvement-extension-v1",
            "gpus": gpus, "gpu_model": "any" if gpus == 0 else "a6000",
            "env": "main", "cpus": 16, "memory": "64g" if gpus == 0 else "150g",
            "priority": priority, "max_invalid_retries": 0 if gpus == 0 else 3,
            "depends_on": list(depends), "command": command}
    return spec


def selector_spec(stage, model, dependency, priority):
    spec = job_base(f"{stage.upper()}_selector_stats_{model}",
                    f"{stage.upper()}_SELECTOR_STATS",
                    priority, [dependency], selector_command(model), gpus=1)
    return gpu_shape(spec, model)


def find_or_create_selector(stage, model, dependency, priority, generated):
    # A P21 statistics artifact contains all three P50 candidate columns and can
    # therefore be reused by P52 without rerunning an already completed GPU job.
    if stage != "p21":
        for earlier in ("P21", "P42"):
            path = CR / "queue/jobs" / f"{earlier}_selector_stats_{model}.json"
            if path.exists():
                return f"{earlier}_selector_stats_{model}"
    jid = f"{stage.upper()}_selector_stats_{model}"
    spec = selector_spec(stage, model, dependency, priority)
    path = CR / "queue/jobs" / f"{jid}.json"
    write_new(path, spec)
    generated.append(path)
    return jid


def stage_jobs(stage, need_selector):
    matrix = {"p21": "P21_K2_DENSITY_BREADTH",
              "p42": "P42_DRAW_AGG_BREADTH",
              "p52": "P52_SELECTOR_BREADTH"}[stage]
    base_priority = {"p21": 97, "p42": 118, "p52": 122}[stage]
    generated = []
    selector_deps = {}
    previous = f"{stage.upper()}_gate"
    if need_selector:
        for offset, model in enumerate(MODELS):
            selector_deps[model] = find_or_create_selector(
                stage, model, previous, base_priority + offset, generated)
            previous = selector_deps[model]
    map_previous = previous
    for offset, model in enumerate(MODELS):
        jid = f"{stage.upper()}_maps_{model}"
        dependencies = [map_previous]
        if model in selector_deps:
            dependencies.append(selector_deps[model])
        spec = job_base(jid, matrix, base_priority + 3 + offset,
                        dependencies,
                        ["-m", "campaign.extension_breadth_maps",
                         "--stage", stage, "--model", model])
        path = CR / "queue/jobs" / f"{jid}.json"
        write_new(path, spec); generated.append(path)
        map_previous = jid
    previous = map_previous
    for offset, model in enumerate(MODELS):
        jid = f"{stage.upper()}_quality_{model}"
        spec = job_base(jid, matrix, base_priority + 6 + offset,
                        [previous], quality_command(stage, model), gpus=1)
        gpu_shape(spec, model)
        path = CR / "queue/jobs" / f"{jid}.json"
        write_new(path, spec); generated.append(path)
        previous = jid
    jid = f"{stage.upper()}_analysis"
    spec = job_base(jid, matrix, base_priority + 9, [previous],
                    ["-m", "campaign.extension_breadth_analysis", "--stage", stage])
    path = CR / "queue/jobs" / f"{jid}.json"
    write_new(path, spec); generated.append(path)
    terminal = job_base(f"{stage.upper()}_terminal", matrix, base_priority + 10,
                        [jid], ["-m", "campaign.extension_terminal",
                                "--stage", stage])
    path = CR / "queue/jobs" / f"{stage.upper()}_terminal.json"
    write_new(path, terminal); generated.append(path)
    return generated


def stopped_terminal(stage):
    matrix = {"p21": "P21_K2_DENSITY_BREADTH",
              "p42": "P42_DRAW_AGG_BREADTH",
              "p52": "P52_SELECTOR_BREADTH"}[stage]
    priority = {"p21": 107, "p42": 128, "p52": 132}[stage]
    spec = job_base(f"{stage.upper()}_terminal", matrix, priority,
                    [f"{stage.upper()}_gate"],
                    ["-m", "campaign.extension_terminal", "--stage", stage])
    path = CR / "queue/jobs" / f"{stage.upper()}_terminal.json"
    write_new(path, spec)
    return [path]


def gate_inputs(stage):
    p10_run = latest_complete("P10_P11_analysis")
    p20_run = latest_complete("P20_analysis")
    p10_path = p10_run / "analysis_breadth/P10_P11_BREADTH_ANALYSIS.json"
    p20_path = p20_run / "analysis_p20/P20_DENSITY_CONTROL_ANALYSIS.json"
    p10, p20 = load(p10_path), load(p20_path)
    common = {"P10_P11_analysis": {"path": str(p10_path),
                                     "sha256": runtime.sha256_file(p10_path)},
              "P20_analysis": {"path": str(p20_path),
                                "sha256": runtime.sha256_file(p20_path)}}
    if stage == "p21":
        decisions = {"G1": bool(p10["gates"]["G1"]["passed"]),
                     "G3": bool(p20["G3"]["passed"])}
        authorized = all(decisions.values())
        need_selector = p20["strongest_deterministic_control"]["selector"] == "activation_weighted"
        reason = "P10/P11 G1 and P20 G3 both passed" if authorized else "P10/P11 G1 and/or P20 G3 failed"
    elif stage == "p42":
        p41_path = CR / "provenance/P41_AGGREGATION_SELECTION.json"
        p30_run = latest_complete("P30_analysis")
        p30_path = p30_run / "analysis_draws/P30_FIVE_DRAW_ANALYSIS.json"
        common.update(P41_selection={"path": str(p41_path),
                                     "sha256": runtime.sha256_file(p41_path)},
                      P30_analysis={"path": str(p30_path),
                                    "sha256": runtime.sha256_file(p30_path)})
        decisions = {"G2": bool(p10["gates"]["G2"]["passed"]),
                     "G3": bool(p20["G3"]["passed"])}
        authorized = all(decisions.values())
        need_selector = p20["strongest_deterministic_control"]["selector"] == "activation_weighted"
        reason = "P10/P11 G2 and P20 G3 both passed" if authorized else "P10/P11 G2 and/or P20 G3 failed"
    else:
        p51_path = CR / "provenance/P51_G6_DECISION.json"
        p41_path = CR / "provenance/P41_AGGREGATION_SELECTION.json"
        p51 = load(p51_path)
        decisions = {"G6": bool(p51["G6"]["passed"]),
                     "replacement_selector_accepted": bool(p51["replacement_selector_accepted"])}
        authorized = all(decisions.values())
        need_selector = True
        common.update(P51_decision={"path": str(p51_path),
                                    "sha256": runtime.sha256_file(p51_path)},
                      P41_selection={"path": str(p41_path),
                                     "sha256": runtime.sha256_file(p41_path)})
        reason = "P51 G6 accepted the frozen replacement selector" if authorized else "P51 G6 did not accept a replacement selector"
    return authorized, need_selector, decisions, reason, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("p21", "p42", "p52"), required=True)
    args = ap.parse_args()
    stage = args.stage
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    authorized, need_selector, decisions, reason, inputs = gate_inputs(stage)
    generated = (stage_jobs(stage, need_selector) if authorized
                 else stopped_terminal(stage))
    resolution = {
        "schema_version": "1.0",
        "matrix_id": {"p21": "P21_K2_DENSITY_BREADTH",
                      "p42": "P42_DRAW_AGG_BREADTH",
                      "p52": "P52_SELECTOR_BREADTH"}[stage],
        "status": "authorized_enqueued" if authorized else "stopped_by_gate",
        "protocol_sha256": PROTOCOL,
        "operationalization_sha256": runtime.sha256_file(OP),
        "authorized": authorized, "gate_decisions": decisions,
        "requires_selector_statistics": bool(authorized and need_selector),
        "reason": reason, "decision_inputs": inputs,
        "generated_queue_specs": [{"path": str(path),
                                    "sha256": runtime.sha256_file(path)}
                                   for path in generated],
    }
    resolution_path = CR / "provenance" / f"{stage.upper()}_GATE_RESOLUTION.json"
    write_new(resolution_path, resolution, readonly=True)
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1",
        "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": f"{stage}-breadth-gate",
                   "model_revision": runtime.sha256_file(resolution_path),
                   "tokenizer_revision": "not_applicable", "model_class": "gate",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None,
                 "evaluation_manifest_sha256": None, "token_hashes": {},
                 "overlap_audit": None},
        "policies": [],
        "results": {"raw_outputs": [str(resolution_path)]
                                  + [str(path) for path in generated],
                    "summary": {"status": resolution["status"],
                                "gate_decisions": decisions},
                    "uncertainty": {},
                    "attempted_endpoints": [f"{stage.upper()}_GATE"],
                    "missing_endpoints": []},
        "logs": [], "failures": []})
    print(json.dumps(resolution, sort_keys=True))


if __name__ == "__main__":
    main()
