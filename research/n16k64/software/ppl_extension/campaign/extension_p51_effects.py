"""Measure frozen P51 exact single-tile CE/KL and local reconstruction effects."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

from campaign import calibrate as C
from campaign import data as D
from campaign import models as MOD
from campaign import policies as P
from campaign import quant as Q
from campaign import runtime
from campaign.extension_p50_maps import PROTOCOL
from campaign.extension_selector_stats import reconstruction_scores
from campaign.fidelity import losses


CR = Path(os.environ["CAMPAIGN_ROOT"])


def load(path):
    return json.loads(Path(path).read_text())


def tile_slices(shape, row_tile, k_tile):
    if shape[0] % 16 or shape[1] % 64:
        raise ValueError("weight shape is not N16K64 divisible")
    if not (0 <= row_tile < shape[0] // 16 and 0 <= k_tile < shape[1] // 64):
        raise IndexError("tile coordinate outside weight")
    return (slice(row_tile * 16, (row_tile + 1) * 16),
            slice(k_tile * 64, (k_tile + 1) * 64))


def stderr(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    return float(values.std(unbiased=True) / math.sqrt(values.numel())) if values.numel() > 1 else 0.0


def installed_weight_sha256(modules):
    """Hash current scoped weight bytes with the Installer's exact convention."""
    h = hashlib.sha256()
    for name, module in modules.items():
        weight = module.weight.detach().contiguous().view(torch.uint8).cpu().numpy()
        h.update(name.encode())
        h.update(memoryview(weight))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("llama8b", "qwen4b", "mistral7b"))
    ap.add_argument("--sample-manifest", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--freeze-sha256", required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    if args.freeze_sha256 != PROTOCOL or runtime.sha256_file(args.freeze) != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    sample_path = Path(args.sample_manifest)
    sample = load(sample_path)
    sample_freeze = load(CR / "provenance/P51_SAMPLE_FREEZE.json")
    frozen = next((row for row in sample_freeze["manifests"] if row["model"] == args.model), None)
    if frozen is None or frozen["sha256"] != runtime.sha256_file(sample_path):
        raise SystemExit("sample manifest is not the frozen P51 manifest")
    if sample.get("status") != "FROZEN_BEFORE_EXACT_EFFECTS" or len(sample["tiles"]) != 60:
        raise SystemExit("invalid P51 sample manifest")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0); torch.set_num_threads(8)
    out = runtime.out_dir("tile_fidelity")
    report_path = out / "P51_TILE_EFFECTS.json"
    report = {"schema_version": "1.0", "status": "running", "model": args.model,
              "protocol_sha256": PROTOCOL, "sample_manifest": str(sample_path),
              "sample_manifest_sha256": runtime.sha256_file(sample_path),
              "sampling_spec_sha256": sample["sampling_spec_sha256"],
              "audit_indices": sample["audit_indices"], "audit_token_sha256": sample["audit_token_sha256"],
              "quantization": {"baseline": "all FourOverSix E2M1 weights with causal per-token FourOverSix rows",
                               "intervention": "one frozen N16K64 tile replaced by E0M3; all else unchanged",
                               "teacher": "pristine BF16 model", "tf32": False},
              "baseline": None, "tiles": [], "failures": []}
    save = lambda: runtime.atomic_json(report_path, report)
    save()

    tok = MOD.load_tokenizer(args.model)
    seqs, _, _ = C.calibration_inputs(args.model, "seed0", tok)
    if [D.sha(seqs[i]) for i in sample["audit_indices"]] != sample["audit_token_sha256"]:
        raise RuntimeError("P51 audit sequence hashes differ from freeze")
    audit = [seqs[i] for i in sample["audit_indices"]]
    runtime.phase("load_pristine_model")
    model, loading = MOD.load_model(args.model, attn="sdpa", device_map="cuda")
    modules = MOD.scope(model, args.model)
    mm_sha, _ = MOD.module_manifest(modules)
    stats = torch.load(sample["selector_statistics"]["path"], map_location="cpu", weights_only=False)
    if runtime.sha256_file(sample["selector_statistics"]["path"]) != sample["selector_statistics"]["sha256"]:
        raise RuntimeError("selector-statistics digest drift")
    if stats["module_manifest_sha256"] != mm_sha or list(modules) != stats["names"]:
        raise RuntimeError("P51 model/module manifest differs from selector freeze")

    needed = defaultdict(list)
    for row in sample["tiles"]:
        if row["module"] not in modules:
            raise RuntimeError("sample references a module outside frozen scope")
        needed[row["module"]].append(row)
    cached_inputs = {name: [] for name in needed}

    def capture(name):
        def hook(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            cached_inputs[name].append(x.to("cpu", dtype=torch.bfloat16, copy=True))
        return hook

    handles = [modules[name].register_forward_pre_hook(capture(name)) for name in needed]
    teacher = []
    dev0 = model.get_input_embeddings().weight.device
    runtime.phase("capture_audit_teacher_and_inputs")
    with torch.no_grad():
        for start in range(0, len(audit), args.batch):
            ids = torch.cat(audit[start:start + args.batch]).to(dev0)
            logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite pristine teacher logits")
            teacher.extend(logits[i].detach().to("cpu", dtype=torch.bfloat16, copy=True)
                           for i in range(logits.shape[0]))
            del logits
    for handle in handles:
        handle.remove()
    expected_rows = len(audit) * audit[0].shape[1]
    if any(sum(x.shape[0] for x in chunks) != expected_rows for chunks in cached_inputs.values()):
        raise RuntimeError("a sampled module did not receive every audit input")

    # Calculate every sampled local reconstruction effect while the model is
    # still pristine.  One all-tile calculation is shared by samples in a module.
    local = {}
    runtime.phase("local_reconstruction_begin")
    with torch.no_grad():
        for mi, name in enumerate(sorted(needed)):
            w = modules[name].weight.detach()
            scores = reconstruction_scores(cached_inputs[name], w, Q.four_over_six(w), Q.e0m3(w))
            for row in needed[name]:
                index = int(row["flat_tile_in_module"])
                local[row["sample_id"]] = float(scores[index])
            cached_inputs[name].clear()
            if (mi + 1) % 8 == 0:
                runtime.phase(f"local_reconstruction_modules_{mi + 1:03d}")
    del cached_inputs

    installer = P.Installer(args.model, model, modules, cache="cpu",
                            protocol_id="ppl-improvement-extension-v1", campaign_root=str(CR))
    runtime.phase("four_over_six_baseline")
    install_info = installer.install({"name": "p51_four_over_six_baseline", "kind": "four_over_six"})
    t0 = time.time()
    base_ce, base_kl = losses(model, audit, teacher, args.batch)
    if not torch.isfinite(base_ce).all() or not torch.isfinite(base_kl).all():
        raise FloatingPointError("nonfinite P51 baseline")
    report["baseline"] = {"ce_mean": float(base_ce.mean()), "ce_per_sequence": base_ce.tolist(),
                          "kl_mean": float(base_kl.mean()), "kl_per_sequence": base_kl.tolist(),
                          "installed_weight_sha256": install_info["installed_weight_sha256"],
                          "seconds": time.time() - t0}
    if installed_weight_sha256(modules) != install_info["installed_weight_sha256"]:
        raise RuntimeError("P51 baseline installed-weight digest is not reproducible")
    save()

    alt_cache = {}
    for ti, frozen_row in enumerate(sample["tiles"]):
        runtime.phase(f"tile_{ti:03d}")
        name = frozen_row["module"]
        weight = modules[name].weight
        rs, cs = tile_slices(tuple(weight.shape), int(frozen_row["row_tile"]), int(frozen_row["k_tile"]))
        if name not in alt_cache:
            alt_cache.clear()
            alt_cache[name] = Q.e0m3(installer.pristine[name].to(weight.device))
        baseline_tile = weight[rs, cs].detach().clone()
        with torch.no_grad():
            weight[rs, cs] = alt_cache[name][rs, cs]
        started = time.time()
        try:
            ce, kl = losses(model, audit, teacher, args.batch)
        finally:
            with torch.no_grad():
                weight[rs, cs] = baseline_tile
        dce, dkl = ce - base_ce, kl - base_kl
        if not torch.isfinite(dce).all() or not torch.isfinite(dkl).all():
            raise FloatingPointError(f"nonfinite P51 effect: {frozen_row['sample_id']}")
        row = dict(frozen_row)
        row.update(actual_ce=float(dce.mean()), actual_ce_se=stderr(dce),
                   actual_ce_per_sequence=dce.tolist(), actual_kl=float(dkl.mean()),
                   actual_kl_se=stderr(dkl), actual_kl_per_sequence=dkl.tolist(),
                   actual_layer_reconstruction=local[frozen_row["sample_id"]],
                   seconds=time.time() - started)
        report["tiles"].append(row); save()

    runtime.phase("baseline_restoration_check")
    # Re-evaluate the same first batch shape used for the baseline.  A different
    # batch shape can select a different SDPA/GEMM kernel and is not a bit-exact
    # restoration test even when every weight byte has been restored.
    checked = min(args.batch, len(audit))
    check_ce, check_kl = losses(model, audit[:checked], teacher[:checked], checked)
    restored_weight_sha256 = installed_weight_sha256(modules)
    report["baseline_restoration"] = {
        "ce_exact": bool(torch.equal(check_ce, base_ce[:checked])),
        "kl_exact": bool(torch.equal(check_kl, base_kl[:checked])),
        "ce_max_abs_delta": float((check_ce - base_ce[:checked]).abs().max()),
        "kl_max_abs_delta": float((check_kl - base_kl[:checked]).abs().max()),
        "weight_bytes_exact": restored_weight_sha256 == install_info["installed_weight_sha256"],
        "baseline_installed_weight_sha256": install_info["installed_weight_sha256"],
        "restored_installed_weight_sha256": restored_weight_sha256,
        "checked_sequences": checked,
        "baseline_batch_size": args.batch,
        "restoration_batch_size": checked}
    save()
    if not all(report["baseline_restoration"][key]
               for key in ("ce_exact", "kl_exact", "weight_bytes_exact")):
        raise RuntimeError("P51 baseline restoration is not bit-exact")
    installer.remove()
    report.update(status="complete", model_class=type(model).__name__, module_manifest_sha256=mm_sha,
                  loading_info=loading, completed_tiles=len(report["tiles"]))
    save()
    spec = MOD.REGISTRY[args.model]
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": spec["model_id"], "model_revision": spec["revision"],
                   "tokenizer_revision": spec["revision"], "model_class": type(model).__name__,
                   "module_manifest_sha256": mm_sha,
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(attention_backend="sdpa", activation_quantizer="four_over_six_rows"),
        "data": {"calibration_manifest_sha256": stats["calibration_manifest_sha256"],
                 "evaluation_manifest_sha256": None,
                 "token_hashes": {"fidelity_audit": sample["audit_token_sha256"]},
                 "overlap_audit": {"reconstruction_fit_indices": stats["reconstruction_fit_indices"],
                                   "fidelity_audit_indices": sample["audit_indices"],
                                   "disjoint": not bool(set(stats["reconstruction_fit_indices"]) & set(sample["audit_indices"]))}},
        "policies": [{"name": "p51_four_over_six_baseline_plus_single_tile_E0M3",
                      "weight_format": "FourOverSix/E0M3 single N16K64 intervention",
                      "activation_format": "four_over_six_rows", "scale_block": 16,
                      "type_block": [16, 64], "map_path": None, "map_sha256": None,
                      "selected_tiles": 1, "total_tiles": sample["eligible_tiles"],
                      "map_reloaded_for_evaluation": None}],
        "results": {"raw_outputs": [str(report_path)],
                    "summary": {"sampled_tiles": len(report["tiles"]),
                                "baseline_restored_exact": True,
                                "audit_candidate": sample["audit_candidate"]},
                    "uncertainty": {"per_sequence_standard_errors": True,
                                    "selection_weights_in_sample_manifest": True},
                    "attempted_endpoints": ["P51_TILE_FIDELITY_CE", "P51_TILE_FIDELITY_KL",
                                            "P51_TILE_FIDELITY_LAYER_RECONSTRUCTION"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "model": args.model, "tiles": len(report["tiles"]),
                      "report": str(report_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
