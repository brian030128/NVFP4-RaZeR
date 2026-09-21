"""P20/P50 calibration-only activation and reconstruction selector statistics.

Inputs are captured from the pristine BF16 network.  The first 8 math and first 8 code
sequences are cached only in RAM for the reconstruction fit and discarded after the
score artifact is durably written.  All 64+64 sequences contribute to the input RMS
and diagonal-Hessian statistics.  No WikiText/C4 evaluation outcome is consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch

from campaign import calibrate as C
from campaign import data as D
from campaign import models as MOD
from campaign import quant as Q
from campaign import runtime
from campaign import tiles as T


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
PARENT = {"llama8b": "V30_calib_llama8b_seed0_attempt1",
          "qwen4b": "V30_calib_qwen4b_seed0_attempt1",
          "mistral7b": "V61_calib_mistral7b_seed0_attempt2",
          "qwen27b": "V30_calib_qwen27b_seed0_attempt1",
          "phi4": "V61_calib_phi4_seed0_attempt1",
          "olmo2_13b": "V61_calib_olmo2_13b_seed0_attempt2"}
HELDOUT = ("granite8b", "falcon3_10b")


def load(path):
    return json.loads(Path(path).read_text())


def latest_complete(prefix):
    accepted = []
    for run in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(run.name.rsplit("_attempt", 1)[1])
            launch, status, validation = (load(run / name) for name in
                                           ("launch_record.json", "job_status.json",
                                            "run_record_validation.json"))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if (launch.get("status") == status.get("status") == "complete"
                and validation.get("valid")):
            accepted.append((attempt, run))
    if not accepted:
        raise FileNotFoundError(prefix)
    return max(accepted)[1]


def reference_calibration(model):
    if model in PARENT:
        return PR / "runs" / PARENT[model]
    if model in HELDOUT:
        return latest_complete(f"P70_calib_{model}_seed0")
    raise ValueError(f"selector statistics are not authorized for {model}")


def reconstruction_scores(x_chunks, w, qf, qe, type_block=T.N16):
    """Exact single-tile output-MSE change via cross term + within-K-block Gram.

    Matrix products use float32 (the same precision used by the fake-quant diagnostic),
    while independent chunk contributions are accumulated in float64.  Returned scores
    are normalized by token rows times out_features.
    """
    bm, bk = type_block
    o, k = w.shape
    if o % bm or k % bk:
        raise ValueError("weight is not tile divisible")
    dev = w.device
    r = qf.float() - w.float()
    delta = qe.float() - qf.float()
    go, gk = o // bm, k // bk
    total = torch.zeros(go * gk, dtype=torch.float64)
    rows = 0
    for raw in x_chunks:
        x = raw.to(dev, dtype=torch.float32)
        rows += x.shape[0]
        error = x @ r.T
        cross = error.T @ x
        cross_tiles = (2.0 * cross * delta).reshape(go, bm, gk, bk).sum((1, 3)).reshape(-1)
        xg = x.reshape(x.shape[0], gk, bk)
        gram = torch.einsum("ngi,ngj->gij", xg, xg)
        dg = delta.reshape(o, gk, bk)
        quad = torch.einsum("ogi,gij,ogj->og", dg, gram, dg)
        quad_tiles = quad.reshape(go, bm, gk).sum(1).reshape(-1)
        total += (cross_tiles + quad_tiles).double().cpu()
        del x, error, cross, xg, gram, dg, quad, cross_tiles, quad_tiles
    if rows == 0:
        raise ValueError("no reconstruction inputs")
    return total / (rows * o)


def tile_score(values, tb=T.N16):
    return T.tile_sum(values, tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=tuple(PARENT) + HELDOUT)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--freeze-sha256", required=True)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--max-memory-gib", type=float, default=None,
                    help="Per-visible-GPU load cap for models that require homogeneous multi-GPU sharding.")
    ap.add_argument("--reconstruction-chunk-rows", type=int, default=512)
    a = ap.parse_args()
    if runtime.sha256_file(a.freeze) != a.freeze_sha256 or a.freeze_sha256 != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.set_num_threads(8)
    t0 = time.time()
    model_key = a.model
    parent = reference_calibration(model_key)
    parent_report = load(parent / "calibration/calibration_report.json")
    tok = MOD.load_tokenizer(model_key)
    seqs, domains, cal_meta = C.calibration_inputs(model_key, "seed0", tok)
    token_hashes = [D.sha(x) for x in seqs]
    if token_hashes != parent_report["sequence_token_sha256"]:
        raise RuntimeError("canonical seed0 tokens differ from reference calibration")
    fit_indices = list(range(8)) + list(range(64, 72))
    if [domains[i] for i in fit_indices] != ["math"] * 8 + ["code"] * 8:
        raise RuntimeError("reconstruction-fit domain split differs from frozen spec")
    out = runtime.out_dir("selector_stats")
    rep = {"schema_version": "1.0", "status": "running", "model": model_key,
           "protocol_sha256": PROTOCOL, "selector_spec_sha256": runtime.sha256_file(CR / "provenance/P50_P51_SELECTOR_FIDELITY_SPEC.json"),
           "p20_spec_sha256": runtime.sha256_file(CR / "provenance/P20_CONTROL_DERIVATION_SPEC.json"),
           "calibration_manifest_sha256": D.manifest_sha256(cal_meta), "sequence_token_sha256": token_hashes,
           "activation_sequences": 128, "reconstruction_fit_indices": fit_indices,
           "reconstruction_fit_token_sha256": [token_hashes[i] for i in fit_indices],
           "arithmetic": {"input_second_moment": "BF16 inputs squared/reduced and accumulated as CUDA float64",
                          "reconstruction": "float32 exact cross-term plus within-K64 Gram quadratic; chunk contributions accumulated float64",
                          "normalization": "fit token rows times module out_features"}}
    save = lambda: runtime.atomic_json(out / "selector_stats_report.json", rep)
    save()
    runtime.phase("load_model")
    ngpu = torch.cuda.device_count()
    if ngpu < 1:
        raise RuntimeError("selector statistics require at least one visible GPU")
    max_memory = ({i: f"{a.max_memory_gib}GiB" for i in range(ngpu)}
                  if ngpu > 1 and a.max_memory_gib else None)
    model, loading = MOD.load_model(
        model_key, attn=a.attn,
        device_map=("cuda" if ngpu == 1 else "balanced"),
        max_memory=max_memory)
    modules = MOD.scope(model, model_key)
    mm_sha, mm_rows = MOD.module_manifest(modules)
    if mm_sha != parent_report["module_manifest_sha256"]:
        # The digest canonically covers scope, order, shapes, dtypes, biases, and weights.
        raise RuntimeError("module/weight manifest differs from reference calibration")
    names = list(modules)
    shapes = {n: tuple(modules[n].weight.shape) for n in names}
    bad = [n for n in names if shapes[n][0] % 16 or shapes[n][1] % 64]
    if bad:
        raise RuntimeError(f"non-N16-divisible scoped modules: {bad[:4]}")
    dev0 = model.get_input_embeddings().weight.device
    sums = {n: torch.zeros(shapes[n][1], dtype=torch.float64, device=modules[n].weight.device) for n in names}
    counts = {n: 0 for n in names}
    cached = {n: [] for n in names}
    state = {"index": -1}

    def hook(name):
        def capture(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            sums[name].add_(x.double().square().sum(0))
            counts[name] += int(x.shape[0])
            if state["index"] in fit_indices:
                cached[name].append(x.to("cpu", dtype=torch.bfloat16, copy=True))
        return capture

    handles = [modules[n].register_forward_pre_hook(hook(n)) for n in names]
    runtime.phase("capture_inputs_begin")
    with torch.no_grad():
        for i, ids in enumerate(seqs):
            state["index"] = i
            logits = model(input_ids=ids.to(dev0), use_cache=False).logits
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"nonfinite BF16 logits at calibration sequence {i}")
            del logits
            if (i + 1) % 16 == 0:
                runtime.phase(f"capture_inputs_{i + 1:03d}")
                rep["captured_sequences"] = i + 1; save()
    for h in handles:
        h.remove()
    if any(counts[n] != 128 * 512 for n in names) or any(len(cached[n]) != 16 for n in names):
        raise RuntimeError("a scoped Linear did not receive every frozen calibration input")
    cached_bytes = sum(x.numel() * x.element_size() for xs in cached.values() for x in xs)
    rep["cached_fit_input_bytes_ram_only"] = cached_bytes
    rep["input_rows_per_module"] = 128 * 512
    save()

    scores = {"activation_weighted_error": OrderedDict(),
              "diagonal_hessian_gptq_proxy": OrderedDict(),
              "layer_output_reconstruction": OrderedDict()}
    rms_out = OrderedDict()
    damping = {}
    current_layer = None
    runtime.phase("derive_selector_scores_begin")
    with torch.no_grad():
        for ni, n in enumerate(names):
            layer = n.split("layers.", 1)[1].split(".", 1)[0] if "layers." in n else "other"
            if layer != current_layer:
                runtime.phase(f"derive_selector_scores_layer_{layer}")
                current_layer = layer
            module = modules[n]
            w = module.weight.detach()
            qf = Q.four_over_six(w)
            qe = Q.e0m3(w)
            mean2 = sums[n] / counts[n]
            rms = mean2.sqrt()
            damp = 0.01 * mean2.mean()
            aw = ((w.float() - qe.float()).abs() - (w.float() - qf.float()).abs()) * rms.float()[None, :]
            hs = (((w.float() - qe.float()).square() - (w.float() - qf.float()).square())
                  * (mean2 + damp).float()[None, :])
            rec_chunks = []
            for x in cached[n]:
                rec_chunks.extend(x[s:s + a.reconstruction_chunk_rows] for s in range(0, x.shape[0], a.reconstruction_chunk_rows))
            scores["activation_weighted_error"][n] = tile_score(aw).double().cpu()
            scores["diagonal_hessian_gptq_proxy"][n] = tile_score(hs).double().cpu()
            scores["layer_output_reconstruction"][n] = reconstruction_scores(rec_chunks, w, qf, qe)
            rms_out[n] = rms.cpu()
            damping[n] = float(damp)
            for key in scores:
                if not torch.isfinite(scores[key][n]).all():
                    raise FloatingPointError(f"nonfinite {key} score in {n}")
            cached[n].clear()
            del w, qf, qe, mean2, rms, aw, hs, rec_chunks
            if (ni + 1) % 16 == 0:
                rep["derived_modules"] = ni + 1; save()
    artifact = {"schema_version": "1.0", "model": model_key, "protocol_sha256": PROTOCOL,
                "names": names, "shapes": shapes, "module_manifest_sha256": mm_sha,
                "calibration_manifest_sha256": rep["calibration_manifest_sha256"],
                "sequence_token_sha256": token_hashes, "reconstruction_fit_indices": fit_indices,
                "input_rms": rms_out, "input_rows": counts, "hessian_damping": damping,
                "scores": scores, "score_direction": "ascending", "type_block": [16, 64],
                "arithmetic": rep["arithmetic"]}
    artifact_path = out / "selector_statistics.pt"
    torch.save(artifact, artifact_path)
    # Reload enough metadata and every tensor before accepting the artifact.
    check = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if check["names"] != names or check["protocol_sha256"] != PROTOCOL:
        raise RuntimeError("selector statistic round-trip failed")
    if any(not torch.isfinite(v).all() for group in check["scores"].values() for v in group.values()):
        raise RuntimeError("selector statistic round-trip produced nonfinite tensors")
    rep.update(status="complete", module_manifest_sha256=mm_sha, modules=len(names), model_class=type(model).__name__,
               loading_info=loading, statistic_artifact={"path": str(artifact_path), "sha256": runtime.sha256_file(artifact_path),
                                                         "bytes": artifact_path.stat().st_size},
               wall_seconds=time.time() - t0, cached_inputs_persisted=False)
    save()
    launch = load(runtime.run_dir / "launch_record.json")
    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": MOD.REGISTRY[model_key]["model_id"], "model_revision": MOD.REGISTRY[model_key]["revision"],
                   "tokenizer_revision": MOD.REGISTRY[model_key]["revision"], "model_class": type(model).__name__,
                   "module_manifest_sha256": mm_sha, "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(attention_backend=a.attn),
        "data": {"calibration_manifest_sha256": rep["calibration_manifest_sha256"], "evaluation_manifest_sha256": None,
                 "token_hashes": {"calibration": token_hashes}, "overlap_audit": None}, "policies": [],
        "results": {"raw_outputs": [str(artifact_path), str(out / "selector_stats_report.json")],
                    "summary": {"modules": len(names), "input_rows_per_module": 128 * 512,
                                "reconstruction_fit_sequences": 16, "cached_input_bytes": cached_bytes},
                    "uncertainty": {}, "attempted_endpoints": (["P20_ACTIVATION_WEIGHTED", "P50_SELECTOR_STATISTICS"]
                        if model_key in PARENT else ["P70_HELDOUT_SELECTOR_STATISTICS"]),
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "model": model_key, "artifact": str(artifact_path),
                      "sha256": rep["statistic_artifact"]["sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
