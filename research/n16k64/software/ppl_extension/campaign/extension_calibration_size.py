"""Incremental P31/P32 nested calibration-size scoring and exact-map derivation.

Only newly appended calibration documents are sent through the GPU scorer.  Their float64
sufficient statistics are added to a hash-verified prefix state, which is mathematically
identical to rescoring the unchanged prefix.  P32 refuses to run unless an accepted P31
attempt exists; campaign-level G5 authorization remains the launcher's caller's responsibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from campaign import calibrate as C
from campaign import data as D
from campaign import mapio as MIO
from campaign import models as MOD
from campaign import runtime
from campaign import tiles as T


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
PROTOCOL_ID = "ppl-improvement-extension-v1"
PARENT = {
    "llama8b": "V30_calib_llama8b_seed0_attempt1",
    "qwen4b": "V30_calib_qwen4b_seed0_attempt1",
    "mistral7b": "V61_calib_mistral7b_seed0_attempt2",
}
STATE_FIELDS = ("ce_sum", "ce_sq", "kl_sum", "kl_sq", "cross")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return runtime.sha256_file(path)


def latest_complete(prefix):
    accepted = []
    for run in (CR / "runs").glob(prefix + "_attempt*"):
        try:
            attempt = int(run.name.rsplit("_attempt", 1)[1])
            launch = load(run / "launch_record.json")
            status = load(run / "job_status.json")
            validation = load(run / "run_record_validation.json")
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if launch.get("status") == status.get("status") == "complete" and validation.get("valid"):
            accepted.append((attempt, run))
    if not accepted:
        raise FileNotFoundError(prefix)
    return max(accepted)[1]


def add_moment_states(prefix, addition):
    """Pure float64 sufficient-statistic addition with no variance reconstruction."""
    if set(prefix) != set(addition) or any(k not in prefix for k in ("n", *STATE_FIELDS)):
        raise ValueError("moment state fields differ")
    out = {"n": int(prefix["n"]) + int(addition["n"])}
    for key in STATE_FIELDS:
        a, b = prefix[key].double(), addition[key].double()
        if a.shape != b.shape:
            raise ValueError(f"moment shape differs for {key}")
        out[key] = a + b
    return out


def states_from_moments(moments):
    return {name: {key: (value.detach().cpu() if torch.is_tensor(value) else value)
                   for key, value in moment.state().items()}
            for name, moment in moments.items()}


def assert_p31_p32_model(model):
    """Validate the explicit frozen extension set, not the legacy parent panel label."""
    if model not in PARENT:
        raise ValueError("P31/P32 are development-only")
    return True


def parent_seed0_manifest(model):
    """Load the exact hash-locked seed0 manifest from the frozen parent run."""
    assert_p31_p32_model(model)
    calibration = PR / "runs" / PARENT[model] / "calibration"
    manifest = load(calibration / "calibration_manifest.json")
    report = load(calibration / "calibration_report.json")
    if report.get("status") != "complete":
        raise RuntimeError(f"parent seed0 calibration is not complete for {model}")
    got = D.manifest_sha256(manifest)
    if got != report["calibration_manifest_sha256"]:
        raise RuntimeError(f"parent seed0 manifest digest mismatch for {model}")
    return manifest


def combined_manifest(prefix_meta, addition_meta, stage, prefix_sha):
    """Canonical domain-major manifest: prefix then addition within math and code."""
    out = {"schema_version": "1.0", "stage": stage, "prefix_manifest_sha256": prefix_sha,
           "domains": ["math", "code"], "sequence_length": 512}
    for domain in ("math", "code"):
        pd, ad = prefix_meta[domain], addition_meta[domain]
        out[domain] = {
            "documents": list(pd["documents"]) + list(ad["documents"]),
            "token_sha256": list(pd["token_sha256"]) + list(ad["token_sha256"]),
            "prefix_count": len(pd["documents"]), "addition_count": len(ad["documents"]),
            "addition_draw": ad.get("draw"),
            "rule": "hash-verified prefix followed by frozen keyed nested addition",
        }
    return out


def seed0_and_exclusions(model, tok):
    # The frozen extension specification defines its P31/P32 development set
    # explicitly as Llama, Qwen and Mistral.  Mistral retains the parent
    # campaign's legacy ``confirmatory`` registry label, which must not
    # override that locked extension membership.
    seed0 = parent_seed0_manifest(model)
    _, rebuilt = D.crops_from_manifest(tok, seed0)
    # Rebuild every crop from its document hash and offset to verify tokenizer
    # identity, while retaining the exact parent manifest representation and
    # digest (Mistral records its rule inside each domain rather than at top level).
    for domain in ("math", "code"):
        for field in ("repo", "revision", "path", "documents", "token_sha256"):
            if rebuilt[domain][field] != seed0[domain][field]:
                raise RuntimeError(f"parent seed0 {domain} {field} differs for {model}")
    used = {d["document_sha256"] for dom in ("math", "code") for d in seed0[dom]["documents"]}
    draw_meta = {}
    for index in range(1, 5):
        _, meta = D.keyed_draw(tok, f"draw{index}", set(used))
        draw_meta[f"draw{index}"] = meta
        used |= {d["document_sha256"] for dom in ("math", "code") for d in meta[dom]["documents"]}
    return seed0, used, draw_meta


def prefix_inputs(model, tok, stage):
    seed0, used, draws = seed0_and_exclusions(model, tok)
    seed0_sha = D.manifest_sha256(seed0)
    if stage == "p31":
        addition_batches, addition_meta = D.keyed_draw(tok, "size128_v1", set(used), count=64)
        prefix_run = PR / "runs" / PARENT[model]
        prefix_report = load(prefix_run / "calibration/calibration_report.json")
        prefix_moments = prefix_run / "calibration/moments/moments_full.pt"
        prefix_map_manifest = prefix_run / "calibration/map_manifest.json"
        prefix_meta = seed0
        if prefix_report["calibration_manifest_sha256"] != seed0_sha:
            raise RuntimeError("parent seed0 manifest differs from reconstructed prefix")
        expected_moment_sha = prefix_report["moment_files"]["moments_full.pt"]
        if sha(prefix_moments) != expected_moment_sha:
            raise RuntimeError("parent prefix moment digest mismatch")
    elif stage == "p32":
        size128_batches, size128_meta = D.keyed_draw(tok, "size128_v1", set(used), count=64)
        used |= {d["document_sha256"] for dom in ("math", "code") for d in size128_meta[dom]["documents"]}
        addition_batches, addition_meta = D.keyed_draw(tok, "size256_v1", set(used), count=128)
        prefix_run = latest_complete(f"P31_calib_{model}")
        prefix_report = load(prefix_run / "calibration_size/calibration_size_report.json")
        prefix_moments = prefix_run / "calibration_size/moments/moments_combined.pt"
        prefix_map_manifest = prefix_run / "calibration_size/map_manifest.json"
        prefix_meta = combined_manifest(seed0, size128_meta, "p31", seed0_sha)
        if prefix_report["calibration_manifest_sha256"] != D.manifest_sha256(prefix_meta):
            raise RuntimeError("accepted P31 manifest differs from reconstructed prefix")
        expected_moment_sha = prefix_report["moment_files"]["moments_combined.pt"]
        if sha(prefix_moments) != expected_moment_sha:
            raise RuntimeError("P31 prefix moment digest mismatch")
    else:
        raise ValueError(stage)
    seqs = addition_batches["math"] + addition_batches["code"]
    domains = ["math"] * len(addition_batches["math"]) + ["code"] * len(addition_batches["code"])
    combined = combined_manifest(prefix_meta, addition_meta, stage, D.manifest_sha256(prefix_meta))
    all_docs = [d["document_sha256"] for dom in ("math", "code") for d in combined[dom]["documents"]]
    if len(all_docs) != len(set(all_docs)):
        raise RuntimeError("nested calibration manifest has overlapping documents")
    return {
        "seqs": seqs, "domains": domains, "addition_meta": addition_meta,
        "combined_meta": combined, "prefix_run": prefix_run, "prefix_report": prefix_report,
        "prefix_moments": prefix_moments, "prefix_map_manifest": prefix_map_manifest,
        "draw_manifest_sha256": {name: D.manifest_sha256(meta) for name, meta in draws.items()},
        "excluded_document_count_before_addition": len(used),
    }


def assert_prefix_maps(blob, manifest_path, names, shapes):
    rows = {row["policy"]: row for row in load(manifest_path)}
    checks = {}
    for resolution, tb in (("n8", T.N8), ("n16", T.N16)):
        policy = f"{resolution}_k2"
        row = rows[policy]
        _, masks, digest = MIO.read_map(row["path"], row["sha256"])
        exact = True
        for name in names:
            elected = T.elect(T.Moments.from_state(blob[resolution][name]), 2, "ce_kl")
            expected = masks[name].reshape(-1)
            exact = exact and torch.equal(elected.cpu(), expected)
        checks[policy] = {"map_sha256": digest, "exact": exact,
                          "selected_tiles": sum(int(masks[name].sum()) for name in names),
                          "type_block": list(tb)}
        if not exact:
            raise RuntimeError(f"prefix moments do not reconstruct {policy} exactly")
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=tuple(PARENT))
    ap.add_argument("--stage", required=True, choices=("p31", "p32"))
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--freeze-sha256", required=True)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--teacher", choices=("ram", "disk"), default="ram")
    ap.add_argument("--alt-on-gpu", action="store_true")
    ap.add_argument("--moments-device", default=None)
    args = ap.parse_args()
    if sha(args.freeze) != args.freeze_sha256 or args.freeze_sha256 != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.set_num_threads(8)
    started = time.time()
    out = runtime.out_dir("calibration_size")
    moments_dir = runtime.out_dir("calibration_size", "moments")
    report = {
        "schema_version": "1.0", "status": "running", "stage": args.stage,
        "model": args.model, "protocol_sha256": PROTOCOL,
        "incremental_spec_sha256": sha(CR / "provenance/P31_P32_INCREMENTAL_SPEC.json"),
        "incremental_equivalence": "float64 addition of n/sum/sumsq/cross sufficient statistics; prefix not rescored",
    }
    save = lambda: runtime.atomic_json(out / "calibration_size_report.json", report)
    save()

    tok = MOD.load_tokenizer(args.model)
    inputs = prefix_inputs(args.model, tok, args.stage)
    seqs, domains = inputs["seqs"], inputs["domains"]
    prefix_blob = torch.load(inputs["prefix_moments"], map_location="cpu", weights_only=False)
    names = prefix_blob["names"]
    shapes = {name: tuple(prefix_blob["shapes"][name]) for name in names}
    prefix_checks = assert_prefix_maps(prefix_blob, inputs["prefix_map_manifest"], names, shapes)
    combined_meta = inputs["combined_meta"]
    runtime.atomic_json(out / "calibration_manifest.json", combined_meta)
    runtime.atomic_json(out / "addition_manifest.json", inputs["addition_meta"])
    combined_sha = D.manifest_sha256(combined_meta)
    addition_hashes = [D.sha(x) for x in seqs]
    combined_hashes = combined_meta["math"]["token_sha256"] + combined_meta["code"]["token_sha256"]
    report.update(
        prefix_run_id=inputs["prefix_run"].name, prefix_moments_path=str(inputs["prefix_moments"]),
        prefix_moments_sha256=sha(inputs["prefix_moments"]), prefix_map_assertions=prefix_checks,
        calibration_manifest_sha256=combined_sha,
        addition_manifest_sha256=D.manifest_sha256(inputs["addition_meta"]),
        addition_sequences=len(seqs), combined_sequences=len(combined_hashes),
        addition_sequence_token_sha256=addition_hashes, sequence_token_sha256=combined_hashes,
        addition_domains=domains, draw_manifest_sha256=inputs["draw_manifest_sha256"],
        excluded_document_count_before_addition=inputs["excluded_document_count_before_addition"],
    )
    save()

    runtime.phase("load_model")
    model, loading = MOD.load_model(args.model, attn=args.attn, device_map="cuda")
    modules = MOD.scope(model, args.model)
    mm_sha, mm_entries = MOD.module_manifest(modules)
    if mm_sha != inputs["prefix_report"]["module_manifest_sha256"] or list(modules) != names:
        raise RuntimeError("loaded module/weight manifest differs from prefix")
    if {name: tuple(modules[name].weight.shape) for name in names} != shapes:
        raise RuntimeError("loaded shapes differ from prefix moments")
    runtime.atomic_json(out / "module_manifest.json", mm_entries)
    tok_sha, tok_files = MOD.tokenizer_manifest(args.model)
    report.update(module_manifest_sha256=mm_sha, tokenizer_manifest_sha256=tok_sha,
                  tokenizer_files=tok_files, model_class=type(model).__name__, loading_info=loading,
                  attn_implementation=model.config._attn_implementation)
    save()
    device0 = model.get_input_embeddings().weight.device

    runtime.phase("teacher_addition")
    teacher = []
    teacher_dir = runtime.out_dir("calibration_size", "teacher_tmp") if args.teacher == "disk" else None
    bf16_nll = []
    with torch.no_grad():
        for index, ids in enumerate(seqs):
            logits = model(input_ids=ids.to(device0), use_cache=False).logits
            lp = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
            if not torch.isfinite(lp).all():
                raise FloatingPointError(f"nonfinite teacher output at addition sequence {index}")
            bf16_nll.append(float(F.nll_loss(lp, ids[:, 1:].reshape(-1).to(lp.device))))
            if teacher_dir is None:
                teacher.append(logits[:, :-1].detach().cpu())
            else:
                torch.save(logits[:, :-1].detach().cpu(), teacher_dir / f"{index:03d}.pt")
            del logits, lp
            if (index + 1) % 16 == 0:
                report["teacher_sequences_complete"] = index + 1
                save()
    report["bf16_addition_nll"] = bf16_nll
    save()

    runtime.phase("candidate_weights")
    alt, _ = C.build_candidates(modules, args.alt_on_gpu)
    report["candidate_weight_sha256"] = MOD.module_manifest(modules)[0]
    save()
    provider = ((lambda index: teacher[index]) if teacher_dir is None else
                (lambda index: torch.load(teacher_dir / f"{index:03d}.pt", weights_only=True)))
    scored = C.run_scoring(model, modules, seqs, domains, provider, alt, raw="sample",
                           sample_fraction=0.01, subset_moments=False,
                           progress=lambda index, elapsed: (report.update(score_sequences_complete=index + 1,
                                                                         score_seconds_so_far=elapsed), save()),
                           moments_device=args.moments_device)
    if teacher_dir is not None:
        for path in teacher_dir.glob("*.pt"):
            path.unlink()
        teacher_dir.rmdir()
    teacher.clear()
    report.update(score_seconds=scored["score_seconds"], fit_losses=scored["fit_losses"],
                  score_stream_sha256=scored["score_stream_sha256"],
                  score_stream_sha256_all=hashlib.sha256("".join(scored["score_stream_sha256"][n]
                                                                   for n in names).encode()).hexdigest())
    save()

    runtime.phase("combine_and_elect")
    addition_states = {"names": names, "shapes": shapes,
                       "n8": states_from_moments(scored["full8"]),
                       "n16": states_from_moments(scored["full16"])}
    combined_states = {"names": names, "shapes": shapes, "n8": {}, "n16": {}}
    for resolution in ("n8", "n16"):
        for name in names:
            combined_states[resolution][name] = add_moment_states(prefix_blob[resolution][name],
                                                                   addition_states[resolution][name])
            expected_n = len(combined_hashes)
            if combined_states[resolution][name]["n"] != expected_n:
                raise RuntimeError(f"combined moment count differs in {resolution}/{name}")
    torch.save(addition_states, moments_dir / "moments_addition.pt")
    torch.save(combined_states, moments_dir / "moments_combined.pt")
    samples, raw = scored["samples"], scored["raw_sample"]
    torch.save({"names": names, "shapes": shapes,
                "sample_parents": {n: samples[n].parents for n in names},
                "sample_children": {n: samples[n].children for n in names},
                "scores": raw, "sequence_token_sha256": addition_hashes,
                "rule": "frozen keyed N16-parent sample; addition sequences only"},
               moments_dir / "raw_scores_addition_sample.pt")
    report["moment_files"] = {path.name: sha(path) for path in sorted(moments_dir.iterdir())}
    save()

    maps_dir = runtime.out_dir("maps")
    source_manifest = load(runtime.run_dir / "launch_record.json")["source_manifest_sha256"]
    model_ident = {"model_id": MOD.REGISTRY[args.model]["model_id"],
                   "revision": MOD.REGISTRY[args.model]["revision"],
                   "tokenizer_revision": MOD.REGISTRY[args.model]["revision"],
                   "model_class": type(model).__name__}
    entries = []
    for resolution, tb in (("n8", T.N8), ("n16", T.N16)):
        masks = {name: T.elect(T.Moments.from_state(combined_states[resolution][name]), 2, "ce_kl")
                 .reshape(shapes[name][0] // tb[0], shapes[name][1] // tb[1]).cpu()
                 for name in names}
        policy = {"name": f"{resolution}_k2", "rule": "ce_kl", "k": 2,
                  "resolution": resolution, "calibration_size_stage": args.stage,
                  "combined_sequences": len(combined_hashes)}
        header = MIO.build_header(protocol_id=PROTOCOL_ID, policy=policy, model=model_ident,
                                  type_block=tb, masks=masks,
                                  weight_shapes={n: list(shapes[n]) for n in names},
                                  source_manifest_sha256=source_manifest,
                                  calibration_manifest_sha256=combined_sha)
        path = maps_dir / f"{args.model}_{args.stage}_{resolution}_k2.mixfp4map"
        digest, stored = MIO.write_map(path, header, masks, provenance={
            "run_id": os.environ.get("CAMPAIGN_RUN_ID"), "protocol_freeze_sha256": PROTOCOL,
            "incremental_spec_sha256": report["incremental_spec_sha256"],
            "prefix_moments_sha256": report["prefix_moments_sha256"],
            "combined_moments_sha256": report["moment_files"]["moments_combined.pt"],
            "score_stream_sha256_all": report["score_stream_sha256_all"],
        })
        entries.append({"policy": policy["name"], "type_block": list(tb), "path": stored,
                        "sha256": digest, "selected_tiles": header["totals"]["selected_tiles"],
                        "selected_weights": header["totals"]["selected_weights"],
                        "total_tiles": header["totals"]["total_tiles"],
                        "per_module_selected": {m["name"]: m["selected"] for m in header["modules"]}})
    runtime.atomic_json(out / "map_manifest.json", entries)
    report.update(status="complete", maps=entries, wall_seconds=time.time() - started,
                  counts={row["policy"]: row["selected_tiles"] for row in entries})
    save()

    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": PROTOCOL_ID, "protocol_freeze_sha256": PROTOCOL,
        "source": {"model_id": MOD.REGISTRY[args.model]["model_id"],
                   "model_revision": MOD.REGISTRY[args.model]["revision"],
                   "tokenizer_revision": MOD.REGISTRY[args.model]["revision"],
                   "model_class": type(model).__name__, "module_manifest_sha256": mm_sha,
                   "source_manifest_sha256": source_manifest},
        "environment": runtime.environment(attention_backend=args.attn,
                                             activation_quantizer="four_over_six_rows+STE"),
        "data": {"calibration_manifest_sha256": combined_sha, "evaluation_manifest_sha256": None,
                 "token_hashes": {"calibration": combined_hashes, "addition": addition_hashes},
                 "overlap_audit": {"passed": True, "unique_documents": len(set(
                     d["document_sha256"] for dom in ("math", "code") for d in combined_meta[dom]["documents"]))}},
        "policies": [{"name": row["policy"], "weight_format": "FourOverSix/E0M3 tile mix",
                      "activation_format": "four_over_six_rows", "scale_block": 16,
                      "type_block": row["type_block"], "map_path": row["path"],
                      "map_sha256": row["sha256"], "selected_tiles": row["selected_tiles"],
                      "total_tiles": row["total_tiles"], "map_reloaded_for_evaluation": None}
                     for row in entries],
        "results": {"raw_outputs": [str(out / "calibration_size_report.json"),
                                     str(out / "calibration_manifest.json"),
                                     str(out / "addition_manifest.json"),
                                     str(out / "map_manifest.json")] +
                                    [str(path) for path in sorted(moments_dir.iterdir())] +
                                    [row["path"] for row in entries],
                    "summary": {"stage": args.stage, "addition_sequences": len(seqs),
                                "combined_sequences": len(combined_hashes), "counts": report["counts"]},
                    "uncertainty": {"stored": "complete float64 sufficient statistics"},
                    "attempted_endpoints": [args.stage.upper() + "_CALIBRATION", "n8_k2", "n16_k2"],
                    "missing_endpoints": []}, "logs": [], "failures": []})
    print(json.dumps({"status": "complete", "stage": args.stage, "model": args.model,
                      "counts": report["counts"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
