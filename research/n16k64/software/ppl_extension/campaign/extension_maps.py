"""CPU-only, predeclared map derivations for P20, P40 and P41.

All selectors operate on hash-verified parent sufficient statistics or on an explicitly
named activation-statistics artifact.  The output maps are new immutable MIXFP4MAP/1
objects under the current attempt; parent maps and moments are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from campaign import mapio as MIO
from campaign import models as MOD
from campaign import runtime
from campaign import tiles as T


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
PROTOCOL_ID = "ppl-improvement-extension-v1"
PARENT = {
    "llama8b": {"seed0": "V30_calib_llama8b_seed0_attempt1",
                 "draw1": "V30_calib_llama8b_draw1_attempt1", "draw2": "V30_calib_llama8b_draw2_attempt1",
                 "draw3": "V30_calib_llama8b_draw3_attempt1", "draw4": "V30_calib_llama8b_draw4_attempt1"},
    "qwen4b": {"seed0": "V30_calib_qwen4b_seed0_attempt1",
                "draw1": "V30_calib_qwen4b_draw1_attempt1", "draw2": "V30_calib_qwen4b_draw2_attempt1",
                "draw3": "V30_calib_qwen4b_draw3_attempt1", "draw4": "V30_calib_qwen4b_draw4_attempt2"},
    "mistral7b": {"seed0": "V61_calib_mistral7b_seed0_attempt2",
                   "draw1": "V61_calib_mistral7b_draw1_attempt1", "draw2": "V61_calib_mistral7b_draw2_attempt1",
                   "draw3": "V61_calib_mistral7b_draw3_attempt1", "draw4": "V61_calib_mistral7b_draw4_attempt1"},
}
P20_COUNTS = {"llama8b": 50119, "qwen4b": 32532, "mistral7b": 30611}


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return runtime.sha256_file(path)


def parent_run(model: str, draw="seed0") -> Path:
    return PR / "runs" / PARENT[model][draw]


def locked_moment_sha(model: str, draw: str) -> str:
    spec = load(CR / "provenance/P40_P41_DERIVATION_SPEC.json")
    return spec["parent_moment_inputs"][model][draw][1]


def moments(model: str, draw="seed0") -> tuple[dict, Path, dict]:
    run = parent_run(model, draw)
    path = run / "calibration/moments/moments_full.pt"
    expected = locked_moment_sha(model, draw)
    if sha(path) != expected:
        raise RuntimeError(f"moment digest mismatch: {model}/{draw}")
    rep = load(run / "calibration/calibration_report.json")
    if rep["moment_files"]["moments_full.pt"] != expected or rep["status"] != "complete":
        raise RuntimeError(f"parent calibration report mismatch: {model}/{draw}")
    return torch.load(path, map_location="cpu", weights_only=False), path, rep


def moment_objects(blob: dict, resolution: str) -> OrderedDict[str, T.Moments]:
    return OrderedDict((n, T.Moments.from_state(blob[resolution][n])) for n in blob["names"])


def parent_masks(model: str, policy: str):
    manifest = {x["policy"]: x for x in load(parent_run(model) / "calibration/map_manifest.json")}
    row = manifest[policy]
    header, masks, digest = MIO.read_map(row["path"], expected_sha256=row["sha256"])
    return header, masks, digest


def flat_tie_order(names: list[str], values: dict[str, torch.Tensor]):
    ordered = sorted(names)
    flat = torch.cat([values[n].reshape(-1).double() for n in ordered])
    return ordered, flat


def select_global(names, scores, count, descending=False):
    ordered, flat = flat_tie_order(names, scores)
    if not torch.isfinite(flat).all():
        raise ValueError("nonfinite selector score")
    order = torch.argsort(-flat if descending else flat, stable=True)[:int(count)]
    chosen = torch.zeros(flat.numel(), dtype=torch.bool)
    chosen[order] = True
    out, off = {}, 0
    for n in ordered:
        z = scores[n].numel()
        out[n] = chosen[off:off + z].reshape(scores[n].shape)
        off += z
    return OrderedDict((n, out[n]) for n in names)


def select_per_module(names, scores, counts, descending=False):
    out = OrderedDict()
    for n in names:
        v = scores[n].reshape(-1).double()
        if not torch.isfinite(v).all():
            raise ValueError(f"nonfinite selector score: {n}")
        order = torch.argsort(-v if descending else v, stable=True)[:int(counts[n])]
        m = torch.zeros(v.numel(), dtype=torch.bool); m[order] = True
        out[n] = m.reshape(scores[n].shape)
    return out


def random_scores(names, shapes, scope, master_seed):
    out = {}
    if scope == "global":
        labels = {n: "__GLOBAL__" for n in names}
        digest = hashlib.sha256(b"P20:global:__GLOBAL__").digest()
        words = np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32).tolist()
        rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([master_seed, *map(int, words)])))
        for n in sorted(names):
            out[n] = torch.from_numpy(rng.random(int(np.prod(shapes[n])), dtype=np.float64)).reshape(shapes[n])
    else:
        labels = {n: n for n in names}
        for n in names:
            digest = hashlib.sha256(("P20:per_module:" + n).encode()).digest()
            words = np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32).tolist()
            rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([master_seed, *map(int, words)])))
            out[n] = torch.from_numpy(rng.random(int(np.prod(shapes[n])), dtype=np.float64)).reshape(shapes[n])
    return out, labels


class Writer:
    def __init__(self, model, names, shapes, calibration_sha, model_class, spec_sha):
        self.model, self.names, self.shapes = model, names, shapes
        self.calibration_sha, self.model_class, self.spec_sha = calibration_sha, model_class, spec_sha
        self.out = runtime.out_dir("derived_maps")
        self.entries = []
        self.launch = load(runtime.run_dir / "launch_record.json")

    def write(self, name, masks, policy, rule, tb=T.N16,
              calibration_manifest_sha256=None):
        grid = OrderedDict((n, masks[n].reshape(self.shapes[n][0] // tb[0], self.shapes[n][1] // tb[1]).bool()) for n in self.names)
        ident = {"model_id": MOD.REGISTRY[self.model]["model_id"], "revision": MOD.REGISTRY[self.model]["revision"],
                 "tokenizer_revision": MOD.REGISTRY[self.model]["revision"], "model_class": self.model_class}
        p = dict(policy, name=name)
        calibration_sha = calibration_manifest_sha256 or self.calibration_sha
        header = MIO.build_header(protocol_id=PROTOCOL_ID, policy=p, model=ident, type_block=tb, masks=grid,
                                  weight_shapes=self.shapes, source_manifest_sha256=self.launch["source_manifest_sha256"],
                                  calibration_manifest_sha256=calibration_sha)
        digest, path = MIO.write_map(self.out / f"{self.model}_{name}.mixfp4map", header, grid,
                                     provenance={"run_id": os.environ.get("CAMPAIGN_RUN_ID"), "rule": rule,
                                                 "protocol_freeze_sha256": PROTOCOL, "derivation_spec_sha256": self.spec_sha})
        e = {"policy": name, "type_block": list(tb), "path": path, "sha256": digest,
             "selected_tiles": header["totals"]["selected_tiles"], "total_tiles": header["totals"]["total_tiles"],
             "selected_weights": header["totals"]["selected_weights"],
             "selected_fraction": header["totals"]["selected_fraction"],
             "calibration_manifest_sha256": calibration_sha,
             "per_module_selected": {m["name"]: m["selected"] for m in header["modules"]}}
        self.entries.append(e)
        return e

    def finish(self, report):
        runtime.atomic_json(self.out / "map_manifest.json", self.entries)
        report["maps"] = self.entries
        report["status"] = "complete"
        runtime.atomic_json(self.out / "derivation_report.json", report)
        runtime.atomic_json(runtime.run_dir / "job_result.json", {
            "protocol_id": PROTOCOL_ID, "protocol_freeze_sha256": PROTOCOL,
            "source": {"model_id": MOD.REGISTRY[self.model]["model_id"], "model_revision": MOD.REGISTRY[self.model]["revision"],
                       "tokenizer_revision": MOD.REGISTRY[self.model]["revision"], "model_class": self.model_class,
                       "module_manifest_sha256": report.get("module_manifest_sha256", "from_parent_moments"),
                       "source_manifest_sha256": self.launch["source_manifest_sha256"]},
            "environment": runtime.environment(),
            "data": {"calibration_manifest_sha256": self.calibration_sha, "evaluation_manifest_sha256": None,
                     "token_hashes": {}, "overlap_audit": None},
            "policies": [{"name": e["policy"], "weight_format": "FourOverSix/E0M3 tile mix",
                          "activation_format": "four_over_six_rows", "scale_block": 16,
                          "type_block": e["type_block"], "map_path": e["path"], "map_sha256": e["sha256"],
                          "selected_tiles": e["selected_tiles"], "total_tiles": e["total_tiles"],
                          "map_reloaded_for_evaluation": None} for e in self.entries],
            "results": {"raw_outputs": [str(self.out / "derivation_report.json"), str(self.out / "map_manifest.json")],
                        "summary": {e["policy"]: e["selected_tiles"] for e in self.entries}, "uncertainty": {},
                        "attempted_endpoints": [os.environ.get("CAMPAIGN_MATRIX_ID", "map_derivation")], "missing_endpoints": []},
            "logs": [], "failures": []})


def p20(model: str, activation_stats: str | None):
    spec_path = CR / "provenance/P20_CONTROL_DERIVATION_SPEC.json"
    spec = load(spec_path); spec_sha = sha(spec_path)
    blob, mom_path, rep = moments(model)
    names, shapes = blob["names"], {n: tuple(blob["shapes"][n]) for n in blob["names"]}
    n16 = moment_objects(blob, "n16"); n8 = moment_objects(blob, "n8")
    u16 = {n: T.upper_bound(n16[n], 2, "ce_kl").reshape(shapes[n][0] // 16, shapes[n][1] // 64) for n in names}
    u8 = {n: T.upper_bound(n8[n], 2, "ce_kl").reshape(shapes[n][0] // 8, shapes[n][1] // 64) for n in names}
    h16, natural, _ = parent_masks(model, "n16_k2")
    if sum(int(x.sum()) for x in natural.values()) != P20_COUNTS[model]:
        raise RuntimeError("frozen P20 target differs from parent exact map")
    assert all(torch.equal(natural[n], u16[n] < 0) for n in names)
    counts16 = {n: int(natural[n].sum()) for n in names}
    target = P20_COUNTS[model]
    wpath = parent_run(model) / "calibration/moments/weight_tile_stats.pt"
    expected_w = spec["parent_moment_inputs"][model]["weight_tile_stats_sha256"]
    if sha(wpath) != expected_w:
        raise RuntimeError("weight tile statistic digest mismatch")
    ws = torch.load(wpath, map_location="cpu", weights_only=False)
    score_sets = {"weight_mse": ws["mse_gain16"], "magnitude": ws["l2_16"], "change_norm": ws["dnorm16"]}
    if activation_stats:
        apath = Path(activation_stats)
        a = torch.load(apath, map_location="cpu", weights_only=False)
        if a.get("model") != model or a.get("protocol_sha256") != PROTOCOL or a.get("names") != names:
            raise RuntimeError("activation statistic identity mismatch")
        score_sets["activation_weighted"] = a["scores"]["activation_weighted_error"]
    writer = Writer(model, names, shapes, rep["calibration_manifest_sha256"], rep["model_class"], spec_sha)
    report = {"schema_version": "1.0", "mode": "P20", "model": model, "status": "running",
              "protocol_sha256": PROTOCOL, "derivation_spec_sha256": spec_sha,
              "parent_moments": str(mom_path), "parent_moments_sha256": sha(mom_path),
              "weight_stats_sha256": expected_w, "target_selected_tiles": target,
              "target_selected_weights": target * 1024, "scopes": {},
              "module_manifest_sha256": rep["module_manifest_sha256"]}
    for scope in ("global", "per_module"):
        target_counts = counts16 if scope == "per_module" else None
        # N8 shape match: exactly two N8 tiles for each N16 tile.
        n8_masks = (select_global(names, u8, 2 * target) if scope == "global"
                    else select_per_module(names, u8, {n: 2 * counts16[n] for n in names}))
        writer.write(f"p20_{scope}_n8_u2_matched", n8_masks,
                     {"selector": "n8_u2_matched", "scope": scope, "matched_to": "natural_n16_k2"},
                     "ascending N8 U2 at exactly twice the N16 tile count", tb=T.N8)
        for seed in spec["controls"]["random_s0_to_s4"]["master_seeds"]:
            rs, labels = random_scores(names, {n: u16[n].shape for n in names}, scope, int(seed))
            masks = (select_global(names, rs, target) if scope == "global"
                     else select_per_module(names, rs, target_counts))
            writer.write(f"p20_{scope}_random_{seed}", masks,
                         {"selector": "random", "scope": scope, "master_seed": int(seed), "matched_to": "natural_n16_k2"},
                         "PCG64DXSM named independent tile priorities; ascending")
        for selector, scores in score_sets.items():
            descending = selector in ("weight_mse", "magnitude", "change_norm")
            masks = (select_global(names, scores, target, descending) if scope == "global"
                     else select_per_module(names, scores, target_counts, descending))
            writer.write(f"p20_{scope}_{selector}", masks,
                         {"selector": selector, "scope": scope, "matched_to": "natural_n16_k2"},
                         ("descending " if descending else "ascending ") + selector + " score")
        report["scopes"][scope] = {"target_tiles": target, "target_weights": target * 1024,
                                    "expected_residual_weights": 0}
    writer.finish(report)
    print(json.dumps({"mode": "P20", "model": model, "maps": len(writer.entries)}, sort_keys=True), flush=True)


def p40(model: str):
    spec_path = CR / "provenance/P40_P41_DERIVATION_SPEC.json"; spec_sha = sha(spec_path)
    blob, mom_path, rep = moments(model)
    names, shapes = blob["names"], {n: tuple(blob["shapes"][n]) for n in blob["names"]}
    objs = moment_objects(blob, "n16")
    writer = Writer(model, names, shapes, rep["calibration_manifest_sha256"], rep["model_class"], spec_sha)
    assertions = {}
    for k in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        scores = {n: T.upper_bound(objs[n], k, "ce_kl").reshape(shapes[n][0] // 16, shapes[n][1] // 64) for n in names}
        masks = OrderedDict((n, scores[n] < 0) for n in names)
        label = str(k).replace(".", "p")
        writer.write(f"p40_n16_k{label}_seed0", masks, {"selector": "first_order_ce_kl", "k": k, "draw": "seed0"},
                     f"U_{k}=max(mean_CE+{k}SE_CE,mean_KL+{k}SE_KL)<0")
        if k in (2.0, 3.0):
            _, old, old_sha = parent_masks(model, f"n16_k{int(k)}")
            exact = all(torch.equal(masks[n], old[n]) for n in names)
            assertions[f"k{int(k)}"] = {"mask_exact": exact, "parent_map_sha256": old_sha}
            if not exact:
                raise RuntimeError(f"P40 {model} k{k} mask differs from parent")
    report = {"schema_version": "1.0", "mode": "P40", "model": model, "status": "running",
              "protocol_sha256": PROTOCOL, "derivation_spec_sha256": spec_sha,
              "parent_moments": str(mom_path), "parent_moments_sha256": sha(mom_path),
              "map_assertions": assertions, "module_manifest_sha256": rep["module_manifest_sha256"]}
    writer.finish(report)
    print(json.dumps({"mode": "P40", "model": model, "maps": len(writer.entries), "assertions": assertions}, sort_keys=True), flush=True)


def _stack(names, values_by_draw):
    return {n: torch.stack([d[n].reshape(-1).double() for d in values_by_draw]) for n in names}


def _mask_jaccard(a, b):
    inter = sum(int((a[n] & b[n]).sum()) for n in a)
    union = sum(int((a[n] | b[n]).sum()) for n in a)
    return inter / union if union else 1.0


def _fixed_consensus(names, votes, median, mean, count):
    ordered = sorted(names)
    vf = torch.cat([votes[n].reshape(-1).long() for n in ordered])
    med = torch.cat([median[n].reshape(-1).double() for n in ordered])
    avg = torch.cat([mean[n].reshape(-1).double() for n in ordered])
    if not torch.isfinite(med).all() or not torch.isfinite(avg).all():
        raise ValueError("nonfinite consensus tie score")
    idx = torch.arange(vf.numel())
    idx = idx[torch.argsort(avg[idx], stable=True)]
    idx = idx[torch.argsort(med[idx], stable=True)]
    idx = idx[torch.argsort(-vf[idx], stable=True)][:int(count)]
    chosen = torch.zeros(vf.numel(), dtype=torch.bool); chosen[idx] = True
    out, off = {}, 0
    for n in ordered:
        z = votes[n].numel(); out[n] = chosen[off:off + z].reshape(votes[n].shape); off += z
    return OrderedDict((n, out[n]) for n in names)


def _natural_and_fixed(name, names, scores, target, consensus=None):
    if consensus is None:
        natural = OrderedDict((n, scores[n] < 0) for n in names)
        fixed = select_global(names, scores, target)
    else:
        votes, med, avg, threshold = consensus
        natural = OrderedDict((n, votes[n] >= threshold) for n in names)
        fixed = _fixed_consensus(names, votes, med, avg, target)
    return natural, fixed


def _pooled_minus(names, pooled, omitted_blob, k):
    out = {}
    for n in names:
        total = pooled[n]
        old = omitted_blob["n16"][n]
        st = {"n": total.n - int(old["n"])}
        for key in ("ce_sum", "ce_sq", "kl_sum", "kl_sq", "cross"):
            st[key] = getattr(total, key) - old[key].double()
        out[n] = T.upper_bound(T.Moments.from_state(st), k, "ce_kl")
    return out


def p41(model: str, k: float):
    spec_path = CR / "provenance/P40_P41_DERIVATION_SPEC.json"; spec_sha = sha(spec_path)
    resolution_path = CR / "provenance/P41_SEED0_LODO_RESOLUTION.json"
    if not resolution_path.exists():
        raise RuntimeError("P41 seed0 leave-one-draw-out ambiguity must be resolved before derivation")
    resolution_sha = sha(resolution_path)
    names = shapes = rep0 = None
    per_draw = []
    pooled = None
    moment_inputs = []
    calibration_inputs = []
    for di, draw in enumerate(("seed0", "draw1", "draw2", "draw3", "draw4")):
        blob, path, rep = moments(model, draw)
        if names is None:
            names = blob["names"]
            shapes = {n: tuple(blob["shapes"][n]) for n in names}
            rep0 = rep
            pooled = OrderedDict((n, T.Moments.from_state(blob["n16"][n])) for n in names)
        else:
            if blob["names"] != names or any(tuple(blob["shapes"][n]) != shapes[n] for n in names):
                raise RuntimeError("draw module geometry mismatch")
            for n in names:
                pooled[n].add(T.Moments.from_state(blob["n16"][n]))
        objs = moment_objects(blob, "n16")
        per_draw.append({n: T.upper_bound(objs[n], k, "ce_kl").reshape(shapes[n][0] // 16, shapes[n][1] // 64) for n in names})
        moment_inputs.append({"draw": draw, "path": str(path), "sha256": sha(path), "n": next(iter(objs.values())).n})
        manifest_path = parent_run(model, draw) / "calibration/calibration_manifest.json"
        calibration_inputs.append({
            "draw": draw,
            "canonical_manifest_sha256": rep["calibration_manifest_sha256"],
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": sha(manifest_path),
        })
        del blob, objs
    stack = _stack(names, per_draw)
    mean = {n: stack[n].mean(0).reshape(per_draw[0][n].shape) for n in names}
    median = {n: stack[n].median(0).values.reshape(per_draw[0][n].shape) for n in names}
    votes = {n: (stack[n] < 0).sum(0).reshape(per_draw[0][n].shape) for n in names}
    pooled_score = {n: T.upper_bound(pooled[n], k, "ce_kl").reshape(per_draw[0][n].shape) for n in names}
    aggregates = OrderedDict([
        ("seed0", {"scores": per_draw[0], "consensus": None}),
        ("pooled_per_sequence_moments", {"scores": pooled_score, "consensus": None}),
        ("draw_mean", {"scores": mean, "consensus": None}),
        ("draw_median", {"scores": median, "consensus": None}),
        ("consensus_3_of_5", {"scores": median, "consensus": (votes, median, mean, 3)}),
    ])
    _, seed0_k2, seed0_sha = parent_masks(model, "n16_k2")
    target = sum(int(x.sum()) for x in seed0_k2.values())
    derived_out = runtime.out_dir("derived_maps")
    manifest_set_path = derived_out / "calibration_manifest_set.json"
    runtime.atomic_json(manifest_set_path, {
        "schema_version": "1.0",
        "kind": "ordered_five_draw_calibration_manifest_set",
        "model": model,
        "draws": calibration_inputs,
    })
    manifest_set_sha = sha(manifest_set_path)
    writer = Writer(model, names, shapes, manifest_set_sha, rep0["model_class"], spec_sha)
    full_masks, stability = {}, {}
    klabel = str(float(k)).replace(".", "p")
    for agg, obj in aggregates.items():
        natural, fixed = _natural_and_fixed(agg, names, obj["scores"], target, obj["consensus"])
        full_masks[(agg, "natural")] = natural
        full_masks[(agg, "fixed")] = fixed
        calibration_sha = (rep0["calibration_manifest_sha256"]
                           if agg == "seed0" else manifest_set_sha)
        writer.write(f"p41_{agg}_natural_k{klabel}", natural,
                     {"selector": "first_order_ce_kl", "aggregation": agg, "form": "natural_density", "k": k},
                     f"{agg}; natural election at frozen global k={k}",
                     calibration_manifest_sha256=calibration_sha)
        writer.write(f"p41_{agg}_fixed_k{klabel}", fixed,
                     {"selector": "first_order_ce_kl", "aggregation": agg,
                      "form": "fixed_budget_matched_to_seed0_k2", "target_tiles": target, "k": k},
                     f"{agg}; fixed global target {target} at frozen global k={k}",
                     calibration_manifest_sha256=calibration_sha)
    # Recompute each aggregation after omitting each draw.  The pre-result resolution
    # specifies that seed0-only uses the earliest retained draw in frozen draw order.
    draw_names = ["seed0", "draw1", "draw2", "draw3", "draw4"]
    for agg in aggregates:
        stability[agg] = {}
        for form in ("natural", "fixed"):
            vals = []
            for omit in range(5):
                keep = [i for i in range(5) if i != omit]
                substack = {n: stack[n][keep] for n in names}
                submean = {n: substack[n].mean(0).reshape(per_draw[0][n].shape) for n in names}
                submedian = {n: substack[n].median(0).values.reshape(per_draw[0][n].shape) for n in names}
                if agg == "seed0":
                    first = keep[0]
                    score, con = per_draw[first], None
                elif agg == "draw_mean":
                    score, con = submean, None
                elif agg == "draw_median":
                    score, con = submedian, None
                elif agg == "consensus_3_of_5":
                    subvotes = {n: (substack[n] < 0).sum(0).reshape(per_draw[0][n].shape) for n in names}
                    score, con = submedian, (subvotes, submedian, submean, 3)
                else:
                    omitted_blob, _, _ = moments(model, draw_names[omit])
                    score = {n: z.reshape(per_draw[0][n].shape) for n, z in _pooled_minus(names, pooled, omitted_blob, k).items()}
                    con = None
                    del omitted_blob
                nat, fix = _natural_and_fixed(agg, names, score, target, con)
                chosen = nat if form == "natural" else fix
                vals.append({"omitted_draw": draw_names[omit], "jaccard": _mask_jaccard(full_masks[(agg, form)], chosen),
                             "selected_tiles": sum(int(x.sum()) for x in chosen.values())})
            stability[agg][form] = {"leave_one_draw_out": vals,
                                    "median_jaccard": float(np.median([x["jaccard"] for x in vals]))}
    score_path = runtime.out_dir("derived_maps") / "aggregation_scores.pt"
    torch.save({"schema_version": "1.0", "model": model, "k": k, "names": names, "shapes": shapes,
                "draws": draw_names, "per_draw_u": per_draw, "pooled_u": pooled_score,
                "draw_mean_u": mean, "draw_median_u": median, "votes": votes,
                "protocol_sha256": PROTOCOL, "derivation_spec_sha256": spec_sha}, score_path)
    report = {"schema_version": "1.0", "mode": "P41", "model": model, "status": "running",
              "protocol_sha256": PROTOCOL, "derivation_spec_sha256": spec_sha,
              "seed0_lodo_resolution_sha256": resolution_sha, "global_k": k,
              "moment_inputs": moment_inputs, "calibration_inputs": calibration_inputs,
              "calibration_manifest_set": {"path": str(manifest_set_path),
                                             "sha256": manifest_set_sha},
              "seed0_calibration_manifest_sha256": rep0["calibration_manifest_sha256"],
              "target_seed0_k2_tiles": target,
              "target_seed0_k2_map_sha256": seed0_sha, "stability": stability,
              "score_artifact": {"path": str(score_path), "sha256": sha(score_path)},
              "module_manifest_sha256": rep0["module_manifest_sha256"]}
    writer.finish(report)
    print(json.dumps({"mode": "P41", "model": model, "k": k, "maps": len(writer.entries)}, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("p20", "p40", "p41"))
    ap.add_argument("--model", required=True, choices=tuple(PARENT))
    ap.add_argument("--activation-stats")
    ap.add_argument("--global-k", type=float)
    a = ap.parse_args()
    if sha(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    if a.mode == "p20":
        p20(a.model, a.activation_stats)
    elif a.mode == "p40":
        p40(a.model)
    else:
        if a.global_k not in (0.5, 1.0, 1.5, 2.0, 2.5):
            raise SystemExit("P41 requires the frozen P40 global k")
        p41(a.model, a.global_k)


if __name__ == "__main__":
    main()
