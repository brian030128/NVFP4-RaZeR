"""Shared, outcome-blind utilities for the N16K64 boundary/corruption campaign."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch

from campaign import mapio


CAMPAIGN_START_UTC = "2026-09-18T08:03:03Z"
SEED_ROOT = 20260918
K_PRIMARY = 3.0
MODELS = ("llama8b", "qwen4b", "mistral7b")
DOMAINS = ("c4", "wiki")
CANDIDATE_B = (8, 6, 4)
PARTITIONS_BY_B = {8: 2, 6: 3, 4: 4}
CORRUPTION_POOLS = 4
CORRUPTION_LEVELS = (0.10, 0.25, 0.50, 0.75, 1.00)

REPO_ROOT = Path(os.environ.get("MIXFP4_WORKSPACE_ROOT", Path.cwd()))
PRIMARY_ROOT = Path(os.environ.get(
    "MIXFP4_PRIMARY_CAMPAIGN",
    REPO_ROOT / "research_runs/mixfp4_n16k64_full_validation_20260911T065444Z",
))
MECHANISM_ROOT = Path(os.environ.get(
    "MIXFP4_MECHANISM_CAMPAIGN",
    REPO_ROOT / "research_runs/mixfp4_mechanism_24h_20260916T205501Z",
))
FOLLOWUP_ROOT = Path(os.environ.get(
    "MIXFP4_FOLLOWUP_CAMPAIGN",
    REPO_ROOT / "research_runs/mixfp4_n16k64_followup_20260917T090438Z",
))
HANDOFF_ROOT = Path(os.environ.get("MIXFP4_BOUNDARY_HANDOFF", REPO_ROOT / (
    "research_handoffs/mixfp4_n16k64_boundary_corruption_20260918_readonly/"
    "mixfp4_n16k64_boundary_corruption_agent_handoff_2026-09-18"
)))
OUTER_ZIP = Path(os.environ.get(
    "MIXFP4_BOUNDARY_HANDOFF_ZIP",
    REPO_ROOT / "mixfp4_n16k64_boundary_corruption_agent_handoff_2026-09-18.zip",
))

MODEL_RUN = {
    "llama8b": ("V30_calib_llama8b_seed0_attempt1", "llama8b"),
    "qwen4b": ("V30_calib_qwen4b_seed0_attempt1", "qwen4b"),
    "mistral7b": ("V61_calib_mistral7b_seed0_attempt2", "mistral7b"),
}
MECHANISM_RUN = {
    "llama8b": "V22_full_llama8b_attempt1",
    "qwen4b": "V21_full_qwen4b_attempt2",
    "mistral7b": "V30_mistral_one_shot_attempt1",
}
REVISIONS = {
    "llama8b": ("meta-llama--Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
    "qwen4b": ("Qwen--Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "mistral7b": ("mistralai--Mistral-7B-v0.3", "caa1feb0e54d415e2df31207e5f4e273e33509b1"),
}
EXPECTED_INPUTS = {
    "llama8b": {
        "map": "0920f55ddc053a5f5a8b0d046d0b74a62d4abafcad5e36051d7682da961e1f2b",
        "moments": "e28c87a08db30cbd799a1886fe8b8fb2c821ed2c7adf8a6ffdab1735dd8095c6",
    },
    "qwen4b": {
        "map": "188bf0e51c372cd831b158e457ede4b53121f0513261f39df8403971b20c01e8",
        "moments": "2315102bf1b8b7f95cd3db62e976ee7cfd4a857e7cd4ca3489f315ee1965cb5e",
    },
    "mistral7b": {
        "map": "0c3d822a18d0480ca2aeff0abd78cf6e4ca3155bb207156a400fde124e1cf13d",
        "moments": "68321bb3fd9ecb8f0050659cdbe62a39c17eb262995d075e64de1630ece27ab1",
    },
}
EXPECTED_PAIRED_ARRAYS = {
    "llama8b_c4": "71ccd13f27dce3361cf73448080db7bdf428d04b3466d8be523859f2de23e5a6",
    "llama8b_wiki": "656f912fd31bc65aef8c319fdc6e501eb8527d4bc8688a06e71f90661e179f45",
    "qwen4b_c4": "1e20cfdc2cd9789a1a0a5a84fabb81feecddc8913884af65e4031ed46f9aa410",
    "qwen4b_wiki": "2d48bab554d9e3174fb429a83d75e2ce06b3202e5bc34afafc5ca0e7952fcd8f",
    "mistral7b_c4": "d8c88b482d764f838fe3b511fd82d498974d6bdf0db944934aee52092ddaae9c",
    "mistral7b_wiki": "6d9f3599cecdb938b949548391a6c791d7d7a621fd696196221b3401288dbbee",
}


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def atomic_json(path: Path | str, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def primary_paths(model: str) -> tuple[Path, Path]:
    run, stem = MODEL_RUN[model]
    root = PRIMARY_ROOT / "runs" / run
    return root / f"maps/{stem}_seed0_n16_k3.mixfp4map", root / "calibration/moments/moments_full.pt"


def module_stratum(name: str) -> tuple[int, str, str]:
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        raise ValueError(f"cannot parse layer from {name}")
    projection = name.rsplit(".", 1)[-1]
    family = "attention" if ".self_attn." in name else "mlp" if ".mlp." in name else None
    if family is None:
        raise ValueError(f"out-of-scope module {name}")
    return int(match.group(1)), projection, family


def mean_se(moment: dict, objective: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(moment["n"])
    total = moment[f"{objective}_sum"].double()
    squares = moment[f"{objective}_sq"].double()
    mean = total / n
    variance = ((squares - total * total / n) / (n - 1)).clamp_min(0)
    return mean, (variance / n).sqrt()


def objective_kappa(mean: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
    positive = se > 0
    out = torch.empty_like(mean)
    out[positive] = -mean[positive] / se[positive]
    zero = ~positive
    out[zero & (mean < 0)] = math.inf
    out[zero & (mean >= 0)] = -math.inf
    return out


def load_model_scores(model: str) -> dict:
    map_path, moments_path = primary_paths(model)
    header, anchor, map_sha = mapio.read_map(map_path, EXPECTED_INPUTS[model]["map"])
    moments = torch.load(moments_path, map_location="cpu", weights_only=True, mmap=True)
    names = list(moments["names"])
    if names != list(anchor):
        raise ValueError(f"{model}: moments/map module order mismatch")
    stats: dict[str, dict[str, torch.Tensor]] = {}
    reconstructed: dict[str, torch.Tensor] = {}
    zero_rows: list[dict] = []
    upper_mismatch = 0
    for name in names:
        st = moments["n16"][name]
        ce_mean, ce_se = mean_se(st, "ce")
        kl_mean, kl_se = mean_se(st, "kl")
        ce_kappa = objective_kappa(ce_mean, ce_se)
        kl_kappa = objective_kappa(kl_mean, kl_se)
        kappa = torch.minimum(ce_kappa, kl_kappa)
        via_kappa = (kappa > K_PRIMARY).reshape(anchor[name].shape)
        via_upper = ((ce_mean + K_PRIMARY * ce_se < 0) & (kl_mean + K_PRIMARY * kl_se < 0)).reshape(anchor[name].shape)
        upper_mismatch += int((via_kappa != via_upper).sum())
        reconstructed[name] = via_kappa
        width = int(anchor[name].shape[1])
        zero = (ce_se == 0) | (kl_se == 0)
        for flat in torch.nonzero(zero, as_tuple=False).flatten().tolist():
            i = int(flat)
            zero_rows.append({
                "module": name,
                "flat_tile_index": i,
                "tile_row": i // width,
                "tile_col": i % width,
                "ce_mean": float(ce_mean[i]),
                "ce_se": float(ce_se[i]),
                "kl_mean": float(kl_mean[i]),
                "kl_se": float(kl_se[i]),
                "kappa_ce": float(ce_kappa[i]),
                "kappa_kl": float(kl_kappa[i]),
            })
        stats[name] = {
            "ce_mean": ce_mean,
            "ce_se": ce_se,
            "kl_mean": kl_mean,
            "kl_se": kl_se,
            "kappa_ce": ce_kappa,
            "kappa_kl": kl_kappa,
            "kappa": kappa,
        }
    mismatch = sum(int((reconstructed[name] != anchor[name]).sum()) for name in names)
    shapes = {name: tuple(int(v) for v in moments["shapes"][name]) for name in names}
    return {
        "model": model,
        "moments": moments,
        "stats": stats,
        "anchor": anchor,
        "reconstructed": reconstructed,
        "header": header,
        "shapes": shapes,
        "map_path": map_path,
        "moments_path": moments_path,
        "map_sha256": map_sha,
        "moments_sha256": EXPECTED_INPUTS[model]["moments"],
        "anchor_mismatches": mismatch,
        "upper_score_mismatches": upper_mismatch,
        "zero_se_tiles": zero_rows,
    }


def empty_like(reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(mask) for name, mask in reference.items()}


def mask_from_lists(reference: dict[str, torch.Tensor], lists: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    out = empty_like(reference)
    for name, indices in lists.items():
        if indices:
            out[name].view(-1)[torch.tensor(indices, dtype=torch.long)] = True
    return out


def tile_lists(mask: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {name: [int(v) for v in torch.nonzero(grid.flatten(), as_tuple=False).flatten().tolist()]
            for name, grid in mask.items() if bool(grid.any())}


def tile_set_sha(lists: dict[str, list[int]]) -> str:
    normalized = {name: sorted(int(v) for v in values) for name, values in sorted(lists.items()) if values}
    return canonical_sha(normalized)


def keyed_digest(label: str, index: int) -> bytes:
    return hashlib.sha256(f"{SEED_ROOT}:{label}:{index}".encode()).digest()


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def stratified_priority(indices: list[int], scores: torch.Tensor, label: str, strata: int = 8) -> list[int]:
    """Nested, score-stratified hash order with balanced prefixes across octiles."""
    ordered = sorted((int(i) for i in indices), key=lambda i: (-float(scores[i]), i))
    buckets: list[list[int]] = [[] for _ in range(strata)]
    n = len(ordered)
    for rank, index in enumerate(ordered):
        bucket = min(strata - 1, (rank * strata) // max(1, n))
        buckets[bucket].append(index)
    for bucket, values in enumerate(buckets):
        values.sort(key=lambda i: keyed_digest(f"{label}:octile{bucket}", i))
    result: list[int] = []
    cursor = [0] * strata
    offset = int.from_bytes(hashlib.sha256(f"{SEED_ROOT}:{label}:rotation".encode()).digest()[:2], "big") % strata
    while len(result) < n:
        progressed = False
        for step in range(strata):
            bucket = (offset + step) % strata
            if cursor[bucket] < len(buckets[bucket]):
                result.append(buckets[bucket][cursor[bucket]])
                cursor[bucket] += 1
                progressed = True
        if not progressed:
            break
    if len(result) != n or len(set(result)) != n:
        raise AssertionError(f"bad stratified priority for {label}")
    return result


def score_summary(lists: dict[str, list[int]], stats: dict[str, dict[str, torch.Tensor]]) -> dict:
    kappas, ce, kl = [], [], []
    for name, values in lists.items():
        if not values:
            continue
        idx = torch.tensor(values, dtype=torch.long)
        kappas.append(stats[name]["kappa"][idx])
        ce.append(stats[name]["ce_mean"][idx])
        kl.append(stats[name]["kl_mean"][idx])
    if not kappas:
        return {"tiles": 0, "mean_kappa": None, "median_kappa": None, "min_kappa": None,
                "max_kappa": None, "sum_ce_mean": 0.0, "sum_kl_mean": 0.0,
                "ce_bottleneck_share": None, "kl_bottleneck_share": None}
    k = torch.cat(kappas)
    c = torch.cat(ce)
    l = torch.cat(kl)
    finite = k[torch.isfinite(k)]
    ce_bottleneck = 0
    total = 0
    for name, values in lists.items():
        if values:
            idx = torch.tensor(values, dtype=torch.long)
            ce_bottleneck += int((stats[name]["kappa_ce"][idx] <= stats[name]["kappa_kl"][idx]).sum())
            total += len(values)
    return {
        "tiles": int(k.numel()),
        "mean_kappa": float(finite.mean()) if finite.numel() else None,
        "median_kappa": float(finite.median()) if finite.numel() else None,
        "min_kappa": float(finite.min()) if finite.numel() else None,
        "max_kappa": float(finite.max()) if finite.numel() else None,
        "nonfinite_kappa": int((~torch.isfinite(k)).sum()),
        "sum_ce_mean": float(c.sum()),
        "sum_kl_mean": float(l.sum()),
        "ce_bottleneck_share": ce_bottleneck / total,
        "kl_bottleneck_share": (total - ce_bottleneck) / total,
    }
