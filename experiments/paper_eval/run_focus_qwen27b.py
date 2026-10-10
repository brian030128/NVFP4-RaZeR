"""Part Q (PROTOCOL.md): FOCUS on Qwen3.8-27B from the trained state on HF (edgeai-lab/Qwen3.8-27B-FOCUS-NVFP4 @
b84ff372a7, focus_256/focus.pt), evaluated exactly as part K.

    python run_focus_qwen27b.py provenance   # CPU: the state's windows and hyperparameters vs part K (provenance.json)
    python run_focus_qwen27b.py ppl          # GPU: native PPL, part K's command -> RUN/focus/qwen3.8-27b/ppl.json

The windows check compares the state's 256 window token sha256 (report.json data.records, in order) with the release
fit set flipquant.data.calibration_release builds for qwen3.8-27b: the record's 128 fit windows (math, then code) and the
release 16x64 trainer report's 128 extension windows; its fit_extension.json must equal that report's fit extension.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402
import run_sm120 as S  # noqa: E402

MODEL = "qwen3.8-27b"
REPO, REPO_SHA = "edgeai-lab/Qwen3.8-27B-FOCUS-NVFP4", "b84ff372a7ccab5b8541395db26056c3acd14aff"
WORK = G.RUN / "focus_qwen27b"
OUT = G.RUN / "focus" / MODEL / "ppl.json"
# part K's settings (run_sm120.FOCUS_COMMON, with --data release and the release revision), as train_focus records them
PART_K = {"epochs": "1", "batch": "32", "lr_scale": "0.005", "lr_sub": "0.001", "topk": "1000", "num_sub": "2",
          "init_q": "6.0", "act": "fourover6", "act_scope": "row", "seed": "42", "ppl_datasets": "", "seqlen": "512",
          "data": "release", "revision": S.REV[MODEL], "max_blocks": "0", "zeroshot": "False"}


def provenance():
    h = json.loads((WORK / "report.json").read_text())
    e = json.loads((WORK / "fit_extension.json").read_text())
    rec = json.loads(S.record(MODEL).read_text())
    tr = json.loads(S.trainer_report(MODEL).read_text())
    ours = [x for s in ("math", "code") for x in rec["fit"][s]["token_sha256"]] + \
        [r["token_sha256"] for r in tr["fit_extension"]["records"]]
    a = h["args"]
    out = dict(repo=REPO, revision=REPO_SHA, model=h["model"], source=h["source"], status=h["status"], steps=h["steps"],
               windows=dict(state=len(h["data"]["records"]), release=len(ours), identical_in_order=h["data"]["records"] == ours,
                            extension_records_identical=e["fit_extension"]["records"] == tr["fit_extension"]["records"],
                            extension_settings_identical=all(e["fit_extension"][k] == tr["fit_extension"][k]
                                                             for k in ("seed", "windows", "per_source", "skipped_documents"))),
               hyperparameters={k: dict(state=a.get(k), part_k=v, equal=a.get(k) == v) for k, v in PART_K.items()},
               deviations=dict(micro=a["micro"], device_map=a["device_map"], gpus=h["resources"].get("gpus"),
                               job=h["resources"].get("job_id"), windows_arg=a["windows"],
                               note="micro-batch 1 on 2 x H200 with device_map balanced (part K: micro 8, one GPU); the "
                                    "global batch (32, 8 steps) is the same, only the summation order differs; "
                                    "--windows 256 = all 256 release windows (part K: 0 = all)"),
               kl_top=dict(first=h["log"][0]["kl_top"], last=h["log"][-1]["kl_top"]), parameters=h["parameters"],
               training_seconds=h["training_seconds"])
    out["ok"] = (out["windows"]["identical_in_order"] and out["windows"]["extension_records_identical"]
                 and out["windows"]["extension_settings_identical"] and all(v["equal"] for v in out["hyperparameters"].values())
                 and h["model"] == MODEL and h["source"]["revision"] == S.REV[MODEL] and h["status"] == "complete")
    (WORK / "provenance.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("windows", "deviations", "ok")}, indent=1))
    if not out["ok"]:
        G.stop("part Q: the HF FOCUS state's windows or settings differ from part K's")


def ppl():
    prov = json.loads((WORK / "provenance.json").read_text())
    dl = json.loads((WORK / "download.json").read_text())
    if not (prov["ok"] and dl["sha256_equals_hub_lfs"]):
        G.stop("part Q: provenance or download check not passed")
    q = [*G.NATIVE, "--kernel-set", "auto", "--weight", "focus", "--focus", dl["path"], "--focus-deploy",
         "--act", "fourover6", "--act-scope", "row", "--paper-convention"]
    rc = G.run(f"focus_ppl_{MODEL}_partQ", S.ppl_cmd(MODEL, q, OUT), OUT, S.FQ2)
    if rc not in (None, 0):
        logf = (G.RUN / "logs" / f"focus_ppl_{MODEL}_partQ.log").read_text(errors="replace")
        oom = "OutOfMemoryError" in logf or "CUDA out of memory" in logf
        G.stop(f"part Q: FOCUS PPL {MODEL} rc={rc}" + (" (out of GPU memory in the deploy path; settings unchanged)" if oom else ""))


if __name__ == "__main__":
    {"provenance": provenance, "ppl": ppl}[sys.argv[1]]()
