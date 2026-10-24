"""Part AC (PROTOCOL.md): the paper's reruns at the 1,024-window calibration of part AA.

    python run_paper1k.py item2      # tab:latency SM120 rows (the drift anchor first), then the memory probe

Records under RUN/paper1k/; START / END in RUN/queue.log; one GPU job at a time on an idle GPU (run_gpu). A finished
record is not run again; a failed job or check stops the item (STOP). Before every job the GPU's state (power, clocks,
driver, temperature; nothing is changed) is appended to RUN/paper1k/gpu_state.jsonl.
"""
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_calib_size as CS  # noqa: E402
import run_gpu as G  # noqa: E402
import run_sm120 as S  # noqa: E402
import run_zeroshot6 as Z  # noqa: E402

FQ = CS.FQ3                                    # flipquant calib-size
#: calib-size HEAD: 9f19863's code (results only since); evaluation/, flipquant/, models/, kernels/ and scripts/ are
#: byte-identical to paper-sm120-runs 6f5ad27 (part W)
CODE = "d940fbf"
ROOT = G.RUN / "paper1k"
MODELS = ["llama3.2-1b-instruct", "llama3.1-8b-instruct", "mistral-7b", "nemotron-nano-9b-v2", "phi4-14b", "qwen3.8-27b"]
UNITS = {"8x64": "fq8", "16x64": "fq16", "256x64": "fq256"}
ANCHOR = "mistral-7b"                          # the drift anchor, as part W's check: FO6 and FQ16 with the release map
A_COMMIT = "58fe644"                           # part A's latency.json
ANCHOR_CELL_PCT = 3.0                          # every (policy, shape) cell within +-3 % of A's
ANCHOR_RATIO_PP = 1.0                          # the fq-16x64 / fo6 ratio's median over shapes within +-1.0 pp of A's
MEM_TOL_GIB = 0.02
#: item 2 amendment 1 (the coordinator's decision, 2026-10-24): the anchor's cell failure is accepted as a deviation (the
#: paired ratio passed), and an unrecorded ~10 min prefill warm-up runs right before the panel
ANCHOR_ACCEPTED = "the coordinator, 2026-10-24: option 1 (accept; the ratio check passed; -3 % session offset recorded)"
WARMUP = ["--shapes", "4x2048", "--reps", "2400", "--no-graph"]     # Mistral fo6, eager: about 2400 x 0.25 s = 10 min
GPU_QUERY = "name,driver_version,power.draw,power.limit,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,pstate"


def stop(msg):
    G.stop(f"paper1k: {msg}")


def checkout():
    def git(*a):
        return subprocess.run(["git", "-C", str(FQ), *a], capture_output=True, text=True).stdout.strip()
    head, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=no")
    if not head.startswith(CODE) or dirty:
        stop(f"flipquant {FQ} at {head[:7]} (dirty: {bool(dirty)}), not a clean {CODE}")
    return head


def map1k(model, unit):
    return CS.ROOT / model / f"{UNITS[unit]}_1024" / "map.pt"


def verify_maps():
    """Every 1k map's sha256 against its job record (part AA / amendment 6)."""
    out = {}
    for model in MODELS:
        for unit, tag in UNITS.items():
            want = json.loads((CS.ROOT / model / f"{tag}_1024.job.json").read_text())["map_file_sha256"]
            got = CS.file_sha256(map1k(model, unit))
            out[f"{model}/{unit}"] = dict(sha256=got, recorded=want, equal=got == want)
            if got != want:
                stop(f"map {model} {unit}: sha256 {got[:12]} != its job record's {want[:12]}")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "maps_1k.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def policy_args(model, pol, release=False):
    """run_gpu.POLICIES with each "@<unit>" map the 1k map (release: the release / part P map)."""
    out = []
    for a in G.POLICIES[pol]:
        if a.startswith("@"):
            u = a[1:]
            out.append(str(Z.map_path(model, u) if release else map1k(model, u)))
        else:
            out.append(a)
    return out


def gpu_state(job):
    """nvidia-smi's view of the GPU before a job (read only)."""
    q = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"], capture_output=True,
                       text=True).stdout.strip()
    row = dict(zip(GPU_QUERY.split(","), [x.strip() for x in q.split(",")]))
    row["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "gpu_state.jsonl", "a") as f:
        f.write(json.dumps(dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), job=job, **row)) + "\n")


# ---------------------------------------------------------------------------------------------- item 2: latency
def latency_job(model, pol, r, root, libs, release=False):
    out = ROOT / root / model / pol / f"round{r}.json"
    name = f"p1k_{root}_{model}_{pol}_r{r}"
    cmd = [G.py(model), "-m", "evaluation.latency", "prefill", "--model", model, *policy_args(model, pol, release),
           "--shapes", G.SHAPES, "--reps", "7", "--label", pol, "--round", str(r), "--out", out]
    if not G.done(out):
        checkout()
        gpu_state(name)
    rc = G.run(name, cmd, out, FQ)
    if rc is None:
        return
    if rc != 0 or not out.exists():
        stop(f"latency {model} {pol} r{r}: rc={rc}")
    bad = G.check_fq_record(out, pol, libs)
    if bad:
        stop(f"latency {model} {pol} r{r}: {bad}")


def a_latency():
    out = subprocess.run(["git", "-C", str(G.RZ), "show", f"{A_COMMIT}:results/paper_eval/latency/latency.json"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def anchor_check():
    """Step 0: the anchor's medians over rounds against part A's (registered tolerances)."""
    a, m = a_latency(), ANCHOR
    rec = {p: {r: json.loads((ROOT / "latency_anchor" / m / p / f"round{r}.json").read_text()) for r in range(1, 6)}
           for p in ("fo6", "fq-16x64")}
    cells, ratio = {}, {}
    for s in G.SHAPES.split(","):
        for p in rec:
            new = statistics.median(rec[p][r]["graph"][s]["ms"] for r in rec[p])
            cells[f"{p} {s}"] = dict(new=new, a=a["ms"][m][p][s], pct=100 * (new / a["ms"][m][p][s] - 1))
        ratio[s] = statistics.median(100 * (rec["fq-16x64"][r]["graph"][s]["ms"] / rec["fo6"][r]["graph"][s]["ms"] - 1)
                                     for r in range(1, 6))
    new_r, a_r = statistics.median(ratio.values()), a["ratios"][m]["fq-16x64 vs fo6"]["median_over_shapes"]
    worst = max(abs(c["pct"]) for c in cells.values())
    ok = worst <= ANCHOR_CELL_PCT and abs(new_r - a_r) <= ANCHOR_RATIO_PP
    res = dict(model=m, a_commit=A_COMMIT, cells=cells, ratio_per_shape=ratio, ratio_median_over_shapes=new_r,
               a_ratio_median_over_shapes=a_r, worst_cell_pct=worst, cell_tol_pct=ANCHOR_CELL_PCT,
               ratio_tol_pp=ANCHOR_RATIO_PP, passed=ok)
    if not ok:
        res["accepted"] = ANCHOR_ACCEPTED
    (ROOT / "anchor_check.json").write_text(json.dumps(res, indent=1) + "\n")
    G.log(f"paper1k item2 anchor check: worst cell {worst:+.2f} %, ratio {new_r:+.2f} vs A {a_r:+.2f} pp -> "
          f"{'PASS' if ok else 'FAIL, accepted: ' + ANCHOR_ACCEPTED}")


def warmup():
    """Amendment 1: about 10 min of unrecorded prefill (Mistral fo6, 4x2048, eager) right before the panel, so that it
    starts at a steady temperature; its output is not used by any report."""
    out = ROOT / "warmup" / "mistral-7b_fo6_4x2048.json"
    if G.done(out):
        return
    checkout()
    gpu_state("p1k_warmup")
    cmd = [G.py(ANCHOR), "-m", "evaluation.latency", "prefill", "--model", ANCHOR, *policy_args(ANCHOR, "fo6"), *WARMUP,
           "--label", "warmup", "--round", "0", "--out", out]
    rc = G.run("p1k_warmup_mistral-7b_fo6", cmd, out, FQ)
    if rc not in (None, 0):
        stop(f"warm-up: rc={rc}")


def item2_latency():
    checkout()
    verify_maps()
    libs = G.manifests()
    for r in range(1, 6):                      # the drift anchor (release map), part W's order rule
        order = ["fo6", "fq-16x64"]
        random.Random(G.SEED + r).shuffle(order)
        G.log(f"paper1k item2 anchor {ANCHOR} round {r} order {','.join(order)}")
        for pol in order:
            latency_job(ANCHOR, pol, r, "latency_anchor", libs, release=True)
    anchor_check()
    warmup()
    for model in MODELS:
        for r in range(1, 6):
            order = list(G.POLICIES)
            random.Random(G.SEED + r).shuffle(order)
            G.log(f"paper1k item2 {model} round {r} order {','.join(order)} (seed {G.SEED + r})")
            for pol in order:
                latency_job(model, pol, r, "latency", libs)
    G.log("paper1k item2 latency done")


def memory_job(model, pol, root, release=False):
    out = ROOT / root / model / f"{pol}.json"
    name = f"p1k_{root}_{model}_{pol}"
    cmd = [G.py(model), HERE / "mem_probe.py", "--model", model, *policy_args(model, pol, release), "--label", pol,
           "--out", out]
    if not G.done(out):
        checkout()
        gpu_state(name)
    rc = G.run(name, cmd, out, FQ, {"MEM_PROBE_FLIPQUANT": str(FQ)})
    if rc not in (None, 0):
        stop(f"memory {model} {pol}: rc={rc}")


def memory_anchor_check():
    new = json.loads((ROOT / "memory_anchor" / ANCHOR / "fq-16x64.json").read_text())
    old = json.loads((G.RUN / "memory" / ANCHOR / "fq-16x64.json").read_text())
    res = dict(installed_equal=new["quantized_linears"]["installed_bytes"] == old["quantized_linears"]["installed_bytes"],
               model_equal=new["model_bytes"] == old["model_bytes"],
               after_load_diff_gib=(new["allocated_after_load"] - old["allocated_after_load"]) / 2 ** 30,
               peak_diff_gib=(new["peak_prefill_1x2048"] - old["peak_prefill_1x2048"]) / 2 ** 30, tol_gib=MEM_TOL_GIB)
    res["passed"] = (res["installed_equal"] and res["model_equal"] and abs(res["after_load_diff_gib"]) <= MEM_TOL_GIB
                     and abs(res["peak_diff_gib"]) <= MEM_TOL_GIB)
    (ROOT / "memory_anchor_check.json").write_text(json.dumps(res, indent=1) + "\n")
    G.log(f"paper1k item2 memory anchor check: {res}")
    if not res["passed"]:
        stop(f"item2 memory anchor check failed: {res}")


def item2_memory():
    checkout()
    memory_job(ANCHOR, "fq-16x64", "memory_anchor", release=True)     # against part B's record
    memory_anchor_check()
    for model in MODELS:
        for pol in G.POLICIES:
            memory_job(model, pol, "memory")
    G.log("paper1k item2 memory done")


# ---------------------------------------------------------------------------------------------- items 3 on: paper-1k-reruns
FQN = Path("/home/dev/n16k64_campaign/paper1k/wt")      # flipquant paper-1k-reruns
CODE_N = "2739458"                                      # calib-size d940fbf + tmopt_ext --fit-windows / --teacher-host
CODE_DIRS = ("calibration", "evaluation", "flipquant", "models", "kernels", "scripts")
DATA = CS.DATA                                          # fqrel/tmopt_data: the release runs' calibration records
ABL_MODELS = ["nemotron-nano-9b-v2", "qwen3.8-27b"]
ABL = ROOT / "ablation"
TINY = 3                                                # the smoke's random-init model: every layer type kept


def checkout_n():
    """paper-1k-reruns: HEAD descends from CODE_N, the code directories are CODE_N's, no tracked change (the
    branch's later commits are reports only)."""
    def git(*a):
        return subprocess.run(["git", "-C", str(FQN), *a], capture_output=True, text=True)
    head = git("rev-parse", "HEAD").stdout.strip()
    anc = git("merge-base", "--is-ancestor", CODE_N, "HEAD").returncode == 0
    diff = git("diff", "--name-only", CODE_N, "HEAD", "--", *CODE_DIRS).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    if not anc or diff or dirty:
        stop(f"flipquant {FQN} at {head[:7]}: descends from {CODE_N}: {anc}; code changed: {bool(diff)}; "
             f"dirty: {bool(dirty)}")
    return head


def run_n(name, cmd, out, extra_env=None):
    """run_gpu.run on paper-1k-reruns, with its code check and the GPU state row."""
    if not G.done(out):
        checkout_n()
        gpu_state(name)
    return G.run(name, cmd, out, FQN, extra_env)


def must(rc, what):
    if rc not in (None, 0):
        stop(f"{what}: rc={rc}")


def maps_module():
    if str(FQN) not in sys.path:
        sys.path.insert(0, str(FQN))
    from flipquant import maps as M
    return M


def ext_hashes(report):
    return [x["token_sha256"] for x in json.loads(Path(report).read_text())["fit_extension"]["records"]]


def aa_report(model, unit="16x64"):
    """Part AA's (amendment 6's) 1k trainer report of a unit: its fit_extension is the 1k set's 896 windows."""
    return CS.ROOT / model / f"{UNITS[unit]}_1024" / "run" / "report.json"


def tiles_equal(a, b):
    M = maps_module()
    ta, ua, _ = M.load(a)
    tb, ub, _ = M.load(b)
    import torch
    same = ua == ub and set(ta) == set(tb) and all(torch.equal(ta[n], tb[n]) for n in ta)
    return same, dict(modules=len(ta), e0m3_a=sum(int(m.sum()) for m in ta.values()),
                      e0m3_b=sum(int(m.sum()) for m in tb.values()))


# ---------------------------------------------------------------------------------------------- item 3: tab:ablation
def item3_step0():
    """(a) tmopt_ext at --fit-windows 1024 on random-init 3-layer models (Nemotron with the GPU teacher, Qwen3.8-27B with
    --teacher-host), 2 optimizer steps: complete, and the fit extension's token hashes equal part AA's 1k set's;
    (b) one 256-window ablation map rebuilt on this code: G's Nemotron 16x64 one-shot map, tiles bit for bit."""
    sm = ROOT / "tmopt_ext_smoke"
    res = dict(smoke={}, rebuild={})
    aa = {m: [ext_hashes(aa_report(m, u)) for u in UNITS] for m in ABL_MODELS}
    for m in ABL_MODELS:
        if any(h != aa[m][0] for h in aa[m]):
            stop(f"item3 step 0: part AA's three 1k trainer reports of {m} hold different fit extensions")
    for model, host in (("nemotron-nano-9b-v2", False), ("qwen3.8-27b", True)):
        key = S.MKEY[model]
        root = sm / "data_root"
        kdir = root / f"tiny{TINY}" / key
        kdir.mkdir(parents=True, exist_ok=True)
        for d in sorted((DATA / key).iterdir()):           # the release root's development records, linked (read only)
            if d.is_dir() and d.name != "calibration" and not (kdir / d.name).exists():
                (kdir / d.name).symlink_to(d)
        must(run_n(f"p1k_item3_smoke_record_{model}", [G.py(model), "-m", "calibration.tmopt_ext", "record", "--model", model,
                   "--tiny-layers", TINY, "--data-root", root], kdir / "calibration" / "report.json"), "smoke record")
        tiny_map = sm / f"{model}_tiny{TINY}_16x64.pt"
        must(run_n(f"p1k_item3_smoke_remap_{model}", [G.py(model), "-m", "models.tiny", "remap", "--model", model,
                   "--tiny-layers", TINY, "--map", map1k(model, "16x64"), "--out", tiny_map], tiny_map), "smoke remap")
        out = sm / f"{model}_oneshot_1024.pt"
        must(run_n(f"p1k_item3_smoke_oneshot_{model}", [G.py(model), "-m", "calibration.tmopt_ext", "oneshot", "--model",
                   model, "--unit", "16x64", "--tiny-layers", TINY, "--fit-windows", 1024, "--oneshot-steps", 2,
                   "--match", tiny_map, "--data-root", root, "--out", out, *(["--teacher-host"] if host else [])],
                   out.with_suffix(".json")), "smoke oneshot")
        rep = json.loads(out.with_suffix(".json").read_text())
        trainer = json.loads((out.with_suffix(".run") / "run" / "report.json").read_text())
        got = ext_hashes(out.with_suffix(".run") / "run" / "report.json")
        r = dict(status=rep["status"], steps=rep["meta"]["oneshot"]["steps"], fit_windows=rep["meta"].get("fit_windows"),
                 non_release=rep["meta"].get("non_release"), teacher=trainer.get("teacher_storage"),
                 extension_windows=len(got), extension_equal_part_aa=got == aa[model][0],
                 trainer_fit_windows=trainer["args"].get("fit_windows"))
        r["passed"] = (r["status"] == "complete" and r["steps"] == 2 and r["fit_windows"] == 1024 and len(got) == 896
                       and r["extension_equal_part_aa"] and r["trainer_fit_windows"] == 1024
                       and (r["teacher"]["devices"] == ["cpu"]) == host)
        res["smoke"][model] = r
        G.log(f"paper1k item3 step 0 smoke {model}: {'PASS' if r['passed'] else 'FAIL'} ({r})")
        if not r["passed"]:
            (ROOT / "item3_step0.json").write_text(json.dumps(res, indent=1) + "\n")
            stop(f"item3 step 0 smoke {model} failed: {r}")
    model = "nemotron-nano-9b-v2"
    out = ROOT / "ablation_step0" / f"{model}_16x64_oneshot_256.pt"
    must(run_n(f"p1k_item3_step0_rebuild_{model}_16x64_oneshot", [G.py(model), "-m", "calibration.tmopt_ext", "oneshot",
               "--model", model, "--unit", "16x64", "--match", G.REL / model / "flipquant_16x64.pt", "--data-root", DATA,
               "--out", out], out.with_suffix(".json")), "step 0 rebuild")
    ref = G.RUN / "ablation" / model / "16x64" / "oneshot.pt"
    same, info = tiles_equal(out, ref)
    grads_same = CS.file_sha256(out.with_suffix(".run") / "oneshot_grads.pt") == \
        CS.file_sha256(ref.with_suffix(".run") / "oneshot_grads.pt")
    res["rebuild"] = dict(model=model, unit="16x64", map="oneshot", tiles_equal=same, grads_file_equal=grads_same, **info)
    (ROOT / "item3_step0.json").write_text(json.dumps(res, indent=1) + "\n")
    G.log(f"paper1k item3 step 0 rebuild: tiles equal {same}, gradient file equal {grads_same}")
    if not same:
        stop("item3 step 0: the rebuilt 256 one-shot map differs from G's")


def native_ppl(model, map_path, out):
    """run_sm120.native_ppl (G's PPL command)."""
    return [G.py(model), "-m", "evaluation.ppl", "--model", model, *G.NATIVE, "--kernel-set", "auto", "--weight", "mixfp4",
            "--act", "fourover6", "--map", map_path, "--paper-convention", "--out", out]


def item3_panel():
    """G's maps and PPL at 1k: per model and unit the k_l-matched baselines of the 1k map (random 0 / 1 / 2;
    activation-weighted on the 1k windows; the one-shot gradient over the 1k windows), then each map's native PPL."""
    for model in ABL_MODELS:
        for u in UNITS:
            match, d = map1k(model, u), ABL / model / u
            for s in (0, 1, 2):
                out = d / f"random{s}.pt"
                must(run_n(f"p1k_abl_map_{model}_{u}_random{s}", [G.py(model), "-m", "calibration.baseline_maps", "--method",
                           "random", "--seed", s, "--model", model, "--match", match, "--out", out],
                           out.with_suffix(".json")), f"ablation map {model} {u} random{s}")
            out = d / "act.pt"
            must(run_n(f"p1k_abl_map_{model}_{u}_act", [G.py(model), "-m", "calibration.baseline_maps", "--method", "act",
                       "--model", model, "--match", match, "--calib-record", S.record(model), "--calib-extension",
                       aa_report(model, u), "--batch", 1 if model == "qwen3.8-27b" else 8, "--out", out],
                       out.with_suffix(".json")), f"ablation map {model} {u} act")
            out = d / "oneshot.pt"
            must(run_n(f"p1k_abl_map_{model}_{u}_oneshot", [G.py(model), "-m", "calibration.tmopt_ext", "oneshot", "--model",
                       model, "--unit", u, "--match", match, "--data-root", DATA, "--fit-windows", 1024,
                       *(["--teacher-host"] if model == "qwen3.8-27b" else []), "--out", out], out.with_suffix(".json")),
                 f"ablation map {model} {u} oneshot")
            bad = item3_map_checks(model, u)
            if bad:
                stop(f"item3 {model} {u}: {bad}")
            for m in S.ABLATION_MAPS:
                out = d / f"ppl_{m}.json"
                must(run_n(f"p1k_abl_ppl_{model}_{u}_{m}", native_ppl(model, d / f"{m}.pt", out), out),
                     f"ablation ppl {model} {u} {m}")
    G.log("paper1k item3 panel done")


def item3_map_checks(model, u):
    """Every baseline map has the 1k map's per-layer E0M3 counts; the act map saw the 1k windows (128 + part AA's 896);
    the one-shot run's fit extension is part AA's."""
    M = maps_module()
    d = ABL / model / u
    ref, _, _ = M.load(map1k(model, u))
    bad = []
    for m in S.ABLATION_MAPS:
        t, _, _ = M.load(d / f"{m}.pt")
        if set(t) != set(ref) or any(int(t[n].sum()) != int(ref[n].sum()) or t[n].shape != ref[n].shape for n in ref):
            bad.append(f"{m}: per-layer E0M3 counts or shapes differ from the 1k map")
    act = json.loads((d / "act.json").read_text())
    want = ext_hashes(aa_report(model, u))
    if act["data"]["fit_windows"] != 1024 or act["data_release"]["token_sha256"][128:] != want:
        bad.append("act: not the 1k windows")
    if ext_hashes(d / "oneshot.run" / "run" / "report.json") != want:
        bad.append("oneshot: the fit extension is not part AA's")
    return bad


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "item2":
        item2_latency()
        item2_memory()
        G.log("DONE paper1k item2")
    elif what == "item3":
        item3_step0()
        item3_panel()
        G.log("DONE paper1k item3")
    else:
        raise SystemExit(f"unknown item {what}")
