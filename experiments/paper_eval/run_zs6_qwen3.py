"""Part X amendment 2 "X2" (PROTOCOL.md): part X's zero-shot panel for Qwen3-1.7B and Qwen3-8B, run between part Y's
native and IF4 / MixFP4 rows (part Y amendment 1).

    python run_zs6_qwen3.py preflight              # CPU: the inputs against the registered records
    python run_zs6_qwen3.py panel --part-y-pid=N   # waits for part Y's last native row, holds the GPU, runs the 20 rows

Part X's functions (run_zeroshot6) do the work: the commands, run_gpu.run, the registered checks and the records under
RUN/zeroshot6/panel/<model>/. X2 changes only two things:
- **The Qwen3 entries of verify_inputs:** the FOCUS states and the GPTQ‡ codes files, against the hashes registered in
  the amendment.
- **The hold:** once part Y's last native row has started (or part Y's driver has exited), this process holds a CUDA
  context until it exits.
  - Part Y's runner waits for an idle GPU before every job, so its next job waits until X2 is done.
  - This process's own idle check ignores itself.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_gpu as G  # noqa: E402
import run_zeroshot6 as Z  # noqa: E402

MODELS = ["qwen3-1.7b", "qwen3-8b"]
X2 = Z.ZS / "x2"
#: registered in PROTOCOL.md (part X amendment 2): the FOCUS upload's record (results/paper_eval/focus_hf/stage.json)
FOCUS_SHA = {"qwen3-1.7b": "5fa877acad86e7b9f64a39936717f8032cab0696d0e657b0525010b4f06a4945",
             "qwen3-8b": "e4689e64c66fe50afa6022485604efc01c5bf21e369ab54dbc4f7775f4a1b7ae"}
#: registered: the codes files' sha256 at the HF staging of 2026-10-15, whose code digests equal the main-table records
CODES_FILE_SHA = {"qwen3-1.7b": "bb50de2383f7690adcf8438681a3fc86e482418ae4de10876cd38e074e995342",
                  "qwen3-8b": "335a69b142c726a98b6879e6a47c8d7de4e295fd9efe889e7cab9a995606d269"}
TRIGGER = "START gsm8k6_qwen3.8-27b_fq-256x64"          # part Y's last native row

_verify_inputs = Z.verify_inputs


def verify_inputs(model, row):
    """part X's verify_inputs, with the registered Qwen3 FOCUS-state and GPTQ‡-codes hashes."""
    if model in MODELS and row == "focus":
        got = Z.sha256_file(Z.focus_state(model))
        return [] if got == FOCUS_SHA[model] else [f"FOCUS state sha256 {got[:12]} != {FOCUS_SHA[model][:12]}"]
    if model in MODELS and row == "gptq":
        got = Z.sha256_file(Z.codes_file(model))
        return [] if got == CODES_FILE_SHA[model] else [f"codes file sha256 {got[:12]} != {CODES_FILE_SHA[model][:12]}"]
    return _verify_inputs(model, row)


Z.verify_inputs = verify_inputs                         # run_zeroshot6.run_row looks it up at call time


def preflight():
    """Every input of the 20 rows against its record, and the code; RUN/zeroshot6/x2/preflight.json."""
    out = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), models={})
    head = subprocess.run(["git", "-C", Z.FQ, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", Z.FQ, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    out["flipquant"] = dict(path=str(Z.FQ), head=head, clean=not dirty, ok=head == Z.FQ_COMMIT and not dirty)
    out["part_x_step0_ok"] = bool(Z.step0_value("ok"))
    for m in MODELS:
        rows = {}
        for row in Z.ROWS:
            bad = verify_inputs(m, row)
            rows[row] = dict(ok=not bad, problems=bad)
            if row in ("if4", "zou"):
                want = Z.installed_sha(json.loads(Z.mainrow_record(m, row).read_text()))
                rows[row].update(main_table_installed_weight_sha256=want, ok=bool(want))
            if row == "gptq":
                rec = json.loads(Z.gptq_record(m).read_text())["ptq"]
                rows[row].update(main_table_codes_sha256=rec["codes_sha256"], calib_record=str(Z.calib_record(m)),
                                 calib_extension=str(Z.calib_extension(m)))
                for p in (Z.calib_record(m), Z.calib_extension(m)):
                    if not Path(p).exists():
                        rows[row]["ok"] = False
                        rows[row]["problems"].append(f"missing {p}")
        out["models"][m] = dict(revision=Z.REV[m], rows=rows)
    out["ok"] = out["flipquant"]["ok"] and out["part_x_step0_ok"] and all(
        r["ok"] for m in out["models"].values() for r in m["rows"].values())
    X2.mkdir(parents=True, exist_ok=True)
    (X2 / "preflight.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(dict(ok=out["ok"], flipquant=out["flipquant"],
                          problems={m: {r: v["problems"] for r, v in d["rows"].items() if v["problems"]}
                                    for m, d in out["models"].items()}), indent=1))
    return out


def alive(pid):
    try:
        return "run_gsm8k6.py" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def triggered():
    with open(G.RUN / "queue.log", "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - (1 << 16)))
        return TRIGGER in f.read().decode(errors="replace")


def hold(part_y_pid):
    """Wait for part Y's last native row (or its driver's exit), then hold a CUDA context for this process's life."""
    G.log(f"zeroshot6 X2: waiting for part Y's last native row ({TRIGGER}) or part Y's driver (pid {part_y_pid}) to exit")
    while not triggered() and alive(part_y_pid):
        time.sleep(30)
    import torch
    torch.cuda.init()
    held = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    G.log(f"zeroshot6 X2 hold: pid {os.getpid()} holds a CUDA context (part Y amendment 1; part Y's driver "
          f"{'running' if alive(part_y_pid) else 'not running'})")
    own = str(os.getpid())

    def idle():
        """run_gpu.idle, ignoring this process's own (held) context."""
        while True:
            q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                               text=True).stdout.split()
            if not [p for p in q if p.strip() and p.strip() != own]:
                return
            time.sleep(15)

    G.idle = idle                                       # run_gpu.run looks it up at call time
    return held


def panel(part_y_pid):
    pre = preflight()
    if not pre["ok"]:
        G.stop("zeroshot6 X2: preflight failed (RUN/zeroshot6/x2/preflight.json)")
    held = hold(part_y_pid)
    head = subprocess.run(["git", "-C", Z.FQ, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", Z.FQ, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if head != Z.FQ_COMMIT or dirty:
        G.stop(f"zeroshot6 X2: flipquant {Z.FQ} is at {head[:7]} (dirty: {bool(dirty)}), not a clean {Z.FQ_COMMIT[:7]}")
    libs = G.manifests()
    G.log(f"zeroshot6 X2 panel models {','.join(MODELS)}")
    for model in MODELS:
        digests = {}
        for row in Z.ROWS:
            Z.run_row(model, row, libs, digests)
        G.log(f"zeroshot6 X2 {model}: all rows done")
    G.log("DONE zeroshot6 X2 panel; the hold ends with this process")
    del held


if __name__ == "__main__":
    what = sys.argv[1]
    pid = next((int(a.split("=", 1)[1]) for a in sys.argv[2:] if a.startswith("--part-y-pid=")), None)
    if not (G.RUN / "env" / "fast_env_verified.json").exists():
        G.stop("the hybrid models run only in n16k64-fast, after its verification (amendment 3)")
    if what == "preflight":
        preflight()
    elif what == "panel":
        if pid is None:
            raise SystemExit("panel needs --part-y-pid=<part Y's driver pid>")
        panel(pid)
    else:
        raise SystemExit(f"unknown part {what}")
