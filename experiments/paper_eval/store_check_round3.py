"""Part O, O3 (the coordinator's request): do the trainer's dense fallback and the hook's new ``pack`` path build the
same lean store? A GPU job (between jobs, before ppl3): the weights and every computation sit on the GPU, as in the
trainer -- on the CPU, flipquant's RTN FourOverSix grid differs from the GPU's in near-tie blocks (the 4-or-6 choice
compares summed squared errors, summed in another order), and the hook's grid guard refuses the codes (the first
attempt of this check, on the CPU, stopped at the first module for that reason).

Nemotron-Nano-9B-v2's GPTQ-candidate map (13:33-15:34 UTC) ran with flipquant 276ce86's hook: run_train_map's RTN
pre-packer ``pack()`` refused the GPTQ candidates of every module, and the trainer stored them through its dense
fallback, ``store.add(n, w, b, a)``. Qwen3.8-27B's attempt 2 runs with a53e122's hook, which supplies ``pack``:
``store.add(n, w, decode_base(p), decode_alt(p))``. The trainer trains on the lean store only, and it was not saved.
So both store entries are rebuilt here for every module, from the same code files and the checkpoint's weights (each
checked against the calibration record's sha256, as the trainer checks them), and compared byte for byte:

- path A, the 15:34 run: 276ce86's tmopt_launch.Candidates (``git show``); its prepared candidates and decoded b / a;
  the trainer's own ``pack`` (must return None, as in that run); ``native_dev.pack_candidates(n, w, b, a)``, with rq's
  encoders returning the given codes as that hook's ``_encode`` did;
- path B, the new hook: a53e122's Candidates; ``Candidates.pack`` (checked with the trainer's decoders);
  ``native_dev.pack_candidates(n, w, decode_base(p), decode_alt(p))``.

    python store_check_round3.py [--model nemotron-nano-9b-v2]   ->  RUN/ptq_round3/<model>/store_check.json
    (run_ptq_round3.py ppl3 calls check() for a map trained through the dense fallback)

The record holds per module whether the five store tensors (E2M1 / E0M3 packed codes, their scale bytes, the global
scale) are equal, their sha256 on each path, and the total bytes (the 15:34 run reported 8.0798 GiB).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402
import run_ptq_round3 as P  # noqa: E402
import run_sm120 as S  # noqa: E402

OLD_HOOK = "276ce86"        # the flipquant commit Nemotron's GPTQ-candidate run used (its run note)

CODE = r"""
import hashlib, importlib.util, json, sys, time, types
from pathlib import Path
REPO = Path(sys.argv[1]); old_hook, record, e2m1, e0m3, out = sys.argv[2:7]
ENGINE = REPO / "calibration" / "tmopt"
sys.path[0:0] = [str(ENGINE), str(ENGINE / "repro_local" / "realquant")]
sys.path.append(str(REPO))
import torch
import candidate_store, native_dev, rq                       # noqa: F401 (the lean store's modules)
from quantize.packed_candidates import decode_alt, decode_base, pack
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_conditional_format import sha
from transformers import AutoModelForCausalLM


def hook_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t0 = time.time()
prior = json.loads(Path(record).read_text())
model = AutoModelForCausalLM.from_pretrained(prior["source"], revision=prior["revision"], torch_dtype=torch.bfloat16,
                                             attn_implementation="eager", device_map="cuda")   # the trainer's own call
modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
assert list(modules) == list(prior["matrices"])
hooks = {}
for label, path in (("A", old_hook), ("B", REPO / "calibration" / "tmopt_launch.py")):
    L = hook_module(path, f"tmopt_launch_{label}")
    R = types.SimpleNamespace(sha=lambda t: "x", quant_nvfp4_4over6=quant_nvfp4_4over6, quant_mix_4_6=quant_mix_4_6,
                              pack=pack)
    hooks[label] = L.Candidates(R, {"weights": {}}, dict(e2m1=e2m1, e0m3=e0m3), list(modules), None)
assert not hasattr(hooks["A"], "pack"), "path A must be the hook without its own pack"
orig_rq = (rq.weight_four_over_six, rq.weight_e0m3)
rec = dict(modules={}, digest={}, bytes={})
digest = {k: hashlib.sha256() for k in "AB"}
total = {k: 0 for k in "AB"}
names = ("e2m1_codes", "e0m3_codes", "e2m1_scale_bytes", "e0m3_scale_bytes", "global_scale")
for n, m in modules.items():
    w = m.weight.detach()
    assert sha(w) == prior["matrices"][n]["source_sha256"], n
    entry = {}
    stores = {}
    for label, hook in hooks.items():
        prepared = hook._prepare(n, w)
        hook.current = (n, w, prepared)
        b, a = hook._decode("e2m1"), hook._decode("e0m3")
        (v4, s4, g4), (v0, s0, g0) = prepared["e2m1"], prepared["e0m3"]
        # the hook's _encode: rq's encoders return the given codes, scales and global scale
        rq.weight_four_over_six = lambda x, v=v4, s=s4, g=g4: (v.clone(), s.clone(), g.clone())
        rq.weight_e0m3 = lambda x, v=v0, s=s0, g=g0: (v.clone(), s.clone(), g.clone())
        if label == "A":
            assert pack(w, b, a) is None, f"{n}: the trainer's pack accepted the GPTQ candidates"
            stores["A"] = native_dev.pack_candidates(n, w, b, a)                     # the dense fallback
        else:
            p = hook.pack(w, b, a)                                                   # the new hook's pack
            stores["B"] = native_dev.pack_candidates(n, w, decode_base(p), decode_alt(p))
        rq.weight_four_over_six, rq.weight_e0m3 = orig_rq
        hook.current = None
    for label, st in stores.items():
        shas = {}
        for name, t in zip(names, st[:5]):
            raw = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            shas[name] = hashlib.sha256(raw).hexdigest()
            digest[label].update(n.encode() + name.encode() + raw)
            total[label] += len(raw)
        entry[label] = shas
    entry["equal"] = all(torch.equal(x, y) for x, y in zip(stores["A"][:5], stores["B"][:5]))
    rec["modules"][n] = entry
rec.update(status="complete", device=torch.cuda.get_device_name(0), digest={k: d.hexdigest() for k, d in digest.items()},
           bytes=total,
           gib={k: v / 2 ** 30 for k, v in total.items()},
           equal_modules=sum(e["equal"] for e in rec["modules"].values()), modules_total=len(rec["modules"]),
           identical=all(e["equal"] for e in rec["modules"].values()), seconds=time.time() - t0)
Path(out).write_text(json.dumps(rec, indent=1) + "\n")
print("STORE CHECK", rec["equal_modules"], "/", rec["modules_total"], "modules equal; digests", rec["digest"],
      "GiB", rec["gib"], "identical", rec["identical"], f"{rec['seconds']:.0f}s", flush=True)
"""


def check(model, root=P.CODES_ROOT):
    """The record (runs the GPU job first if there is none)."""
    out = P.ROOT / model / "store_check.json"
    if not G.done(out):
        old = P.ROOT / model / f"tmopt_launch_{OLD_HOOK}.py"
        old.write_text(subprocess.run(["git", "show", f"{OLD_HOOK}:calibration/tmopt_launch.py"], cwd=S.FQ2, check=True,
                                      capture_output=True, text=True).stdout)
        cmd = [G.PY, "-c", CODE, S.FQ2, old, S.record(model), P.codes(root, model, "fo6"), P.codes(root, model, "e0m3"),
               out]
        rc = G.run(f"round3_store_check_{model}", cmd, out, S.FQ2 / "calibration" / "tmopt")
        if rc not in (None, 0):
            G.stop(f"round3 store check {model}: rc={rc}")
    return json.loads(out.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nemotron-nano-9b-v2")
    a = ap.parse_args()
    rec = check(a.model)
    print({k: rec[k] for k in ("identical", "equal_modules", "modules_total", "digest", "gib", "seconds")})
