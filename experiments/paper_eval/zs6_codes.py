"""Part X, step 0 (c), GPTQ-double-dagger (PROTOCOL.md): every main-table GPTQ codes file the panel loads, checked on
the CPU against its record (the main-table ppl.json):
- the code digest, flipquant's codes_sha256 formula over the stored entries (paper-sm120-runs gptq.entries_sha256: per
  module in name order, the name, the packed nibbles, the E4M3 scale bytes, the float32 global scale and the formats);
- the file's key (the GPTQ settings and the calibration data's sha256) against the record's ptq fields;
- the file's own sha256 and size (recorded here; no earlier record has them).

    python zs6_codes.py --out <json>
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

RUN = Path("/home/dev/n16k64_campaign/paper_eval")
FILES = {"mistral-7b": (RUN / "gptqdd" / "mistral-7b" / "codes_nvfp4_act-fo6.pt", RUN / "gptqdd" / "mistral-7b" / "ppl.json"),
         "phi4-14b": (RUN / "gptqdd" / "phi4-14b" / "codes_nvfp4_act-fo6.pt", RUN / "gptqdd" / "phi4-14b" / "ppl.json"),
         "nemotron-nano-9b-v2": (RUN / "ptq" / "nemotron-nano-9b-v2" / "codes_nvfp4_act-fo6.pt",
                                 RUN / "gptqdd" / "nemotron-nano-9b-v2" / "ppl.json"),
         "qwen3.8-27b": (RUN / "ptq" / "qwen3.8-27b" / "codes_nvfp4_act-fo6.pt", RUN / "gptqdd" / "qwen3.8-27b" / "ppl.json")}
KEY_FIELDS = ("model", "weight", "act", "unit", "mode", "map_sha256", "rotate", "damp", "block", "windows", "calib", "rules",
              "hessian_input", "data_sha256", "propagate")


def entries_sha256(entries):
    h = hashlib.sha256()
    for n in sorted(entries):
        c = entries[n]
        h.update(n.encode())
        for t in (c["q"], c["s"], c["g"].float()):
            h.update(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        h.update(c["fmt"].numpy().tobytes() if torch.is_tensor(c["fmt"]) else str(int(c["fmt"])).encode())
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for model, (path, record) in FILES.items():
        ptq = json.loads(record.read_text())["ptq"]
        obj = torch.load(path, map_location="cpu", weights_only=True)
        key = obj["key"]
        digest = entries_sha256(obj["codes"])
        key_diff = {f: (key.get(f), ptq.get(f)) for f in KEY_FIELDS if key.get(f) != ptq.get(f)}
        out[model] = dict(file=str(path), bytes=path.stat().st_size, file_sha256=file_sha256(path), record=str(record),
                          format=obj.get("format"), modules=len(obj["codes"]), codes_sha256=digest,
                          record_codes_sha256=ptq["codes_sha256"], record_cache=ptq.get("cache"), key=key,
                          key_differences=key_diff)
        out[model]["ok"] = (obj.get("format") == "flipquant-gptq/1" and digest == ptq["codes_sha256"] and not key_diff
                            and ptq.get("cache") == str(path))
        print(model, "ok" if out[model]["ok"] else "FAILED", digest[:16], ptq["codes_sha256"][:16], key_diff or "", flush=True)
        del obj
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(files=out, ok=all(v["ok"] for v in out.values())), indent=1) + "\n")
    if not all(v["ok"] for v in out.values()):
        sys.exit("a GPTQ codes file differs from its record")


if __name__ == "__main__":
    main()
