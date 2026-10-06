"""The records of a measured preparation (a fresh data root) against the release data root, file by file (sha256),
for every file the preparation writes under <root>/<key>/ (scratch draws _draw* aside).

    python compare_prep.py --model <registry key> --fresh DIR --out JSON
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/dev/n16k64_campaign/fqopt/wt")
from calibration import tmopt_common as RC  # noqa: E402

RELEASE = Path("/home/dev/n16k64_campaign/fqrel/tmopt_data")
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--fresh", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
_, key = RC.resolve(args.model)


def files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / key).rglob("*")) if p.is_file() and not any(s.startswith("_draw") for s in p.parts)}


fresh, release = files(args.fresh), files(RELEASE)
rec = dict(model=args.model, key=key, files={f: dict(fresh=fresh.get(f), release=release.get(f),
                                                      identical=fresh.get(f) == release.get(f))
                                              for f in sorted(set(fresh) | set(release))})
rec["identical"] = all(v["identical"] for v in rec["files"].values())
args.out.write_text(json.dumps(rec, indent=1) + "\n")
print(("IDENTICAL" if rec["identical"] else "DIFFERENT") + f" {args.model}: {len(rec['files'])} files; "
      + ", ".join(f for f, v in rec["files"].items() if not v["identical"]))
