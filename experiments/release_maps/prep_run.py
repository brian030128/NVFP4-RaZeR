"""The one-time data preparation of a release model, exactly as calibration.train_map runs it on its first run with
--fit-windows (calibration.tmopt_common.prepare, then prepare_development), into the data root given (fresh).

    python prep_run.py --model <registry key> --root DIR
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/dev/n16k64_campaign/fqopt/wt")
from calibration import tmopt_common as RC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--root", type=Path, required=True)
args = ap.parse_args()
if args.root.exists():
    raise SystemExit(f"{args.root} exists; the preparation is measured from a fresh data root")
spec, key = RC.resolve(args.model)
record = RC.prepare(key, args.root)
development = RC.prepare_development(key, args.root)
print(json.dumps(dict(model=spec.key, key=key, record=str(record), development=development)))
