"""Small dependency-aware dispatchers for queue jobs with frozen dynamic inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from campaign import runtime
from campaign.extension_maps import p41


CR = Path(os.environ["CAMPAIGN_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"


def load(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("p41-map",))
    ap.add_argument("--model", choices=("llama8b", "qwen4b", "mistral7b"))
    args = ap.parse_args()
    if runtime.sha256_file(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    if args.action == "p41-map":
        if not args.model:
            ap.error("p41-map requires --model")
        freeze = load(CR / "provenance/P40_GLOBAL_K_SELECTION.json")
        p41(args.model, float(freeze["global_k"]))


if __name__ == "__main__":
    main()
