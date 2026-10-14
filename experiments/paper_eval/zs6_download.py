"""Part X (PROTOCOL.md): download the Qwen3.8-27B FOCUS state again (the user deleted part Q's HF cache), the file and
revision of part Q, and check it against part Q's record (RUN/focus_qwen27b/download.json): size, sha256, and the
Hub's LFS sha256. Writes RUN/zeroshot6/step0/focus_qwen27b_download.json. The token is the cached one; it is never
read, printed or logged here.

    python zs6_download.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

RUN = Path("/home/dev/n16k64_campaign/paper_eval")
PART_Q = RUN / "focus_qwen27b" / "download.json"
OUT = RUN / "zeroshot6" / "step0" / "focus_qwen27b_download.json"


def main():
    from huggingface_hub import HfApi, hf_hub_download
    q = json.loads(PART_Q.read_text())
    t0 = time.time()
    path = Path(hf_hub_download(q["repo"], q["file"], revision=q["revision"]))
    seconds = time.time() - t0
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    info = HfApi().get_paths_info(q["repo"], [q["file"]], revision=q["revision"])[0]
    lfs = getattr(info, "lfs", None)
    rec = dict(repo=q["repo"], revision=q["revision"], file=q["file"], path=str(path), bytes=path.stat().st_size,
               sha256=h.hexdigest(), hub_lfs_sha256=getattr(lfs, "sha256", None), seconds=seconds,
               part_q=dict(record=str(PART_Q), bytes=q["bytes"], sha256=q["sha256"]))
    rec["equal_to_part_q"] = rec["bytes"] == q["bytes"] and rec["sha256"] == q["sha256"]
    rec["sha256_equals_hub_lfs"] = rec["sha256"] == rec["hub_lfs_sha256"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("bytes", "sha256", "equal_to_part_q", "sha256_equals_hub_lfs", "seconds")}))
    if not (rec["equal_to_part_q"] and rec["sha256_equals_hub_lfs"]):
        sys.exit("the downloaded FOCUS state differs from part Q's record")


if __name__ == "__main__":
    main()
