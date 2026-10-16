"""The coordinator's exact-token check (2026-10-15) before the GPTQ / FOCUS uploads: every staged file's bytes searched for
the Hugging Face tokens present on this machine (the token files uploads use), and for each token's first 15 characters
("hf_" plus 12). In-process only: the tokens are read into memory and never printed, logged or written; the output is
the number of tokens checked and, per file, the number of occurrences (required: 0).

    python hf_token_check.py      # RECS/token_check.json
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_gptq_focus as H  # noqa: E402

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
SOURCES = [HF_HOME / "token", HF_HOME / "stored_tokens", Path.home() / ".cache" / "flipquant" / "hf_upload_token"]


def needles():
    tokens = set()
    for p in SOURCES:
        if p.is_file():
            tokens |= {m.group(0) for m in re.finditer(rb"hf_[A-Za-z0-9]{20,}", p.read_bytes())}
    return sorted(tokens), sorted({t[:15] for t in tokens})


def count(path, full, prefix, bufsize=64 << 20):
    """Occurrences of any full token and of any 15-character prefix, streamed with an overlap."""
    overlap = max(len(x) for x in full + prefix) - 1
    n_full = n_prefix = 0
    prev, pos = b"", 0
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            buf = prev + chunk
            start = 0 if pos == 0 else len(prev)          # count each match once: it must end in the new chunk
            for x in full:
                i = buf.find(x)
                while i >= 0:
                    n_full += i + len(x) > start
                    i = buf.find(x, i + 1)
            for x in prefix:
                i = buf.find(x)
                while i >= 0:
                    n_prefix += i + len(x) > start
                    i = buf.find(x, i + 1)
            prev, pos = buf[-overlap:], pos + len(chunk)
    return n_full, n_prefix


def main():
    full, prefix = needles()
    if not full:
        sys.exit("no token found in the token files; nothing to check against")
    st = json.loads((H.RECS / "stage.json").read_text())
    out = dict(tokens_checked=len(full), token_files=[str(p) for p in SOURCES if p.is_file()], files={})
    for repo, d in st["repos"].items():
        for name in sorted(d["files"]):
            n_full, n_prefix = count(Path(d["folder"]) / name, full, prefix)
            out["files"][f"{repo}/{name}"] = dict(exact_token_hits=n_full, prefix_hits=n_prefix)
    out["all_zero"] = all(v["exact_token_hits"] == 0 and v["prefix_hits"] == 0 for v in out["files"].values())
    del full, prefix
    (H.RECS / "token_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(dict(tokens_checked=out["tokens_checked"], files=len(out["files"]), all_zero=out["all_zero"])))


if __name__ == "__main__":
    main()
