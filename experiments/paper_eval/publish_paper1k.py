"""Part AC (PROTOCOL.md): an item's report on flipquant paper-1k-reruns.

    python publish_paper1k.py ITEM --paper-eval COMMIT --status TEXT --title TEXT

- Copies the files of ``results/paper_eval/paper1k/<ITEM>/`` as committed in paper-eval ``COMMIT`` (``git show``, not the
  working tree) to ``results/paper_1k/<ITEM>/`` of the paper-1k-reruns worktree. The item's ``SUMMARY.md`` becomes the
  head of its ``README.md``, which also cites the paper-eval commit and lists the files.
- Updates the index ``results/paper_1k/README.md`` (from ``results/paper_1k/index.json``): every item's title, status
  and paper-eval commit, and the flipquant commit that published it (the current item's is the commit being made).
- Scans every written file before the commit: local paths, host names and addresses, Hugging Face tokens (the
  ``hf_`` pattern, and the exact tokens and their 15-character prefixes, hf_token_check). Any hit stops before the
  commit, naming the file and the rule (never the text).
- Commits on paper-1k-reruns. The push is a separate step.
"""
import argparse
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hf_token_check as TC  # noqa: E402

RZ = HERE.parents[1]
FQN = Path("/home/dev/n16k64_campaign/paper1k/wt")
SRC = "results/paper_eval/paper1k"
DST = FQN / "results" / "paper_1k"
ORDER = ["item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9", "item11", "item12", "item13", "item1"]
#: the index's rows before an item is published (results/paper_1k/README.md, 9a7b544): paper item, status
PAPER_ITEM = {
    "item2": ("Table 4 (`tab:latency`) SM120 rows; the memory paragraph",
              "latency done; memory probe interrupted by a GPU hang, resumes after the restart"),
    "item3": ("Table 5 (`tab:ablation`)", "registered"),
    "item4": ("Table F.3 (`tab:ptq`)", "pending"),
    "item5": ("Table D.5 (`tab:app-gemm`)", "pending"),
    "item6": ("Table A.2 (`tab:app-native`)", "pending"),
    "item7": ("Appendix D.6 (reading the weights back)", "pending"),
    "item8": ("Appendix D.4 (real tags vs random placement)", "pending"),
    "item9": ("Tables D.2 / D.3 (`tab:app-ktile`, `tab:app-c2`), RTX PRO 6000 rows", "pending"),
    "item11": ("Tables C.2 / C.3 (calibration cost)", "pending"),
    "item12": ("Table C.5 (`tab:app-robust`)", "pending"),
    "item13": ("Appendix F.1 (the tile-gradient check)", "pending; draft re-implementation, not yet run"),
    "item1": ("MMLU column (Table 3, Tables E.1–E.3)", "pending"),
}
TRAILER = ("Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01S2tbNfhvNWyyJXJHwusL3U")
RULES = [("local path", re.compile(rb"/(?:home|vault|tmp|root|mnt|scratch|data|work|opt)/")),
         ("host name", re.compile(re.escape(socket.gethostname()).encode())),
         ("host name (gpu-<uuid>)", re.compile(rb"gpu-[0-9a-f]{8}-[0-9a-f]{4}")),
         ("ip address", re.compile(rb"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")),
         ("hf token pattern", re.compile(rb"hf_[A-Za-z0-9]{20,}")),
         ("ssh / user@host", re.compile(rb"\b[a-z0-9_.-]+@[a-z0-9-]+\.(?:local|lan|internal)\b"))]


def git(repo, *a, check=True):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, check=check)


def scan(paths):
    full, prefix = TC.needles()
    hits = []
    for p in paths:
        data = Path(p).read_bytes()
        for name, rx in RULES:
            if rx.search(data):
                hits.append((str(Path(p).relative_to(FQN)), name))
        if full:
            n_full, n_prefix = TC.count(p, full, prefix)
            if n_full or n_prefix:
                hits.append((str(Path(p).relative_to(FQN)), "exact token or prefix"))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("item", choices=ORDER)
    ap.add_argument("--paper-eval", required=True, help="the paper-eval commit holding the report")
    ap.add_argument("--status", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--dry-run", type=Path, default=None,
                    help="write and scan under this directory instead of the worktree; no commit")
    a = ap.parse_args()
    global DST, FQN
    if a.dry_run is not None:
        FQN = a.dry_run
        DST = FQN / "results" / "paper_1k"
    commit = git(RZ, "rev-parse", "--short=7", a.paper_eval).stdout.decode().strip()
    names = git(RZ, "ls-tree", "--name-only", commit, f"{SRC}/{a.item}/").stdout.decode().split()
    if not names:
        raise SystemExit(f"no files under {SRC}/{a.item}/ in {commit}")
    out = DST / a.item
    out.mkdir(parents=True, exist_ok=True)
    written, files = [], []
    summary = ""
    for n in names:
        body = git(RZ, "show", f"{commit}:{n}").stdout
        base = Path(n).name
        if base == "SUMMARY.md":
            summary = body.decode()
            continue
        (out / base).write_bytes(body)
        written.append(out / base)
        files.append(base)
    readme = [f"# Part AC item {a.item[4:]}: {a.title}", "",
              f"Status: **{a.status}**. Rendered from NVFP4-RaZeR paper-eval `{commit}` "
              f"(`{SRC}/{a.item}/`); the files here are that commit's, unchanged.", ""]
    if summary:
        readme += [summary.rstrip("\n"), ""]
    readme += ["Files: " + ", ".join(f"`{f}`" for f in files) + "."]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    written.append(out / "README.md")
    idx_path = DST / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    prev = {k: v for k, v in idx.items()}
    idx[a.item] = dict(title=a.title, status=a.status, paper_eval=commit, flipquant=None)
    for k, v in prev.items():                       # items published earlier: their own flipquant commit
        if k != a.item and v.get("flipquant") is None and a.dry_run is None:
            log = git(FQN, "log", "-1", "--format=%h", "--", f"results/paper_1k/{k}/README.md").stdout.decode().strip()
            v["flipquant"] = log or None
    idx_path.write_text(json.dumps(idx, indent=1) + "\n")
    written.append(idx_path)
    rows = ["# The paper's reruns at the 1,024-window calibration (NVFP4-RaZeR paper-eval part AC)", "",
            "The plan, with every item's scale, estimate and status: [PLAN.md](PLAN.md).", "",
            "One directory per finished item, `results/paper_1k/<item>/`: its README (summary and source), its report and "
            "tables. Each report is a copy of the NVFP4-RaZeR paper-eval commit it cites. The items run in the order 2 → 3 "
            "→ 4 → 5 → 6 → 7 → 8 → 9 → 11 → 12 → 13 → 1; item 10 (GB200 / SM100) is run by the co-author, not here.", "",
            "| item | paper item | status | report (paper-eval) | report (flipquant) |", "|---|---|---|---|---|"]
    for k in ORDER:
        paper_item, status = PAPER_ITEM[k]
        if k in idx:
            v = idx[k]
            fq = ("`" + v["flipquant"] + "`") if v.get("flipquant") else ("this commit" if k == a.item else "")
            rows.append(f"| [{k[4:]}]({k}/README.md) | {paper_item} | {v['status']} | `{v['paper_eval']}` | {fq} |")
        else:
            rows.append(f"| {k[4:]} | {paper_item} | {status} | | |")
    (DST / "README.md").write_text("\n".join(rows) + "\n")
    written.append(DST / "README.md")
    hits = scan(written)
    if hits:
        raise SystemExit("SCAN FAILED, nothing committed: " + "; ".join(f"{f}: {r}" for f, r in hits))
    if a.dry_run is not None:
        print(f"dry run: {len(written)} files written under {a.dry_run}, scan clean")
        return
    git(FQN, "add", "-f", *[str(p.relative_to(FQN)) for p in written])     # flipquant's .gitignore has results/*
    msg = (f"results/paper_1k/{a.item}: {a.title} ({a.status}); rendered from NVFP4-RaZeR paper-eval {commit}\n\n"
           f"{TRAILER}\n")
    git(FQN, "commit", "-q", "-m", msg)
    print(git(FQN, "log", "-1", "--format=%h %s").stdout.decode().strip())


if __name__ == "__main__":
    main()
