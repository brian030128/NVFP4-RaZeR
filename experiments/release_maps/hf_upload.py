"""Upload the staged release-map repos (hf_stage.py) to Hugging Face as PRIVATE model repos, then verify them.

    python hf_upload.py --check          # read-only: the account, and that none of the repos exists yet
    python hf_upload.py --upload         # create (private=True), check private, upload, verify; hf_upload.json

Per repo: create_repo(private=True, exist_ok=False) -> the repo must report private (and be invisible without a token)
before any file goes up -> one upload_folder commit with exactly the staged files -> verify: still private, the file list,
the card metadata the Hub parsed, and every file downloaded again at that commit with its sha256 equal to the staged one.
Stops at the first failure. The active token is used as it is; nothing about tokens is changed or printed.
"""
import argparse
import hashlib
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

STAGE = Path("/home/dev/flipquant_hf")
OUT = STAGE / "hf_upload.json"
ACCOUNT = "edgeai-lab"
MESSAGE = "FlipQuant MixFP4 maps (8x64, 16x64, 256x64), model card, PPL summary and license files"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def anonymous_status(repo_id):
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/models/{repo_id}"), timeout=30)
        return 200
    except urllib.error.HTTPError as e:
        return e.code


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    stage = json.loads((STAGE / "stage.json").read_text())
    api = HfApi()
    who = api.whoami()
    print("account:", who.get("name"))
    if who.get("name") != ACCOUNT:
        sys.exit(f"the active token is not {ACCOUNT}'s; stopping")
    repos = list(stage["repos"])
    exists = {r: api.repo_exists(r, repo_type="model") for r in repos}
    print("existing:", [r for r, e in exists.items() if e] or "none")
    if any(exists.values()):
        sys.exit("a repo already exists; stopping (nothing is overwritten)")
    if args.check:
        return
    out = dict(account=ACCOUNT, repos={})
    for repo_id, rec in stage["repos"].items():
        folder = STAGE / repo_id.split("/", 1)[1]
        staged = {f.name: sha(f) for f in sorted(folder.iterdir())}
        assert staged == {k: v["sha256"] for k, v in rec["files"].items()}, f"{repo_id}: staged files changed"
        url = api.create_repo(repo_id, repo_type="model", private=True, exist_ok=False)
        info = api.model_info(repo_id)
        anon = anonymous_status(repo_id)
        if info.private is not True or anon == 200:
            sys.exit(f"{repo_id}: not private after creation (private={info.private}, anonymous HTTP {anon}); stopping")
        commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=folder,
                                   allow_patterns=list(staged), commit_message=MESSAGE)
        info = api.model_info(repo_id, files_metadata=True)
        files = sorted(s.rfilename for s in info.siblings)
        anon_after = anonymous_status(repo_id)
        problems = []
        if info.private is not True or anon_after == 200:
            problems.append(f"not private (private={info.private}, anonymous HTTP {anon_after})")
        if set(files) - {".gitattributes"} != set(staged):
            problems.append(f"file list {files}")
        card = info.card_data.to_dict() if info.card_data else {}
        if card.get("base_model") != rec["base_model"] or not card.get("license"):
            problems.append(f"card metadata {card}")
        downloaded = {}
        with tempfile.TemporaryDirectory() as tmp:
            for name in staged:
                p = hf_hub_download(repo_id, name, revision=commit.oid, cache_dir=tmp, force_download=True)
                downloaded[name] = sha(p)
        bad = [n for n in staged if downloaded[n] != staged[n]]
        if bad:
            problems.append(f"sha256 differs after download: {bad}")
        out["repos"][repo_id] = dict(url=str(url), commit=commit.oid, commit_url=commit.commit_url, private=info.private,
                                     anonymous_http=anon_after, files={n: dict(sha256=staged[n], bytes=(folder / n).stat().st_size,
                                                                               download_sha256_equal=downloaded[n] == staged[n])
                                                                       for n in staged},
                                     hub_files=files, card_metadata=card, problems=problems)
        OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
        print(f"{repo_id}: commit {commit.oid} private={info.private} anonymous={anon_after} files={len(files)} "
              f"problems={problems or 'none'}")
        if problems:
            sys.exit(f"{repo_id}: verification failed; stopping")


if __name__ == "__main__":
    main()
