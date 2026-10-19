"""Part AA: the positive control of the 13-gram audit (calib_size_audit.py), CPU. A zero-hit audit is only evidence if
the audit detects evaluation text that is there. For each model's 2,048 set:
- the scanned windows' 13-gram count (the audit's denominator);
- 30 windows, chosen with a fixed seed, re-scanned as they are (must give the audit's zero hits), and again with one
  evaluation item spliced into the middle of the decoded text: a GSM8K test problem, a HellaSwag context with its
  endings, a WikiText-2 test paragraph, a LAMBADA passage and an ARC-Challenge question. Each splice must be
  detected, and attributed to its own evaluation set.

    python calib_size_audit_control.py      # writes RUN/calib_size/windows/audit_control.json
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calib_size_audit as A  # noqa: E402
import run_calib_size as CS  # noqa: E402


def splices(seed=7):
    from datasets import load_dataset
    from flipquant import data as D
    rng = random.Random(seed)
    out = []
    g = load_dataset(*A.GSM8K[:2], revision=A.GSM8K[2], split=A.GSM8K[3])
    i = rng.randrange(len(g))
    out.append(("gsm8k_test", i, g[i]["question"] + " " + g[i]["answer"]))
    repo, cfg, rev, split, _ = A.ZS["hellaswag"]
    h = load_dataset(repo, cfg, revision=rev, split=split)
    i = rng.randrange(len(h))
    out.append(("hellaswag", i, " ".join(A.strings(h[i]))))
    repo, cfg, rev, split, _ = A.ZS["lambada_openai"]
    lam = load_dataset(repo, cfg, revision=rev, split=split)
    i = rng.randrange(len(lam))
    out.append(("lambada_openai", i, lam[i]["text"]))
    repo, cfg, rev, split, _ = A.ZS["arc_challenge"]
    arc = load_dataset(repo, cfg, revision=rev, split=split)
    i = rng.randrange(len(arc))
    out.append(("arc_challenge", i, " ".join(A.strings(arc[i]))))
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=D.WIKI_REVISION, split="test")
    paras = [t for t in wiki["text"] if len(A.words(t)) > 60]
    out.append(("wikitext2_test", "test", rng.choice(paras)))
    return out


def main(models=None):
    from transformers import AutoTokenizer
    from flipquant import data as D
    models = models or list(CS.SIZES)
    toks = {}
    for m in models:
        w = json.loads(CS.windows_file(m, 2048).read_text())
        toks[m] = AutoTokenizer.from_pretrained(w["tokenizer"]["source"], revision=w["tokenizer"]["revision"])
    idx, _ = A.eval_index(toks)
    items = splices()
    path = CS.WIN / "audit_control.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    res = dict(items=[dict(eval_set=s, item=i, words=len(A.words(t))) for s, i, t in items],
               models=prev.get("models", {}))
    for m in models:
        rel = D.calibration_release(toks[m], CS.record(m), CS.windows_file(m, 2048))
        texts = [toks[m].decode(ids[0].tolist(), skip_special_tokens=True) for ids in rel["fit"]]
        scanned = sum(max(0, len(A.words(t)) - A.N + 1) for t in texts)
        pick = random.Random(11).sample(range(len(texts)), 30)
        plain = sum(1 for i in pick if any(g in idx.first for g in A.grams(A.words(texts[i]))))
        detect = {}
        for s, item, text in items:
            ok = 0
            for i in pick:
                ws = A.words(texts[i])
                mid = len(ws) // 2
                spliced = ws[:mid] + A.words(text) + ws[mid:]
                found = {idx.first[g] for g in A.grams(spliced) if g in idx.first}
                ok += (s, item) in found
            detect[s] = ok
        res["models"][m] = dict(windows=len(texts), scanned_13grams=scanned, plain_windows_with_hits=plain,
                                spliced_detected=detect, picked=len(pick))
    res["ok"] = all(v["plain_windows_with_hits"] == 0 and all(d == v["picked"] for d in v["spliced_detected"].values())
                    for v in res["models"].values())
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sel = [a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--models=")]
    main(sel[0] if sel else None)
