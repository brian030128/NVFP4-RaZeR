"""Part AA (PROTOCOL.md): the 13-gram audit of each model's 2,048-window calibration set (which contains its 1,024 and
256 sets) against every evaluation text. CPU.

    python calib_size_audit.py      # after run_calib_size.py windows; writes RUN/calib_size/windows/audit.json

- Windows: flipquant.data.calibration_release(tok, record, windows/<model>_2048.json) on calib-size, so every window is
  rebuilt from its document and checked by token sha256, then decoded with the model's tokenizer.
- Evaluation texts (supersets of what is scored):
  - WikiText-2 raw test @ WIKI_REVISION, the whole split (the PPL windows are its prefix);
  - the C4 validation documents of each model's PPL windows (flipquant.data.eval_windows' records), whole;
  - the six zero-shot tasks' evaluated splits at part X's pinned revisions, every string field of each row;
  - GSM8K main test @ 740312ad (question and answer) and gsm8k_llama's eight few-shot exemplars.
- Normalization: lowercase, every character outside [a-z0-9] a separator; word 13-grams within each text.
- A hit is a window sharing at least one 13-gram with an evaluation text; every hit is listed with the shared n-grams.
  Membership: release (the 128 record windows and each source's first 64 extension windows), 1024 (the next 384 per
  source), 2048 (the rest). pilot_ok: no hit in the 1024 membership of the pilot models.
"""
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_calib_size as CS  # noqa: E402

sys.path.insert(0, str(CS.FQ3))
N = 13
WORD = re.compile(r"[a-z0-9]+")
ZS = {  # task: (repo, config, revision, split, examples) -- part X's dataset_loads and SIZES
    "piqa": ("baber/piqa", None, "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1", "validation", 1838),
    "hellaswag": ("Rowan/hellaswag", None, "218ec52e09a7e7462a5400043bb9a69a41d06b76", "validation", 10042),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "validation",
                   1267),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "210d026faf9955653af8916fad021475a3f00453", "test", 2376),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453", "test", 1172),
    "lambada_openai": ("EleutherAI/lambada_openai", "default", "900124bf3b8235c6daf21033af9948b3f07346c4", "test", 5153)}
GSM8K = ("openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866", "test", 1319)


def words(text):
    return WORD.findall(text.lower())


def grams(ws):
    return (" ".join(ws[i:i + N]) for i in range(len(ws) - N + 1))


def strings(obj):
    """Every string leaf of a dataset row, in field order."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in strings(v)]
    return []


class Index:
    def __init__(self):
        self.first, self.stats = {}, {}

    def add(self, name, item, text):
        ws = words(text)
        st = self.stats.setdefault(name, dict(items=0, words=0, grams=0))
        st["items"] += 1
        st["words"] += len(ws)
        for g in grams(ws):
            st["grams"] += 1
            self.first.setdefault(g, (name, item))


def eval_index(toks):
    from datasets import load_dataset
    import yaml
    from flipquant import data as D
    idx = Index()
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=D.WIKI_REVISION, split="test")
    idx.add("wikitext2_test", "test", "\n\n".join(wiki["text"]))
    c4 = load_dataset("allenai/c4", revision=D.C4_REVISION, data_files={"validation": D.C4_FILE}, split="validation")
    docs = sorted({d["index"] for tok in toks.values() for d in D.eval_windows(tok, "c4", 2048)[1]["documents"]})
    for i in docs:
        idx.add("c4_validation_eval", i, c4[i]["text"])
    for task, (repo, config, rev, split, n) in ZS.items():
        ds = load_dataset(repo, config, revision=rev, split=split)
        assert len(ds) == n, (task, len(ds))
        for i, row in enumerate(ds):
            idx.add(task, i, " ".join(strings(row)))
    repo, config, rev, split, n = GSM8K
    ds = load_dataset(repo, config, revision=rev, split=split)
    assert len(ds) == n
    for i, row in enumerate(ds):
        idx.add("gsm8k_test", i, row["question"] + " " + row["answer"])
    import lm_eval
    cfg = yaml.safe_load((Path(lm_eval.__file__).parent / "tasks" / "llama3" / "instruct" / "gsm8k" / "gsm8k.yaml")
                         .read_text())
    for i, s in enumerate(cfg["fewshot_config"]["samples"]):
        idx.add("gsm8k_llama_fewshot", i, s["question"] + " " + s["target"])
    assert i == 7
    return idx, dict(c4_documents=len(docs))


def membership(i, rec):
    if i < 128:
        return "release"
    k = rec["k"]
    return "release" if k < 64 else "1024" if k < 448 else "2048"


def main():
    from transformers import AutoTokenizer
    from flipquant import data as D
    t0 = time.time()
    models = list(CS.SIZES)
    files = {m: CS.windows_file(m, 2048) for m in models}
    toks = {}
    for m in models:
        w = json.loads(files[m].read_text())
        toks[m] = AutoTokenizer.from_pretrained(w["tokenizer"]["source"], revision=w["tokenizer"]["revision"])
    idx, info = eval_index(toks)
    out = dict(n=N, normalization="lowercase; characters outside [a-z0-9] separate words; word 13-grams within each "
                                  "text", eval_sets=idx.stats, **info, models={})
    for m in models:
        w = json.loads(files[m].read_text())
        rel = D.calibration_release(toks[m], CS.record(m), files[m])
        assert rel["meta"]["token_sha256"] == w["token_sha256"], m
        recs, seen = [], {"math": 0, "code": 0}
        for r in w["fit_extension"]["records"]:
            recs.append(dict(r, k=seen[r["source"]]))
            seen[r["source"]] += 1
        hits, by = [], {}
        for i, ids in enumerate(rel["fit"]):
            text = toks[m].decode(ids[0].tolist(), skip_special_tokens=True)
            shared = {}
            for g in grams(words(text)):
                if g in idx.first:
                    shared.setdefault(idx.first[g], []).append(g)
            if not shared:
                continue
            rec = recs[i - 128] if i >= 128 else None
            mem = membership(i, rec)
            for (name, item), gs in shared.items():
                by.setdefault(name, {}).setdefault(mem, set()).add(i)
                hits.append(dict(window=i, membership=mem, source=rec["source"] if rec else "record",
                                 document_sha256=rec["document_sha256"] if rec else None,
                                 offset=rec["offset"] if rec else None, eval_set=name, item=item,
                                 shared_grams=len(gs), first_gram=gs[0]))
        out["models"][m] = dict(windows=len(rel["fit"]), token_list_sha256=CS.token_list_sha(w["token_sha256"]),
                                hits=hits, windows_with_hits={s: {k: len(v) for k, v in d.items()} for s, d in by.items()})
    out["pilot_ok"] = not any(h["membership"] == "1024" for m in CS.PILOT for h in out["models"][m]["hits"])
    out["seconds"] = time.time() - t0
    path = CS.WIN / "audit.json"
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(dict(pilot_ok=out["pilot_ok"], eval_sets=idx.stats,
                          hits={m: v["windows_with_hits"] for m, v in out["models"].items()}), indent=1))


if __name__ == "__main__":
    main()
