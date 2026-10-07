"""The license texts staged with the Hugging Face repos (hf_stage.py reads them from /home/dev/flipquant_hf_licenses):

- llama/LICENSE and llama/USE_POLICY.md: meta-llama/Llama-3.1-8B's own files at the calibration revision. The local
  snapshot holds neither. They were downloaded read-only on 2026-10-07 with a stored token chosen by name (the user's
  decision), passed only to that download call, never printed, and the active account left unchanged.
- nemotron/LICENSE: the NVIDIA Open Model License Agreement. nvidia/NVIDIA-Nemotron-Nano-9B-v2 has no license file, so
  the agreement is taken from the page its card links (page.html, fetched 2026-10-07 05:26 UTC), rendered to text:
  - the title, the displayed "Last Modified" line, then the agreement block (div#listBold);
  - the section numbers are the page's CSS counters ("1. ", nested "1.1 "), and bullets are "•";
  - the paragraph text is verbatim (whitespace collapsed, &nbsp; as a space);
  - the block hidden on every device ("IMPORTANT NOTICE ...") is not shown on the page and not included.

    python hf_licenses.py render      # offline: page.html -> nemotron/LICENSE
    python hf_licenses.py download    # network: the two Llama files (needs the stored token named below)
"""
import hashlib
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path("/home/dev/flipquant_hf_licenses")
LLAMA, LLAMA_REV, TOKEN_NAME = "meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b", "gputw"


def render():
    import lxml.html as LH
    doc = LH.fromstring((ROOT / "nemotron" / "page.html").read_text(encoding="utf-8"))
    title = doc.xpath('//h1[contains(normalize-space(.), "NVIDIA Open Model License Agreement")]')[0]
    lastmod = [p for p in doc.xpath("//p") if "Last Modified" in p.text_content()][0]
    body = doc.xpath('//div[@id="listBold"]//div[contains(@class, "description")]')[0]

    def text(el):
        return re.sub(r"\s+", " ", "".join(el.itertext()).replace("\xa0", " ")).strip()

    out = []

    def block(el, depth=0, prefix=None):
        for ch in el:
            tag = ch.tag if isinstance(ch.tag, str) else None
            if tag == "p":
                out.append("  " * depth + text(ch))
            elif tag == "ul":
                out.extend("  " * depth + "• " + text(li) for li in ch.findall("li"))
            elif tag == "ol":
                custom = "custom" in (ch.get("class") or "")
                for n, li in enumerate(ch.findall("li"), 1):
                    label = li.get("seq") if custom else (str(n) if prefix is None else f"{prefix}.{n}")
                    num = label + (" " if custom or prefix is not None else ". ")
                    first = True
                    for sub in li:
                        stag = sub.tag if isinstance(sub.tag, str) else None
                        if stag in ("ol", "ul"):
                            holder = LH.Element("div")
                            holder.append(sub)
                            block(holder, depth + 1, label)
                        elif stag is not None:
                            out.append("  " * depth + (num if first else "") + text(sub))
                            first = False
                    if first:
                        out.append("  " * depth + num + text(li))
            elif tag is not None and text(ch):
                out.append("  " * depth + text(ch))

    block(body)
    lines = [text(title), "", text(lastmod), ""]
    for b in out:
        lines += [b, ""]
    (ROOT / "nemotron" / "LICENSE").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def download():
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils._auth import get_stored_tokens
    tok = get_stored_tokens().get(TOKEN_NAME)
    if not tok:
        sys.exit(f"no stored token named {TOKEN_NAME}")
    files = HfApi().list_repo_files(LLAMA, revision=LLAMA_REV, token=tok)
    with tempfile.TemporaryDirectory() as tmp:
        for f in ("LICENSE", "USE_POLICY.md"):
            if f in files:
                p = hf_hub_download(LLAMA, f, revision=LLAMA_REV, token=tok, cache_dir=tmp)
                shutil.copyfile(p, ROOT / "llama" / f)
                print(f, hashlib.sha256((ROOT / "llama" / f).read_bytes()).hexdigest())


if __name__ == "__main__":
    {"render": render, "download": download}[sys.argv[1]]()
