"""Where the raw-byte scan hits of the staged GPTQ codes files lie (hf_gptq_focus.py; the coordinator's question of
2026-10-15): for every hit, the zip member it falls in (torch archives are stored uncompressed) and, for tensor-storage
members, the tensor that storage backs (module and field, from the pickle's persistent ids). CPU only, read-only.

    python hf_scan_hits.py      # RECS/scan_hits.json
"""
import collections
import json
import pickle
import struct
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_gptq_focus as H  # noqa: E402


def members(path):
    """[(name, data_start, data_end)] of a stored (uncompressed) zip archive."""
    out = []
    with zipfile.ZipFile(path) as z, open(path, "rb") as f:
        for info in z.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, info
            f.seek(info.header_offset)
            head = f.read(30)
            assert head[:4] == b"PK\x03\x04"
            n_name, n_extra = struct.unpack("<HH", head[26:30])
            start = info.header_offset + 30 + n_name + n_extra
            out.append((info.filename, start, start + info.file_size))
    return out


class Tensors(pickle.Unpickler):
    """Unpickles a torch archive's data.pkl without loading storages: each tensor becomes (storage key, shape)."""

    def find_class(self, mod, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return lambda storage, offset, size, *a: ("tensor", storage, tuple(size))
        if mod.startswith("torch"):
            return lambda *a, **k: (mod, name)
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        return ("storage", str(pid[2]))


def storage_owner(path):
    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        obj = Tensors(z.open(pkl)).load()
    owner = {}
    for module, entry in obj["codes"].items():
        for field, t in entry.items():
            if isinstance(t, tuple) and t and t[0] == "tensor":
                owner[t[1][1]] = (module, field, t[2])
    return owner


def main():
    st = json.loads((H.RECS / "stage.json").read_text())
    out = {}
    for repo, d in st["repos"].items():
        if d["kind"] != "gptq":
            continue
        path = Path(d["folder"]) / H.CODES_NAME
        mem = members(path)
        owner = storage_owner(path)
        rows, where = [], collections.Counter()
        raw = [h for h in d["scan"] if h["where"].startswith("raw bytes @")]
        other = [h for h in d["scan"] if not h["where"].startswith("raw bytes @")]
        for h in raw:
            off = int(h["where"].split("@")[1])
            end = off + len(h["match"].encode())
            m = next((x for x in mem if x[1] <= off and end <= x[2]), None)
            key = m[0].rsplit("/", 1)[-1] if m and "/data/" in m[0] else None
            mod, field, shape = owner.get(key, (None, None, None))
            rows.append(dict(offset=off, match=h["match"], member=m[0] if m else None, module=mod, field=field,
                             shape=shape))
            where[(field if field else (m[0] if m else "outside any member"))] += 1
        chars = sorted({c for r in rows for c in r["match"][3:]})
        out[repo] = dict(file=str(path), raw_hits=len(raw), other_hits=other, by_field=dict(where),
                         all_inside_tensor_storage=all(r["field"] is not None for r in rows),
                         patterns=sorted({"hf_ token pattern" if r["match"].startswith("hf_") else r["match"] for r in rows}),
                         match_characters_after_prefix="".join(chars), longest=max((len(r["match"]) for r in rows), default=0),
                         examples=rows[:5])
        print(repo, {k: v for k, v in out[repo].items() if k not in ("examples", "file")}, flush=True)
    (H.RECS / "scan_hits.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
