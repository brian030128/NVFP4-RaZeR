"""Part X, deviation 1 (PROTOCOL.md): run a flipquant module as `python -m <module>` would, with paper-sm120-runs
7b472b0's registry entry for llama3.1-8b-instruct added at run time. flipquant main (3db0072) lacks the entry; nothing in
the flipquant checkout changes. Run it from the flipquant worktree:

    cd <flipquant worktree> && python zs6_fq.py evaluation.downstream --model llama3.1-8b-instruct ...
"""
import os
import runpy
import sys

sys.path[0] = os.getcwd()          # the flipquant worktree, where `python -m` puts it (not this script's directory)

from models import registry  # noqa: E402
from models.registry import ModelSpec  # noqa: E402

# paper-sm120-runs 7b472b0, models/registry.py (its alias llama8b-instruct is not needed)
SPEC = ModelSpec("llama3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct",
                 revision="0e9e39f249a16976918f6564b8830bc894c89659", notes="tab:main-ppl (paper-eval part P)")

if __name__ == "__main__":
    if SPEC.key in registry.MODELS:
        sys.exit(f"zs6_fq.py: {SPEC.key} is already registered; run the module directly")
    registry.MODELS[SPEC.key] = SPEC
    module, sys.argv = sys.argv[1], sys.argv[1:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
