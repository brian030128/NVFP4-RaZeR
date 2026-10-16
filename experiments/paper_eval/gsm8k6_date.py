"""Part Y amendment 3 (proposed; runs only with the coordinator's / user's OK): run a flipquant module as `python -m
<module>` would, with the chat template's date pinned.

Llama-3.2-1B-Instruct's chat template writes strftime_now("%d %b %Y") into its system header ("Today Date: ..."). The
model's native GSM8K rows ran on 2026-10-15, so its IF4 / MixFP4 rows must render the same date to see the same prompts
(equal sample_digest). transformers' strftime_now calls `datetime.now()` through chat_template_utils' module global
`datetime`. Only that global is replaced, by a datetime whose now() returns the pinned date at 12:00. Nothing in the
flipquant checkout changes, and no other clock is touched. Run it from the flipquant worktree:

    cd <flipquant worktree> && python gsm8k6_date.py 2026-10-15 evaluation.downstream --model llama3.2-1b-instruct ...
"""
import os
import runpy
import sys
from datetime import datetime

sys.path[0] = os.getcwd()          # the flipquant worktree, where `python -m` puts it (not this script's directory)

import transformers.utils.chat_template_utils as CT  # noqa: E402


def pin(day: str):
    y, m, d = (int(v) for v in day.split("-"))

    class PinnedDate(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(y, m, d, 12, 0, 0, tzinfo=tz)

    if CT.datetime is not datetime:
        raise SystemExit("gsm8k6_date.py: transformers' chat_template_utils.datetime is not datetime.datetime")
    CT.datetime = PinnedDate


if __name__ == "__main__":
    day, module = sys.argv[1], sys.argv[2]
    pin(day)
    sys.argv = sys.argv[2:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
