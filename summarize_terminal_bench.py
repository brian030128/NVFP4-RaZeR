"""
    Turn a harbor job tree into a Terminal-Bench section for the report.

    Reads the `result.json` harbor writes at the root of each job directory, plus the per-trial
    `result.json` files, and reports what actually happened rather than only the headline mean:
    a trial that ERRORED in setup is not a trial the model failed, and conflating the two is the
    easiest way to publish a wrong number from this benchmark.

        python summarize_terminal_bench.py results/terminal_bench/quant_job_*/jobs_* \
            --out results/terminal_bench/SECTION.md
"""

import argparse
import glob
import json
import os


def read_job(job_dir):
    root = os.path.join(job_dir, "result.json")
    if not os.path.isfile(root):
        return None
    top = json.load(open(root))
    trials = []
    for path in sorted(glob.glob(os.path.join(job_dir, "*", "result.json"))):
        try:
            t = json.load(open(path))
        except Exception:
            continue
        trials.append({
            "task": t.get("task_name", os.path.basename(os.path.dirname(path))),
            "reward": (t.get("reward") if isinstance(t.get("reward"), (int, float))
                       else (t.get("verifier_result") or {}).get("reward")),
            "exception": bool(os.path.isfile(
                os.path.join(os.path.dirname(path), "exception.txt"))),
        })
    stats = top.get("stats", {})
    return {"name": os.path.basename(job_dir), "top": top, "stats": stats, "trials": trials}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dirs", nargs="+", help="harbor job directories (containing result.json)")
    ap.add_argument("--out", default="results/terminal_bench/SECTION.md")
    args = ap.parse_args()

    jobs = [j for j in (read_job(d) for d in args.job_dirs) if j]
    assert jobs, f"no harbor result.json under {args.job_dirs}"

    lines = ["## Terminal-Bench", "",
             "| run | trials | completed | errored | mean reward |",
             "|---|---|---|---|---|"]
    for j in jobs:
        s = j["stats"]
        n_total = j["top"].get("n_total_trials", len(j["trials"]))
        n_err = s.get("n_errored_trials", sum(t["exception"] for t in j["trials"]))
        n_done = s.get("n_completed_trials", len(j["trials"])) - n_err
        mean = None
        for ev in (s.get("evals") or {}).values():
            for metric in ev.get("metrics", []):
                if "mean" in metric:
                    mean = metric["mean"]
        lines.append(f"| `{j['name']}` | {n_total} | {n_done} | {n_err} | "
                     f"{'—' if mean is None else f'{mean:.3f}'} |")
    lines.append("")

    # Per-task detail. A benchmark this small is read task by task or not at all.
    lines += ["<details>", "<summary>Per-task outcome</summary>", ""]
    for j in jobs:
        lines += [f"**{j['name']}**", "", "| task | reward | errored |", "|---|---|---|"]
        for t in sorted(j["trials"], key=lambda x: x["task"]):
            r = "—" if t["reward"] is None else f"{t['reward']:.3f}"
            lines.append(f"| {t['task']} | {r} | {'yes' if t['exception'] else 'no'} |")
        lines.append("")
        exc = (j["stats"].get("evals") or {})
        for ev in exc.values():
            for name, who in (ev.get("exception_stats") or {}).items():
                lines.append(f"- `{name}`: {len(who)} trial(s)")
        lines.append("")
    lines += ["</details>", ""]

    out = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(out)
    print(out)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
