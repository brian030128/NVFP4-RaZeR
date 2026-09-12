"""
    Insert the generated CE-and-KL ablation into MIXFP4_REPORT.md.

    It goes inside "3. Why this rule", directly after the "Why both CE and KL" argument it
    tests, so the claim and its evidence are read together. Idempotent: the block lives between
    HTML comment markers and is replaced in place on a rerun, which keeps
    `summarize_objective_ablation.py` the single source of those numbers.

        python scripts/insert_objective_section.py \
            --section results/kse_objective/SECTION.md --report MIXFP4_REPORT.md
"""

import argparse
import re

BEGIN = '<!-- BEGIN GENERATED: objective ablation (summarize_objective_ablation.py) -->'
END = '<!-- END GENERATED: objective ablation -->'

# The block is placed before this heading, i.e. at the end of "Why both CE and KL".
ANCHOR = '### Why k = 3, and why a threshold rather than a fixed count'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--section', default='results/kse_objective/SECTION.md')
    ap.add_argument('--report', default='MIXFP4_REPORT.md')
    args = ap.parse_args()

    section = open(args.section).read().rstrip() + '\n'
    block = f'{BEGIN}\n\n{section}\n{END}\n'

    s = open(args.report).read()
    if BEGIN in s:
        s = re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END) + r'\n', block, s, flags=re.S)
        action = 'replaced'
    else:
        assert ANCHOR in s, f'anchor not found: {ANCHOR}'
        s = s.replace(ANCHOR, block + '\n' + ANCHOR, 1)
        action = 'inserted'

    with open(args.report, 'w') as f:
        f.write(s)
    print(f'{action} {len(block.splitlines())} lines into {args.report}')


if __name__ == '__main__':
    main()
