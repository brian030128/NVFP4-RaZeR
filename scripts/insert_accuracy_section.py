"""
    Insert the generated zero-shot accuracy section into MIXFP4_REPORT.md.

    Idempotent: the section lives between HTML comment markers and is replaced in place on a
    rerun, so `summarize_zeroshot_kse.py` stays the single source of those numbers.

    It also amends one bullet in "4. Limits". That bullet currently lists accuracy among the
    things the report does not measure, which stops being true once the section is in.

        python scripts/insert_accuracy_section.py \
            --section results/zeroshot_kse/SECTION.md --report MIXFP4_REPORT.md
"""

import argparse
import re

BEGIN = '<!-- BEGIN GENERATED: zero-shot accuracy (summarize_zeroshot_kse.py) -->'
END = '<!-- END GENERATED: zero-shot accuracy -->'

# The section is placed before this heading, i.e. immediately after the results it qualifies.
ANCHOR = '## 2. How the element type is chosen'

OLD_LIMIT = """- Simulated W4A4 on text linear weights and their inputs. No KV-cache
  quantization, generation accuracy, or native FP4 kernel throughput is measured,
  and no speedup is claimed."""
NEW_LIMIT = """- Simulated W4A4 on text linear weights and their inputs. No KV-cache
  quantization or native FP4 kernel throughput is measured, and no speedup is
  claimed. Zero-shot multiple-choice accuracy is measured and reported in §1a;
  generation accuracy still is not."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--section', default='results/zeroshot_kse/SECTION.md')
    ap.add_argument('--report', default='MIXFP4_REPORT.md')
    args = ap.parse_args()

    section = open(args.section).read().rstrip() + '\n'
    # The generated file is titled for standalone reading; inside the report it is a subsection
    # of the results rather than a peer of "1. Results".
    section = section.replace('## Zero-shot accuracy', '## 1a. Zero-shot accuracy', 1)
    block = f'{BEGIN}\n\n{section}\n{END}\n'

    s = open(args.report).read()
    if BEGIN in s:
        s = re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END) + r'\n', block, s, flags=re.S)
        action = 'replaced'
    else:
        assert ANCHOR in s, f'anchor not found: {ANCHOR}'
        s = s.replace(ANCHOR, block + '\n' + ANCHOR, 1)
        action = 'inserted'

    if OLD_LIMIT in s:
        s = s.replace(OLD_LIMIT, NEW_LIMIT, 1)
        limit = 'limits bullet updated'
    elif NEW_LIMIT in s:
        limit = 'limits bullet already current'
    else:
        limit = 'WARNING: limits bullet not found, left alone'

    open(args.report, 'w').write(s)
    print(f'{action}; {limit}; report is {len(s.splitlines())} lines')


if __name__ == '__main__':
    main()
