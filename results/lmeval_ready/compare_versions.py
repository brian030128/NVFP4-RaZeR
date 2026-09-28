"""Task 3: what differs between lm-eval 0.4.5, 0.4.9.1 and 0.4.11 for this suite (READINESS.md). CPU only, no model.

For each installed version it compares against 0.4.11:
- the task files: the YAML configs and the Python helpers they reference (byte-level; unified diffs are written);
- the code paths that turn a document into requests and requests into scores, by AST
  (formatting and comments ignored):
  - ConfigurableTask.construct_requests / process_results / doc_to_text / doc_to_target / doc_to_choice;
  - the metrics (acc, acc_norm via process_results; exact_match) and the gsm8k answer filters;
  - HFLM's tokenization and scoring (tok_encode, _encode_pair, _loglikelihood_tokens, generate_until,
    _model_generate).

    python results/lmeval_ready/compare_versions.py   -> versions.{json,md}, versions_diffs/*.diff
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSIONS = {'0.4.11': Path('/home/dev/.conda/envs/n16k64/lib/python3.11/site-packages/lm_eval'),
            '0.4.9.1': Path('/home/dev/n16k64_campaign/lmeval_versions/lmeval049/pydeps/lm_eval'),
            '0.4.5': Path('/home/dev/n16k64_campaign/lmeval045/pydeps/lm_eval')}
REFERENCE = '0.4.11'
TASK_FILES = ['tasks/arc/arc_easy.yaml', 'tasks/arc/arc_challenge.yaml', 'tasks/hellaswag/hellaswag.yaml',
              'tasks/hellaswag/utils.py', 'tasks/openbookqa/openbookqa.yaml', 'tasks/super_glue/boolq/default.yaml',
              'tasks/winogrande/default.yaml', 'tasks/winogrande/preprocess_winogrande.py', 'tasks/piqa/piqa.yaml',
              'tasks/gsm8k/gsm8k.yaml']
CODE = {'api/task.py': ['ConfigurableTask.construct_requests', 'ConfigurableTask.process_results', 'ConfigurableTask.doc_to_text',
                        'ConfigurableTask.doc_to_target', 'ConfigurableTask.doc_to_choice', 'ConfigurableTask.fewshot_context'],
        'api/metrics.py': ['acc_fn', 'acc_norm_fn', 'exact_match_fn', 'mean'],
        'filters/extraction.py': ['RegexFilter.apply', 'RegexFilter.__init__'],
        'filters/selection.py': ['TakeFirstFilter.apply'],
        'models/huggingface.py': ['HFLM.tok_encode', 'HFLM._encode_pair', 'HFLM._loglikelihood_tokens', 'HFLM.generate_until',
                                  'HFLM._model_generate', 'HFLM._model_call', 'HFLM.loglikelihood']}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def functions(path):
    """{qualified name: normalized AST dump} of the functions and methods in a file."""
    out = {}
    if not path.exists():
        return out
    tree = ast.parse(path.read_text())

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f'{prefix}{child.name}.')
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = child.body
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    body = body[1:]           # the docstring
                out[f'{prefix}{child.name}'] = (ast.dump(ast.Module(body=body, type_ignores=[])), ast.unparse(child))
    visit(tree, '')
    return out


def main():
    diffs = HERE / 'versions_diffs'
    diffs.mkdir(exist_ok=True)
    ref = VERSIONS[REFERENCE]
    report = dict(reference=REFERENCE, roots={v: str(p) for v, p in VERSIONS.items()}, task_files={}, code={})
    for rel in TASK_FILES:
        row = {}
        for v, root in VERSIONS.items():
            row[v] = sha(root / rel)
            if v != REFERENCE and row[v] and row[v] != row[REFERENCE]:
                d = difflib.unified_diff((ref / rel).read_text().splitlines(True), (root / rel).read_text().splitlines(True),
                                         f'{REFERENCE}/{rel}', f'{v}/{rel}')
                (diffs / f"{v}_{rel.replace('/', '_')}.diff").write_text(''.join(d))
        report['task_files'][rel] = row
    for rel, names in CODE.items():
        per = {v: functions(root / rel) for v, root in VERSIONS.items()}
        for name in names:
            row = {}
            for v in VERSIONS:
                if name not in per[v]:
                    row[v] = 'absent'
                    continue
                row[v] = 'same' if v == REFERENCE or (name in per[REFERENCE] and per[v][name][0] == per[REFERENCE][name][0]) \
                    else 'differs'
                if row[v] == 'differs' and name in per[REFERENCE]:
                    d = difflib.unified_diff(per[REFERENCE][name][1].splitlines(True), per[v][name][1].splitlines(True),
                                             f'{REFERENCE}/{rel}:{name}', f'{v}/{rel}:{name}')
                    (diffs / f"{v}_{rel.replace('/', '_')}_{name}.diff").write_text(''.join(d))
            report['code'][f'{rel}:{name}'] = row
    (HERE / 'versions.json').write_text(json.dumps(report, indent=1) + '\n')
    md = ['# lm-eval versions against 0.4.11 (generated by compare_versions.py)', '',
          'Task files: sha256 prefix, "=" when byte-identical to 0.4.11. Code: AST comparison (docstrings, comments and '
          'formatting ignored). Diffs of everything that differs: `versions_diffs/`.', '',
          '| task file | ' + ' | '.join(VERSIONS) + ' |', '|---|' + '---|' * len(VERSIONS)]
    for rel, row in report['task_files'].items():
        md.append(f'| {rel} | ' + ' | '.join(('=' if v != REFERENCE and row[v] == row[REFERENCE] else (row[v] or 'absent'))
                                           for v in VERSIONS) + ' |')
    md += ['', '| code | ' + ' | '.join(VERSIONS) + ' |', '|---|' + '---|' * len(VERSIONS)]
    for name, row in report['code'].items():
        md.append(f'| {name} | ' + ' | '.join(row[v] for v in VERSIONS) + ' |')
    (HERE / 'versions.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
