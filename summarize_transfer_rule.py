"""Combine only the matched causal evaluations; keep study boundaries explicit."""
import json
from pathlib import Path
from run_conditional_model import paired


def main():
    root = Path('results/transfer_rule'); root.mkdir(parents=True, exist_ok=True)
    small = ('llama1b', 'opt350m', 'qwen06b', 'pythia14b', 'olmo1b')
    paths = [Path(f'results/causal_replay/model_332374_{m}/report.json') for m in small]
    paths += [Path(f'results/pooled_scale/model_332389_{m}/report.json') for m in ('qwen4b', 'llama8b')]
    reports = [json.loads(p.read_text()) for p in paths]
    assert all(r['status'] == 'complete' for r in reports)
    cells = [(r, d, c) for r in reports for d, c in r['contrasts'].items()]
    diversity = [(r['model'], d, paired(r['evaluation']['mixed64'][d]['nll'], r['evaluation']['c4_64'][d]['nll']))
                 for r in reports for d in r['evaluation']['pooled192']]
    s = dict(models=len(reports), cells=len(cells),
             gains=sum(c['four_over_six']['ppl_delta'] <= -.01 for _, _, c in cells),
             supported_gains=sum(c['four_over_six']['mean_nll']+c['four_over_six']['two_se'] < 0 for _, _, c in cells),
             supported_harms=sum(c['four_over_six']['mean_nll']-c['four_over_six']['two_se'] > 0 for _, _, c in cells),
             beats_weight_mse=sum(c['weight_mse']['mean_nll'] < 0 for _, _, c in cells),
             beats_c4_64=sum(c['c4_64']['mean_nll'] < 0 for _, _, c in cells),
             equal_token_diversity_point_gains=sum(c['mean_nll'] < 0 for _, _, c in diversity),
             equal_token_diversity_supported_gains=sum(c['mean_nll']+c['two_se'] < 0 for _, _, c in diversity),
             equal_token_diversity_supported_harms=sum(c['mean_nll']-c['two_se'] > 0 for _, _, c in diversity),
             original_confirmation_full_screen_passes=False,
             causal_audit_passes=json.loads(Path('results/causal_replay/summary_332374.json').read_text())['passes_causal_audit'],
             scale_transfer_passes=json.loads(Path('results/pooled_scale/summary_332389.json').read_text())['passes_scale_transfer'])
    s['source_reports'] = [str(p) for p in paths]
    (root/'summary.json').write_text(json.dumps(s, indent=2)+'\n')
    lines = ['# A fixed FP4 tile rule with measured transfer', '',
             f'The unchanged procedure improves **{s["gains"]}/{s["cells"]}** matched causal perplexity comparisons '
             f'across **{s["models"]} models**, from OPT350M to Llama8B. '
             f'**{s["supported_gains"]}** gains have supporting descriptive paired2SE intervals; '
             f'**{s["supported_harms"]}** comparisons have supported harm. '
             f'It beats weight-MSE election in **{s["beats_weight_mse"]}/{s["cells"]}** point comparisons.', '',
             'All rows below use per-token FP32 activation factors for both the selected map and its baseline. '
             'There is one fixed map per model across literature, science and government text. '
             'No confirmation loss chooses a map, tile budget, calibration source, seed or checkpoint.', '',
             '## The rule', '',
             '1. Form fixed FourOverSix and E0M3-alpha1 candidates for every legal8x64 weight tile.',
             '2. Build one shared table of CE and teacher-KL tile derivatives on a fixed pool of192 sequences '
             '(64 C4,64 OpenWebMath,64 CodeParrot;512tokens each). All controls reuse that table.',
             '3. For each tile, take the maximum of the CE and KL directional mean+2SE. '
             'Choose up to256 tiles with the most negative scores.',
             '4. Freeze the map. Evaluate all domains with that same map, including failures.', '',
             'The rule is the exact top-k solution of an additive surrogate. '
             'It is not an exact optimizer of nonlinear network loss. '
             'The cap256 and factor2 are common empirical constants, not parameters derived from a universal theorem. '
             'They remain unchanged across all seven models. '
             '[Full method](../pooled_confirmation/METHOD.md).', '',
             '## Matched causal results', '',
             '[Figure (PDF)](causal_transfer.pdf) · [SVG](causal_transfer.svg)', '',
             '| Model / domain | FourOverSix | Selected | ΔPPL | ΔNLL ±2SE |',
             '|---|---:|---:|---:|---:|']
    for r, d, c in cells:
        b = c['four_over_six']; e = r['evaluation']
        lines.append(f'| {r["model"]} / {d} | {e["four_over_six"][d]["ppl"]:.6f} | {e["pooled192"][d]["ppl"]:.6f} | {b["ppl_delta"]:+.6f} | {b["mean_nll"]:+.6f} ±{b["two_se"]:.6f} |')
    lines += ['', '## What the experiments establish', '',
              'The initial confirmation used five model families and three data families not inspected during method development; '
              'Pythia and OLMo supplied two new architecture families. '
              'A subsequent causal audit replayed the exact same maps and inputs. '
              'The4B/8B follow-up tested larger models on these now-inspected data families. '
              'The21 rows above combine the15 causal replays and6 size-transfer results. '
              'They do not count the earlier window-scaling evaluations again as independent evidence.', '',
              'The old window-wide activation factor changed earlier logits when only future tokens were replaced '
              'in all five models checked under both conventions. '
              'Per-token factors removed that dependence, with bitwise-identical prefix logits for both baseline '
              'and selected maps in all seven models. The frozen maps retained useful gains. '
              '[Causal audit](../causal_replay/REPORT_332374.md), '
              '[scale-transfer report](../pooled_scale/REPORT_332389.md).', '',
              '## Claims that remain limited', '',
              'The original confirmation passed its baseline-transfer components but **failed its complete prespecified screen**: '
              'it beat C4-only selection in8/15 point comparisons, short of the required9. '
              'That criterion has not been changed. '
              'The later causal audit and size-transfer test answer separately declared questions; their passes do not '
              'retroactively rescue the earlier failed comparator criterion.', '',
              f'Under the combined causal convention, pooled192 beats C4-only64 in{s["beats_c4_64"]}/21 point comparisons. '
              f'The equal-token mixed64 comparison improves{s["equal_token_diversity_point_gains"]}/21, '
              f'with{s["equal_token_diversity_supported_gains"]} supported gains and'
              f'{s["equal_token_diversity_supported_harms"]} supported harms. '
              'Thus source diversity is not established as uniformly preferable or as the sole cause of the gains.', '',
              'This supports a transferable calibrated procedure, not a universal weight-only rule. '
              'Gradient ranking, distillation and sparse optimization are established tools. '
              'Paper-level novelty requires a precise comparison with prior work; it does not follow from a favorable table. '
              'The present evidence is reference-text loss from simulated nonhead-linear W4A4, '
              'not generation accuracy or native FP4-kernel throughput. '
              'Per-token factors change the activation representation and are not claimed as a free change '
              'to a single-tensor-factor CUDA interface. '
              'Only one fixed calibration pool per model is used here; multi-pool seed replication remains unmeasured.', '',
              '## Research record and artifacts', '',
              '[Full continuation log](../format_directions/CONTINUATION_20260908.md) preserves unsuccessful directions. '
              'All heavy computation, tests, fitting audits and summaries ran in Slurm. '
              'Model revisions, source-weight hashes, input hashes, raw paired losses and maps are retained '
              'alongside each source report. The three older primary maps replayed exactly.', '',
              '```json', json.dumps(s, indent=2), '```']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(s, indent=2))


if __name__ == '__main__': main()
