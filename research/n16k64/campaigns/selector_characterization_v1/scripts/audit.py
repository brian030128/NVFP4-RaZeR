"""T0: bounded single-thread input audit and outcome-blind protocol freeze."""
import ast
import platform
import shutil
import subprocess
import sys
import torch
from common import *

def main():
    start=now();s=load(OUT/'TASK_STATUS.json');print('T0 current:',s['tasks'][0]['status'],flush=True)
    audit=load(REPO/'research/n16k64/validation/CURRENT_EVIDENCE_AUDIT.json')
    inv=[];inputs=[];overlaps=[];fail=[];models={};sealed={}
    seal=PRIMARY/'runs/V83_validate_artifacts_attempt7/artifact_validation/ARTIFACT_MANIFEST.sha256'
    assert sha(seal)=='432d922a49751603740622a084a665983fb337295048112e68311cd8110e19ee'
    for line in seal.read_text().splitlines():
        h,p=line.split(maxsplit=1);sealed[p.lstrip('*')]=h
    def artifact(path,kind,model,draw,expected=None):
        rel=path.relative_to(PRIMARY).as_posix();h=sha(path)
        if expected is not None:assert h==expected,(rel,'recorded digest mismatch')
        sh=sealed.get(rel)
        if sh is not None:assert h==sh,(rel,'campaign seal mismatch')
        inv.append(dict(artifact_id=f'{model}/{draw}/{kind}',model=model,draw=draw,campaign=PRIMARY.name,kind=kind,
                        path=path.relative_to(REPO).as_posix(),sha256=h,bytes=path.stat().st_size,
                        source_commit='historical per-file source manifests; not current HEAD',
                        available=True,schema_verified=True,reuse_decision='verified input',sealed_member=sh is not None))
        return h
    for row in audit['five_draw_inputs']:
        model,draw=row['model'],row['draw'];run=PRIMARY/Path(row['calibration_manifest']).parents[1]
        cr=load(run/'calibration/calibration_report.json');lr=load(run/'launch_record.json')
        assert cr['status']=='complete' and lr['status']=='complete' and not lr.get('invalid_gpu_cotenancy')
        cm=run/'calibration/calibration_manifest.json';cmhash=artifact(cm,'sequence_manifest',model,draw,row['calibration_manifest_sha256'])
        reporthash=artifact(run/'calibration/calibration_report.json','calibration_report',model,draw)
        mp=run/'calibration/moments/moments_full.pt';mh=artifact(mp,'moments',model,draw,cr['moment_files']['moments_full.pt'])
        state=torch.load(mp,map_location='cpu',weights_only=True,mmap=True)
        assert set(state['names'])==set(state['n16'])
        maps={};maprows=[]
        for r in row['natural_maps']:
            f=PRIMARY/r['path'];h,a=mapread(f);artifact(f,r['rule']['rule']+'_map',model,draw,r['sha256'])
            assert h['type_block']==[16,64]
            assert all(h['model'][k]==cr['spec'][k] for k in ['model_id','revision'])
            assert h['model']['tokenizer_revision']==cr['spec']['revision']
            maps[r['rule']['rule']]=a
            maprows.append(dict(path=f.relative_to(REPO).as_posix(),sha256=sha(f),payload_sha256=maskhash(a),rule=r['rule']['rule']))
        for name in state['names']:
            st=state['n16'][name];assert st['n']==128
            mc,sc=moments(st,'ce');mk,sk=moments(st,'kl');ce=mc+3*sc<0;kl=mk+3*sk<0
            for rule,b in [('ce_only',ce),('kl_only',kl),('ce_kl',ce&kl)]:assert np.array_equal(b,maps[rule][name].ravel()),(model,draw,name,rule)
        cal=load(cm);docs={v['document_sha256'] for domain in ['math','code'] for v in cal[domain]['documents']}
        key=dict(spec=cr['spec'],module_manifest_sha256=cr['module_manifest_sha256'],candidate_weight_sha256=cr['candidate_weight_sha256'],
                 activation=cr['activation_quantizer'],baseline=cr['weight_baseline'],alternative=cr['alternative'],
                 tokenizer_manifest_sha256=cr['tokenizer_manifest_sha256'],freeze=cr['freeze_sha256'])
        assert cr['protocol_id'] in ['aligned-primary','aligned-robustness']
        models.setdefault(model,key);assert key==models[model],(model,draw,'cross-draw identity mismatch')
        raw=run/'calibration/moments/raw_scores_full.pt'
        inputs.append(dict(model=model,draw=draw,run=run.relative_to(REPO).as_posix(),moments=mp.relative_to(REPO).as_posix(),
                           moments_sha256=mh,maps=maprows,calibration_manifest=cm.relative_to(REPO).as_posix(),calibration_manifest_sha256=cmhash,
                           token_hashes=cr['sequence_token_sha256'],document_hashes=sorted(docs),report_sha256=reporthash,
                           historical_protocol_label=cr['protocol_id'],sample_unit='sequence',n=128,ddof=1,equal_weight=True,compatibility_key=digest(key),
                           full_n8_scores_available=raw.exists(),full_n8_scores_path=raw.relative_to(REPO).as_posix() if raw.exists() else None,
                           full_n8_scores_bytes=raw.stat().st_size if raw.exists() else 0,
                           raw_expected_sha256=cr['moment_files'].get('raw_scores_full.pt'),
                           original_calibration_gpu_hours=lr['gpu_hours']))
        print('verified',model,draw,'all 3 maps match moments',flush=True)
        del state,maps
    for model in MODELS:
        rows=[r for r in inputs if r['model']==model]
        for i,a in enumerate(rows):
            for b in rows[i+1:]:overlaps.append(dict(model=model,draw_a=a['draw'],draw_b=b['draw'],document_intersection=len(set(a['document_hashes'])&set(b['document_hashes'])),sequence_intersection=len(set(a['token_hashes'])&set(b['token_hashes']))))
    code=REPO/'research/n16k64/software/primary/campaign'
    tree=ast.parse((code/'data.py').read_text());datasets={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['WIKI','C4','MATH','CODE']:
            datasets[node.targets[0].id]=ast.literal_eval(node.value)
    protocol=dict(protocol_version='selector_characterization_v1',status='FROZEN_DESIGN_REQUIRES_GPU_PREFLIGHT_AND_PILOT',freeze_timestamp=start,
        source=dict(repo='brian030128/NVFP4-RaZeR',branch='research/mixfp4-n16k64',commit='db63419cc33b2bbbda2117aad636435a2956532d',
                    handoff_sha256='54a5eadeceafab590cb670f79c7f8a981f0132531d7e33c8e23206777a4e0e74',
                    historical_code_sha256={x.name:sha(x) for x in [code/'tiles.py',code/'quant.py',code/'data.py',code/'policies.py',code/'calibrate.py',code/'evaluate_ppl.py']}),
        models=models,calibration=dict(draws=DRAWS,inputs=inputs,datasets=datasets,sample_unit='128 equal-weight sequences of 512 tokens; CE/KL mean over 511 shifted targets',
                                      estimator='FP64 sum/sumsq; unbiased sample variance ddof=1; SE=sqrt(var/n); clamp negative roundoff variance to zero'),
        objectives=dict(k=3,threshold='strict upper<0, float64, no tolerance',matched_ranking='upper ascending, canonical module then row-major tile ID; entire eligible universe',
                        established_equivalence='followup restricted to own-pass pool; full-universe top-K is identical because joint quota <= number of own-pass candidates',
                        matched_quota='same model/draw/module natural joint count',invalid='nonfinite mean/SE abort module, never favorable'),
        stability=dict(primary_ranking='max(CE mean+3SE, KL mean+3SE) ascending; canonical frozen followup combined margin',
                       secondary_ranking='max(CE mean/SE, KL mean/SE) diagnostic only',zero_se='mu<0:-inf; mu>0:+inf; mu=0:0',
                       quota='module min joint count across five draws',random_repeats=1000,random_seed=20260922,
                       random_algorithm='exact hypergeometric intersection law, independent across modules',global_fractions=[.001,.005,.01,.02,.05]),
        granularity=dict(N=[8,16,32,64,128,256],K=64,draw='seed0',scale_block=[1,16],no_refit=True,no_reorder=True,
                         candidate='same frozen BF16 FourOverSix/E0M3 alpha1 candidates; per-sequence signed N8 scores summed before moments',
                         incomplete_parent='unsupported, no zero padding',anchor_tolerance=dict(map_mismatch_tiles=0,mean_nll_reuse_absolute=1e-10),
                         all_new_N_backend='software_emulated when evaluated; analytical_only otherwise'),
        evaluation=dict(datasets=datasets,length=2048,c4_windows=256,wiki_windows='all complete frozen token windows',
                        metric='sum cluster NLL sums / sum valid tokens',cluster='C4 document; Wiki article, inherited mixed-window cluster convention',
                        bootstrap_repeats=2000,bootstrap_seed=20260921,CI='pointwise percentile',
                        multiplicity='No significance claims; pointwise exploratory intervals. Optional Holm family: 2 matched contrasts x 5 draws per model/corpus.',
                        secondary_accuracy='primary draw baseline/joint/CE-matched/KL-matched eight primary tasks; no GPU without scheduler access; gaps explicit'),
        resources=dict(max_concurrent_gpu_jobs=3,gpu_policy='User explicitly supersedes H200/Slurm: local RTX A6000 or RTX 6000 Ada, <=3 physical GPUs total, homogeneous devices within run, owner/PID check before/during/after, fail closed on foreign/unresolved usage',
                       estimated_gpu_hours='to refine after reuse inventory; historical calibration rates in inputs; missing evaluation jobs require permitted pilot',
                       planned_local_analysis_threads=1,estimated_new_disk_bytes=2_000_000_000),optional_default='NOT_RUN')
    if (OUT/'FROZEN_PROTOCOL.yaml').exists():
        old=load(OUT/'FROZEN_PROTOCOL.yaml');protocol['freeze_timestamp']=old['freeze_timestamp'];assert protocol==old,'frozen protocol changed'
    else:jsonout(OUT/'FROZEN_PROTOCOL.yaml',protocol)
    jsonout(OUT/'results/INPUTS.json',inputs);csvout(OUT/'artifact_inventory.csv',inv);csvout(OUT/'results/calibration_overlap.csv',overlaps)
    env=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,scheduler={k:shutil.which(k) for k in ['sbatch','srun','squeue']},
             source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip(),gpu_hours=0)
    jsonout(OUT/'results/ENVIRONMENT.json',env)
    (OUT/'DATA_AUDIT.md').write_text('# T0 資料稽核\n\n三模型五 draws 的 45 張 N16 natural maps 均重新讀取並與各自 128-sequence FP64 moments 逐 tile 比對一致。\n'
      'maps、moments、manifest 對照既有紀錄 SHA；未重新載入模型或驗算完整 candidate tensors。模型/Tokenizer revision、候選 hash、資料 revision 與每 sequence token hash 見 FROZEN_PROTOCOL.yaml。\n\n'
      'T1 與 T2 map/veto CPU 分析可做。seed0 完整逐 N8 sequence scores 目前僅 Qwen 存在；Llama/Mistral 僅 sampled scores，不能重建全域 parent SE。CE–KL cross moments 不是 children covariance。\n\n'
      '使用者已明確取代 H200/Slurm 限制：本機 A6000/Ada 合計最多三卡、禁止 foreign co-tenancy，無卡則等待。新結果須保留執行身份並通過 anchor。不得以現有 summary 當作新的 150/36 cells。\n\n'
      'seed0 protocol label 為 aligned-primary，draw1–4 為 aligned-robustness；此標籤差異保留。其模型/候選/activation/固定 protocol seal 等實質身份欄位逐項相符，不以標籤相等冒充比較身份。\n\n'
      '來源內容保留唯讀。calibration overlap 已另表；有文件 hash 不等於證明 draws 獨立。evaluation doc/token identity 於 T2 重用稽核另驗。\n')
    outputs=[OUT/'FROZEN_PROTOCOL.yaml',OUT/'artifact_inventory.csv',OUT/'results/INPUTS.json',OUT/'DATA_AUDIT.md',OUT/'results/calibration_overlap.csv',OUT/'results/ENVIRONMENT.json']
    log('T0','CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python scripts/audit.py',start,outputs)
    checkpoint('T0','done',[str(p.relative_to(OUT)) for p in outputs],next_action='T1：讀取 TASK_STATUS，執行三模型五 maps 的 counts/set/rank/null CPU 分析。T2/T3 先做 PID-owner preflight 與最小 pilot。')

if __name__=='__main__':main()
