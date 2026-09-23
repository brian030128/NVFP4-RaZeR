"""Secondary accuracy points; preserve incomplete model panels explicitly."""
from common import *
from secondary_accuracy import TASKS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    start=now();source=OUT/'results/secondary_accuracy_current.csv'
    audit=load(OUT/'results/SECONDARY_ACCURACY_TABLE_AUDIT.json')
    assert audit['source_hashes']['results/secondary_accuracy_current.csv']==sha(source)
    rows=list(csv.DictReader(source.open()));lookup={(r['model'],r['policy'],r['task']):r for r in rows}
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','svg.hashsalt':'selector_characterization_accuracy_v1'})
    fig,axes=plt.subplots(1,3,figsize=(11,4.5),sharex=True,sharey=True,layout='constrained')
    for ax,model in zip(axes,MODELS):
        count=0
        for j,(policy,label) in enumerate([('joint','Joint (historical)'),('ce_matched','CE matched (new)'),('kl_matched','KL matched (new)')]):
            x=[];y=[]
            for i,task in enumerate(TASKS):
                r=lookup[model,policy,task]
                if r['status']=='coverage_gap':continue
                assert r['status'] in ['sealed_report_reuse','validated_new']
                x.append(100*(float(r['value'])-float(lookup[model,'baseline',task]['value'])));y.append(i+(j-1)*.19)
            count+=len(x)
            if x:ax.scatter(x,y,s=23,marker=['o','s','^'][j],label=label)
        ax.axvline(0,color='grey',lw=.7);ax.grid(axis='x',alpha=.2)
        ax.set_title(f'{model}: {count}/24 policy/task points')
        if count<24:ax.text(.02,.02,'CE/KL matched missing',transform=ax.transAxes,fontsize=8)
        ax.set_xlabel('Accuracy change vs FourOverSix (pp)');ax.set_yticks(range(8),TASKS)
    axes[0].invert_yaxis();axes[1].legend(loc='best',fontsize=7)
    fig.suptitle('Fixed-seed0 full8 accuracy: descriptive points only, no new CI or non-inferiority claim\n'
                 'Task metrics follow the frozen evaluator; MMLU is question-weighted within task.',fontsize=10)
    outputs=[]
    for suffix in ['svg','png','pdf']:
        path=OUT/'figures'/f'secondary_accuracy.{suffix}';fig.savefig(path,dpi=180,bbox_inches='tight');outputs.append(path)
    plt.close(fig)
    manifest=OUT/'figures/ACCURACY_SOURCE_MANIFEST.json'
    jsonout(manifest,dict(script_sha256=sha(Path(__file__)),sources={str(source.relative_to(OUT)):sha(source),
        'results/SECONDARY_ACCURACY_TABLE_AUDIT.json':sha(OUT/'results/SECONDARY_ACCURACY_TABLE_AUDIT.json')},
        outputs={str(p.relative_to(OUT)):sha(p) for p in outputs}))
    log('accuracy-plot','python scripts/accuracy_plot.py',start,outputs+[manifest])

if __name__=='__main__':main()
