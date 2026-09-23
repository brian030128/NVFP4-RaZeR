"""Publication assets from tidy CSVs; missing NLL stays missing, no interpolation."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(name):return list(csv.DictReader((OUT/'results'/name).open()))
def save(fig,name):
    for suffix in ['svg','png','pdf']:
        fig.savefig(OUT/'figures'/f'{name}.{suffix}',bbox_inches='tight',dpi=180)
    plt.close(fig)
def main():
    start=now();(OUT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','svg.hashsalt':'selector_characterization_v1'})
    counts=read('map_counts.csv');pairs=read('map_pairs.csv');freq=read('selection_frequency.csv');rank=read('ranking_overlap.csv')
    print('columns',list(counts[0]),list(freq[0]),flush=True)
    fig,axs=plt.subplots(1,3,figsize=(11,3),layout='constrained')
    for ax,m in zip(axs,MODELS):
        rr=[r for r in counts if r['model']==m and r['module']=='ALL']
        ax.bar([r['draw'] for r in rr],[100*float(r['density']) for r in rr]);ax.set_title(m+' N16 joint k=3');ax.tick_params(axis='x',rotation=45);ax.set_ylabel('Selected tiles (%)')
    save(fig,'density')
    fig,axs=plt.subplots(1,3,figsize=(11,3),layout='constrained')
    for ax,m in zip(axs,MODELS):
        a=np.eye(5)
        for r in pairs:
            if r['model']!=m:continue
            i,j=DRAWS.index(r['draw_a']),DRAWS.index(r['draw_b']);a[i,j]=float(r['containment_a_to_b']);a[j,i]=float(r['containment_b_to_a'])
        im=ax.imshow(a,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(5),DRAWS,rotation=45);ax.set_yticks(range(5),DRAWS);ax.set_title(m+' N16 joint');ax.set_xlabel('To draw');ax.set_ylabel('From draw')
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',color='white' if a[i,j]<.6 else 'black',fontsize=7)
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Intersection / from-draw count',shrink=.75);save(fig,'containment')
    fig,axs=plt.subplots(1,3,figsize=(11,3),layout='constrained')
    for ax,m in zip(axs,MODELS):
        rr=[r for r in freq if r['model']==m and r['module']=='ALL'];ax.bar([int(r['frequency']) for r in rr],[int(r['tiles']) for r in rr]);ax.set_yscale('log');ax.set_xticks(range(6));ax.set_xlabel('Selection frequency / 5');ax.set_ylabel('Tiles (log scale)');ax.set_title(m+' N16 joint k=3')
    save(fig,'selection_frequency')
    fig,axs=plt.subplots(1,3,figsize=(11,3),layout='constrained')
    for ax,m in zip(axs,MODELS):
        for ranking in ['joint_upper_primary','joint_z_diagnostic']:
            rr=[r for r in rank if r['model']==m and r['ranking']==ranking and r['scope'].startswith('global')]
            # Derive the fraction from exact K/U, independent of display labels.
            points={}
            for r in rr:points.setdefault(float(r['K_a'])/float(r['U']),[]).append(float(r['jaccard']))
            xx=sorted(points)
            if xx:ax.plot(np.array(xx)*100,[np.mean(points[x]) for x in xx],marker='o',label=ranking)
        ax.set_xlabel('Fixed global budget (%)');ax.set_ylabel('Mean of 10 pair Jaccards');ax.set_title(m+' N16 ranking');ax.set_xscale('log');ax.legend(fontsize=6)
    save(fig,'ranking_sensitivity')
    quality=read('quality_results.csv')
    policies=['ce_natural','kl_natural','joint','ce_matched','kl_matched']
    labels=['CE natural','KL natural','Joint','CE matched quota','KL matched quota']
    colors=['tab:blue','tab:orange','black','tab:green','tab:purple']
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for j,m in enumerate(MODELS):
        for i,corpus in enumerate(['wiki','c4']):
            ax=axs[i,j];available=0
            for k,(policy,label,color) in enumerate(zip(policies,labels,colors)):
                offset=(k-2)*.13
                for d,draw in enumerate(DRAWS):
                    rows=[r for r in quality if r['model']==m and r['corpus']==corpus and r['draw']==draw and r['policy']==policy
                          and r['status'] in ['validated_new','validated_reuse']]
                    assert len(rows)<=1,(m,corpus,draw,policy)
                    if not rows:continue
                    r=rows[0];available+=1;point=float(r['delta_nll']);lo=float(r['ci_low']);hi=float(r['ci_high'])
                    # Draw interval independently of the point: percentile
                    # intervals need not mathematically contain the estimate.
                    ax.vlines(d+offset,lo,hi,color=color,linewidth=.8,alpha=.7)
                    ax.plot(d+offset,point,marker='s' if r['status']=='validated_new' else 'o',color=color,ms=3.5,linestyle='none')
                if i==0 and j==0:ax.plot([],[],color=color,marker='o',linestyle='none',label=label,ms=4)
            ax.axhline(0,color='grey',lw=.7,ls='--')
            ax.set_xticks(range(5),DRAWS,rotation=30);ax.set_xlim(-.5,4.5)
            ax.set_title(f'{m} / {corpus}: {available}/25 verified cells')
            ax.set_ylabel('Delta NLL vs FourOverSix (lower is better)')
            ax.set_xlabel('Frozen calibration draw')
    axs[0,0].legend(fontsize=6,loc='best')
    fig.suptitle('N16K64 k=3 objective comparison: 2,000 paired-cluster bootstrap, pointwise 95% CI\n'
                 'Circles: verified reuse; squares: new verified evaluation. Missing cells are not plotted; no simultaneous-coverage claim.',fontsize=10)
    save(fig,'objective_quality')
    contrasts=read('quality_contrasts.csv')
    fig,axs=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for j,m in enumerate(MODELS):
        for i,corpus in enumerate(['wiki','c4']):
            ax=axs[i,j];available=0
            for name,label,color,offset in [('ce_matched-minus-joint','CE matched − joint','tab:green',-.08),
                                           ('kl_matched-minus-joint','KL matched − joint','tab:purple',.08)]:
                for d,draw in enumerate(DRAWS):
                    rr=[r for r in contrasts if r['model']==m and r['corpus']==corpus and r['draw']==draw
                        and r['contrast']==name and r['status'] in ['validated_new','validated_reuse']]
                    assert len(rr)<=1
                    if not rr:continue
                    r=rr[0];available+=1
                    ax.vlines(d+offset,float(r['ci_low']),float(r['ci_high']),color=color,lw=1)
                    ax.plot(d+offset,float(r['delta_nll']),marker='o',ms=4,color=color,linestyle='none')
                if i==0 and j==0:ax.plot([],[],marker='o',color=color,linestyle='none',label=label,ms=4)
            ax.axhline(0,color='grey',lw=.7,ls='--')
            ax.set_title(f'{m} / {corpus}: {available}/10 paired contrasts')
            ax.set_xticks(range(5),DRAWS,rotation=30);ax.set_xlim(-.4,4.4)
            ax.set_xlabel('Frozen calibration draw');ax.set_ylabel('Delta NLL: matched − joint')
    axs[0,0].legend(fontsize=7)
    fig.suptitle('N16K64 k=3: exact joint module quotas; negative favors matched alternative\n'
                 '2,000 paired-cluster bootstrap, pointwise 95% CI; not multiplicity-adjusted or population-risk inference.',fontsize=10)
    save(fig,'matched_vs_joint')
    diagnostics=read('effect_diagnostics.csv')
    for objective in ['ce','kl']:
        fig,axs=plt.subplots(2,3,figsize=(11,6))
        for j,m in enumerate(MODELS):
            rr=[r for r in diagnostics if r['model']==m and r['objective']==objective]
            for stratum in ['selected','near_threshold','rejected','random']:
                sub=[r for r in rr if r['stratum']==stratum]
                axs[0,j].errorbar([float(r['predicted']) for r in sub],[float(r['actual']) for r in sub],yerr=[1.96*float(r['actual_se']) for r in sub],fmt='.',alpha=.55,label=stratum,elinewidth=.4)
            axs[0,j].axhline(0,color='grey',lw=.6);axs[0,j].axvline(0,color='grey',lw=.6);axs[0,j].set_title(m+' N16 '+objective.upper());axs[0,j].set_xlabel('Predicted first-order mean');axs[0,j].set_ylabel('Measured finite effect ±1.96 SE');axs[0,j].legend(fontsize=6)
            vals=[float(r['effect_over_se']) for r in rr if r['effect_over_se']];axs[1,j].hist(vals,bins=25);axs[1,j].set_xlabel('Finite effect / sequence SE');axs[1,j].set_ylabel('All sampled tiles');axs[1,j].axvline(-1.96,color='grey',ls='--');axs[1,j].axvline(1.96,color='grey',ls='--')
        fig.suptitle('Same calibration sample; descriptive, stratified, uncorrected intervals');fig.tight_layout();save(fig,'individual_'+objective)
    q=read('granularity_quality.csv');s=read('granularity_results.csv')
    fig,axs=plt.subplots(2,3,figsize=(11,6))
    for j,m in enumerate(MODELS):
        for c in ['wiki','c4']:
            rr=sorted([r for r in q if r['model']==m and r['corpus']==c and r['status'] in ['validated_reuse','validated_new']],key=lambda r:int(r['N']))
            axs[0,j].plot([int(r['N']) for r in rr],[float(r['delta_nll_vs_baseline']) for r in rr],marker='o',label=c)
        axs[0,j].set_title(m+' seed0 joint k=3');axs[0,j].set_ylabel('Measured ΔNLL vs FourOverSix');axs[0,j].legend()
        missing=[r for r in q if r['model']==m and r['status'] not in ['validated_reuse','validated_new']]
        if missing:axs[0,j].text(.95,.08,f'{len(missing)}/12 N/corpus cells unavailable',transform=axs[0,j].transAxes,ha='right',fontsize=7)
        rr=sorted([r for r in s if r['model']==m and r['objective']=='ce'],key=lambda r:int(r['N']))
        axs[1,j].plot([int(r['N']) for r in rr],[float(r['ownership_regret']) for r in rr],marker='o');axs[1,j].set_ylabel('CE ownership regret (surrogate)')
        for ax in axs[:,j]:ax.set_xscale('log',base=2);ax.set_xticks([8,16,32,64,128,256],[8,16,32,64,128,256]);ax.set_xlabel('N (K=64; same candidates/scales)')
    fig.tight_layout();save(fig,'granularity_partial')
    sources=['map_counts.csv','map_pairs.csv','selection_frequency.csv','ranking_overlap.csv',
             'quality_results.csv','quality_contrasts.csv','effect_diagnostics.csv','granularity_quality.csv','granularity_results.csv']
    jsonout(OUT/'figures/SOURCE_MANIFEST.json',dict(plot_script_sha256=sha(__file__),
        sources={name:sha(OUT/'results'/name) for name in sources},
        qualification='Figures show only verified rows for measured outcomes; exact analytical quantities are labeled surrogates.'))
    log('T5-plots','python scripts/plot_results.py',start,sorted((OUT/'figures').glob('*')))

if __name__=='__main__':main()
