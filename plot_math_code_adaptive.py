"""Standalone count and transfer-sensitivity figures from completed reports."""
import json
import os
from pathlib import Path


def plot(summary_path,out):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(os.environ['HF_HOME'])/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    summary=json.loads(Path(summary_path).read_text()); out=Path(out)
    labels={'qwen4b':'Qwen3-4B','llama8b':'Llama-3.1-8B','qwen27b':'Qwen3.8-27B'}
    colors={'math':'#0072B2','code':'#D55E00','math_code':'#009E73'}
    fig,axes=plt.subplots(3,3,figsize=(14,11),constrained_layout=True)
    for row,(model,label) in enumerate(labels.items()):
        r=json.loads(Path(summary['source_reports'][model]).read_text())
        for source,color in colors.items():
            sizes=(16,32,64,128) if source=='math_code' else (16,32,64)
            for mode,style,marker in [('adaptive','-','o'),('fixed256','--','s')]:
                policies=[f'{mode}_{source}{n}' for n in sizes]
                if any(p not in r['evaluation'] for p in policies): continue
                counts=[r['block_statistics'][p]['selected_blocks'] for p in policies]
                axes[row,0].plot(sizes,counts,color=color,linestyle=style,marker=marker,markersize=4)
                for col,domain in [(1,'wiki'),(2,'c4')]:
                    cs=[r['contrasts'][p][domain] for p in policies]
                    axes[row,col].errorbar(sizes,[c['mean_nll'] for c in cs],yerr=[c['two_se'] for c in cs],
                        color=color,linestyle=style,marker=marker,markersize=4,capsize=2,linewidth=1.2)
        for col in range(3):
            ax=axes[row,col]
            ax.set_xscale('log',base=2); ax.set_xticks((16,32,64,128),('16','32','64','128'))
            ax.set_xlabel('Calibration sequences'); ax.grid(alpha=.2)
            ax.set_title(f'{label}: '+('E0M3 type blocks' if col==0 else ('WikiText-2' if col==1 else 'C4')))
            if col:
                ax.axhline(0,color='gray',linewidth=.8); ax.set_ylabel('ΔNLL versus FourOverSix')
            else:
                ax.set_yscale('symlog',linthresh=1); ax.set_ylabel('Selected 8×64 blocks (symlog)')
                ax.set_yticks((0,1,2,4,8,16,64,256),('0','1','2','4','8','16','64','256'))
        axes[row,0].set_ylim(bottom=-.1)
    handles=[Line2D([0],[0],color=c,label=s.replace('_','+')) for s,c in colors.items()]
    handles += [Line2D([0],[0],color='black',linestyle=style,marker=marker,label=mode)
                for mode,style,marker in [('Adaptive','-','o'),('Fixed-256','--','s')]]
    fig.legend(handles=handles,loc='outside lower center',ncol=5,frameon=False)
    fig.suptitle('Math/code-only calibration; causal WikiText/C4 evaluation\n'
                 'Error bars: descriptive evaluation-window ±2SE; no calibration-seed replication',fontsize=12)
    for ext in ('pdf','svg','png'): fig.savefig(out/f'sensitivity.{ext}',dpi=180)
    svg=out/'sensitivity.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    (out/'plot_environment.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__),indent=2)+'\n')
