"""Scientific static plots of frozen training curves; no model execution."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/loss_diagnostics_20261014'
COLORS = {'total':'#202938','pixel':'#1769aa','bottleneck':'#d45a27','weighted_gradient':'#369778','weighted_kl':'#9c6ab0'}


def main():
    report = json.loads((OUT/'diagnostics.json').read_text(encoding='utf-8'))
    models = report['models']
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.facecolor':'white', 'figure.facecolor':'white'})
    for kind in dict.fromkeys(m['kind'] for m in models):
        group = [m for m in models if m['kind']==kind]
        fig, axes = plt.subplots(2,4,figsize=(16,8),squeeze=False)
        for ax,m in zip(axes.flat,group):
            blocks=m['blocks100'];x=[b['epoch'] for b in blocks]
            for key in COLORS:
                y=np.array([b[key] for b in blocks])
                if np.any(y != 0):
                    ax.plot(x,y,color=COLORS[key],lw=1.35,label=key)
            ax.set_yscale('symlog',linthresh=1e-4)
            ax.set_title(m['model']);ax.set_xlim(0,5);ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (100-update weighted mean)');ax.grid(alpha=.18)
            for e in range(1,5):ax.axvline(e,color='#dddddd',lw=.7,zorder=0)
        for ax in list(axes.flat)[len(group):]:ax.axis('off')
        handles, labels = [], []
        for ax in axes.flat:
            h,l=ax.get_legend_handles_labels()
            for a,b in zip(h,l):
                if b not in labels:handles.append(a);labels.append(b)
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=5,frameon=False)
        fig.suptitle(kind+' | Actual optimized terms; zero-weight representation diagnostic excluded',fontsize=13,y=.995)
        fig.tight_layout(rect=(0,0,1,.90))
        fig.savefig(OUT/(kind+'_loss_components.png'),dpi=155);plt.close(fig)
    chosen=['sd_C_affine','sd_C','vq_G','vq_drifting_G_asinh','vqgan_pretrained_C','grouped_g8k256_C_power']
    fig, axes=plt.subplots(2,3,figsize=(15,8))
    for ax,name in zip(axes.flat,chosen):
        m=next(m for m in models if m['model']==name)
        arr=np.load(OUT/'curves'/(name+'.npz'))
        # Show ALL individual updates; no smoothing can hide transient failures.
        ax.plot(arr['epoch'],arr['pixel'],lw=.35,alpha=.22,color='#1769aa',label='Per-update reconstruction loss')
        ax.plot([b['epoch'] for b in m['blocks100']],[b['pixel'] for b in m['blocks100']],color='#bc3b32',lw=1.6,label='100-update weighted mean')
        ax.set_yscale('log');ax.set_xlim(0,5);ax.set_xlabel('Epoch');ax.set_ylabel('Normalized reconstruction MSE')
        ax.set_title(name);ax.grid(alpha=.18)
        for e in range(1,5):ax.axvline(e,color='#dddddd',lw=.7,zorder=0)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,frameon=False)
    fig.suptitle('Training loss anomalies: one stable reference and five problematic runs',fontsize=14,y=.995)
    fig.tight_layout(rect=(0,0,1,.90));fig.savefig(OUT/'anomalies_overview.png',dpi=165);plt.close(fig)
    # Independently compare equal-image training error and globally pixel-weighted validation.
    fig, ax=plt.subplots(figsize=(15,6))
    order=sorted(models,key=lambda m:m['validation']['pixel10']['raw']['pixel_weighted_rmse'])
    x=np.arange(len(order))
    train=[m['residual_std_C']*np.sqrt(m['windows']['pixel']['last1000']['mean']) for m in order]
    val=[m['validation']['pixel10']['raw']['pixel_weighted_rmse'] for m in order]
    ax.scatter(x,train,label='Last 1000 train updates: sqrt(equal-image MSE), C',s=23,color='#1769aa')
    ax.scatter(x,val,label='Full validation: pixel-weighted RMSE, C',s=23,color='#d45a27')
    baseline=order[0]['validation']['pixel10']['observation_mean_fill_baseline']['pixel_weighted_rmse']
    ax.axhline(baseline,ls='--',color='#369778',label=f'Observed pixels + oracle mean fill: {baseline:.4f} C')
    ax.set_xticks(x,[m['model'] for m in order],rotation=75,ha='right',fontsize=7)
    ax.set_ylabel('Celsius error (different aggregation; no exact equality implied)')
    ax.set_title('Pixel10: 20 of 39 final models are worse than the recorded no-training baseline')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15);fig.tight_layout()
    fig.savefig(OUT/'train_vs_validation.png',dpi=170);plt.close(fig)
    print('Saved 7 scientific PNG figures for all 39 models')


if __name__=='__main__':main()
