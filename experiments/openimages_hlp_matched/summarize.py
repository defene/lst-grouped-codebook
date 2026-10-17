"""Summaries and scientific figures from exact census moments."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from color_analyze import OUT,PREP,dump

plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'figure.dpi':140})

def load(kind,out):
    accum={}
    for p in sorted((out/'chunks').glob(kind+'_*.npz')):
        with np.load(p) as z:
            for k in z.files:accum.setdefault(k,[]).append(z[k])
    return {k:np.concatenate(v) for k,v in accum.items()}

def family(data,prefix,bands=None):
    x={k[len(prefix)+1:]:v for k,v in data.items() if k.startswith(prefix+'_')}
    if bands is not None:
        for k in ('mean','edge','at_zero','at_one','minimum','maximum'):x[k]=x[k][:,bands]
        x['cov']=x['cov'][:,bands][:,:,bands]
        x['sampled']=x['sampled'][:,:,bands]
    return x

def eigstats(c):
    e=np.maximum(np.linalg.eigvalsh(c),0)[::-1];p=e/e.sum()
    return {'eigenvalues':e.tolist(),'pc1_share':float(p[0]),'effective_rank':float(np.exp(-np.sum(p[p>0]*np.log(p[p>0]))))}

def summarize(x,tiles=None):
    mean=x['mean']; m=mean.mean(0);d=mean-m
    between=d.T@d/len(mean);within=x['cov'].mean(0);total=between+within
    tr=np.trace;diag=np.diag
    qs=[0,.05,.25,.5,.75,.95,1]
    rms=np.sqrt(np.trace(x['cov'],axis1=1,axis2=2)/mean.shape[1])
    std=np.sqrt(diag(total));corr=total/np.outer(std,std)
    relative_rms=np.sqrt((np.diagonal(x['cov'],axis1=1,axis2=2)/(std**2)).mean(1))
    ws=np.sqrt(diag(within));wc=within/np.outer(ws,ws)
    result=dict(n=len(mean),valid_pixels=int(x['pixels'].sum()),mean=m.tolist(),
        covariance_total=total.tolist(),covariance_within=within.tolist(),covariance_between=between.tolist(),
        total_std=std.tolist(),pooled_within_std=ws.tolist(),between_image_std=np.sqrt(diag(between)).tolist(),
        within_variance_share=float(tr(within)/tr(total)),between_variance_share=float(tr(between)/tr(total)),
        within_share_per_channel=(diag(within)/diag(total)).tolist(),
        correlation_total=corr.tolist(),correlation_within=wc.tolist(),
        pca_total=eigstats(total),pca_within=eigstats(within),
        spatial_rms_quantiles=dict(zip(map(str,qs),np.quantile(rms,qs).tolist())),
        relative_spatial_rms_quantiles=dict(zip(map(str,qs),np.quantile(relative_rms,qs).tolist())),
        neighbor_rms=float(np.sqrt(x['edge'].mean())),
        boundary_zero_fraction=x['at_zero'].mean(0).tolist(),boundary_one_fraction=x['at_one'].mean(0).tolist(),
        min=x['minimum'].min(0).tolist(),max=x['maximum'].max(0).tolist())
    if mean.shape[1]==3:
        q=np.ones(3)/np.sqrt(3)
        result['achromatic_axis_share_within']=float(q@within@q/tr(within))
        result['achromatic_axis_share_total']=float(q@total@q/tr(total))
        # Ratio of sums on a pixel, not CIE chromaticity. Each nonblack image gets equal weight.
        pixels=x['sampled'];s=pixels.sum(-1);ok=s>.01
        rg=pixels[:,:,:2]/np.maximum(s[:,:,None],1e-12)
        weights=ok/np.maximum(ok.sum(1,keepdims=True),1)
        hist,*_=np.histogram2d(rg[:,:,0][ok],rg[:,:,1][ok],bins=64,range=((0,1),(0,1)),weights=weights[ok])
        hist/=hist.sum();result['rg_histogram']=hist.tolist()
        distance=np.sqrt(((rg-1/3)**2).sum(-1))
        result['rg_distance_from_gray_mean']=float((distance*weights).sum()/np.count_nonzero(ok.any(1)))
        result['rg_excluded_sample_pixel_fraction']=float(1-ok.mean())
        result['rg_nonblack_images']=int(ok.any(1).sum())
    if tiles is not None:
        names,inv,n=np.unique(tiles,return_inverse=True,return_counts=True)
        tm=np.zeros((len(names),mean.shape[1]));np.add.at(tm,inv,mean);tm/=n[:,None]
        dt=tm-m;bt=(dt.T*n)@dt/len(mean)
        di=mean-tm[inv];bi=di.T@di/len(mean)
        assert np.allclose(between,bt+bi,atol=1e-9)
        result['tile_decomposition']={'tiles':len(names),'pixel_within_image_share':float(tr(within)/tr(total)),
             'image_mean_within_tile_share':float(tr(bi)/tr(total)),'tile_mean_share':float(tr(bt)/tr(total))}
    return result

def make_figures(families,summaries,out):
    labels={'oi_encoded':'自然图像 · 编码 RGB','oi_linear':'自然图像 · 线性化 RGB','hlp_rgb':'HLP · RGB 反射率','hlp_lst':'HLP · LST 温度'}
    keys=list(labels);colors=['#3366A4','#447FA7','#237F69','#D2923E']
    fig,axs=plt.subplots(1,2,figsize=(14,5.3),gridspec_kw={'width_ratios':[1,1.15]})
    ax=axs[0];v=np.array([summaries[k]['within_variance_share'] for k in keys])*100
    ax.barh(range(4),v,color='#237F69',label='图内空间变化')
    ax.barh(range(4),100-v,left=v,color='#C2C9D1',label='图间均值差异')
    for i,x in enumerate(v):
        ax.text(x/2 if x>10 else x+1,i,f'{x:.1f}%',ha='center' if x>10 else 'left',va='center',color='white' if x>10 else 'black',fontsize=11)
        ax.text(x+(100-x)/2,i,f'{100-x:.1f}%',ha='center',va='center',fontsize=11)
    ax.set_yticks(range(4),[labels[k] for k in keys]);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel('总方差的比例（%）');ax.set_title('A  颜色／温度变化主要来自哪里？',loc='left');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False)
    ax=axs[1]
    for k,c in zip(keys[:3],colors[:3]):
        std=np.asarray(summaries[k]['total_std'])
        vals=np.sqrt((np.diagonal(families[k]['cov'],axis1=1,axis2=2)/(std**2)).mean(1))
        v=np.sort(vals);idx=np.unique(np.linspace(0,len(v)-1,1000).astype(int))
        ax.plot(v[idx],(idx+1)/len(v),label=labels[k],color=c,lw=2)
    ax.set_xlim(0,2.5);ax.set_ylim(0,1);ax.set_xlabel('每图空间 RMS / 各通道全局标准差（无量纲）')
    ax.set_ylabel('累计图像比例');ax.set_title('B  相对图内变化的分布',loc='left');ax.legend(frameon=False,fontsize=10);ax.grid(alpha=.2)
    fig.suptitle(f"OpenImages {summaries['oi_encoded']['n']:,} 张 vs HLP v2 train {summaries['hlp_rgb']['n']:,} 对",fontsize=16,x=.04,ha='left')
    fig.text(.04,.01,'等图像权重；全部有效像素。B 先用各数据集的全局逐通道标准差缩放，再计算图内 RMS；未做逐图对比度拉伸。',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.92));fig.savefig(out/'variance_comparison.png');plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(14,4.8),sharex=True,sharey=True)
    for ax,k in zip(axs,keys[:3]):
        h=np.array(summaries[k]['rg_histogram'])
        im=ax.imshow(np.maximum(h.T,1e-7),origin='lower',extent=(0,1,0,1),norm=LogNorm(vmin=1e-6,vmax=.1),cmap='magma',aspect='equal')
        ax.plot([0,1],[1,0],color='white',ls=':',alpha=.6);ax.plot(1/3,1/3,'+',color='#37D3DC',ms=10)
        ax.set_title(labels[k]);ax.set_xlabel('r = R / (R+G+B)');ax.set_xlim(0,1);ax.set_ylim(0,1)
    axs[0].set_ylabel('g = G / (R+G+B)')
    fig.suptitle('RGB 通道比例分布：观察颜色方向，不把反射率当成 sRGB 色度',fontsize=15,x=.04,ha='left')
    fig.subplots_adjust(left=.06,right=.87,bottom=.19,top=.81,wspace=.2)
    cax=fig.add_axes([.9,.25,.014,.5]);fig.colorbar(im,cax=cax,label='每格概率（对数刻度）')
    fig.text(.04,.05,'每图确定性抽 64 个有效像素；R+G+B > 0.01；非黑图像等权。青色 + 为 R=G=B。三图共用坐标、网格与色标。',fontsize=10)
    fig.savefig(out/'rgb_ratio_density.png');plt.close(fig)

    fig,axs=plt.subplots(2,3,figsize=(12,8))
    for j,k in enumerate(keys[:3]):
        for i,metric in enumerate(('correlation_total','correlation_within')):
            c=np.array(summaries[k][metric]);ax=axs[i,j]
            im=ax.imshow(c,vmin=-1,vmax=1,cmap='RdBu_r')
            for a in range(3):
                for b in range(3):ax.text(b,a,f'{c[a,b]:.3f}',ha='center',va='center',color='white' if abs(c[a,b])>.65 else 'black')
            ax.set_xticks(range(3),['R','G','B']);ax.set_yticks(range(3),['R','G','B'])
            if i==0:ax.set_title(labels[k])
            if j==0:ax.set_ylabel('全体颜色变化' if i==0 else '每张图去均值后的空间残差')
    fig.suptitle('通道相关性：整体变化与图内残差分开看',fontsize=16,x=.05,ha='left')
    fig.subplots_adjust(left=.09,right=.88,bottom=.09,top=.9,hspace=.23,wspace=.22)
    cax=fig.add_axes([.91,.2,.018,.6]);fig.colorbar(im,cax=cax,label='Pearson correlation')
    fig.text(.05,.025,'全部有效像素；先计算每图矩，再按图像等权聚合。残差相关性来自聚合图内协方差，不是逐图相关系数平均。',fontsize=10)
    fig.savefig(out/'channel_correlations.png');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    progress=json.loads((OUT/'progress.json').read_text())
    if not a.partial:assert progress['status']=='complete'
    nat=load('natural',OUT);hlp=load('hlp',OUT)
    assert len(np.unique(nat['ids']))==len(nat['ids']);assert len(np.unique(hlp['ids']))==len(hlp['ids'])
    fs={'oi_encoded':family(nat,'encoded'),'oi_linear':family(nat,'linear'),
        'hlp_rgb':family(hlp,'hls',[2,1,0]),'hlp_lst':family(hlp,'lst'),'hlp_six':family(hlp,'hls')}
    stats={k:summarize(v,hlp['tiles'] if k.startswith('hlp') else None) for k,v in fs.items()}
    normal=json.loads((PREP/'prithvi/normalization.json').read_text())
    # Coordinate sensitivity: the actual HLS model uses fixed per-band training z-scores.
    scale=np.asarray(normal['hls']['std']);mu=np.asarray(normal['hls']['mean'])
    h=fs['hlp_six'];within=h['cov'].mean(0)/np.outer(scale,scale)
    m=(h['mean']-mu)/scale;between=np.cov(m,rowvar=False,bias=True)
    standardized=dict(within_variance_share=float(np.trace(within)/np.trace(within+between)),
        pca_within=eigstats(within),pca_total=eigstats(within+between))
    checks={}
    if not a.partial:
        assert len(nat['ids'])==218975 and len(hlp['ids'])==218975
        for name,mod,scale in [('hlp_six','hls',1),('hlp_lst','lst_hr',.01)]:
            st=stats[name];expected=np.asarray(normal[mod]['mean'])*scale
            exstd=np.asarray(normal[mod]['std'])*scale
            checks[name]={'mean_max_abs_error':float(np.max(np.abs(np.array(st['mean'])-expected))),
                          'std_max_abs_error':float(np.max(np.abs(np.array(st['total_std'])-exstd)))}
            assert np.allclose(st['mean'],expected,atol=1e-5)
            assert np.allclose(st['total_std'],exstd,atol=1e-5)
    # Does the within/between result survive excluding high-reflectance image means?
    rgb=fs['hlp_rgb'];bright=rgb['mean'].mean(1)
    sensitivity={}
    for threshold in (.5,.8):
        keep=bright<threshold
        sub={k:v[keep] for k,v in rgb.items()}
        s=summarize(sub,hlp['tiles'][keep]);s.pop('rg_histogram',None)
        sensitivity[f'mean_rgb_below_{threshold}']=dict(excluded_images=int((~keep).sum()),**s)
    result={'status':'partial' if a.partial else 'complete','statistics':stats,'bright_image_sensitivity':sensitivity,'hlp_six_training_zscore':standardized,
            'natural_embedded_icc_count':int(nat['icc'].sum()),'normalization_crosscheck':checks,
            'coverage':{'natural_images':len(nat['ids']),'hlp_pairs':len(hlp['ids']),'hlp_tiles':len(np.unique(hlp['tiles']))},
            'interpretation_limits':['Natural RGB assumes sRGB and ignores embedded ICC just like prior VAR evaluation.',
            'HLS reflectance RGB is not calibrated CIE XYZ/sRGB; linearization is sensitivity only.',
            'Pixel grids are both 256x256 but physical field of view and sensor resolution differ.',
            'Census descriptors are not proof of codebook collapse or causal explanation of model behavior.']}
    dest=OUT/('partial' if a.partial else 'results');dest.mkdir(exist_ok=True)
    dump(dest/'summary.json',result);make_figures(fs,stats,dest)
    # Persist one-to-one IDs with per-image color moments for independent reanalysis.
    for kind,data in [('natural',nat),('hlp',hlp)]:
        np.savez_compressed(dest/(kind+'_per_image_moments.npz'),**{k:v for k,v in data.items() if not k.endswith('_sampled')})
    print(json.dumps({'coverage':result['coverage'],'metrics':{k:{q:s[q] for q in ['within_variance_share','spatial_rms_quantiles','pca_total','pca_within']} for k,s in stats.items()},'checks':checks},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
