"""Reproducible final HLC scientific figures and numerical audit from saved artifacts."""
from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from eo_data import open_dataset
BASE=ROOT/'output/efdiff_hlc_raw_mmdit_b256_v_20261014'
NEW=BASE/'train/v'
OLD=ROOT/'output/efdiff_hlc_dit_timegeo_v_20261004/train/v'
UNET=ROOT/'output/hlc_hls_context_20261013/unet_same_cases'
OUT=BASE/'final_results'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})

def read(p):return json.loads(p.read_text())
def metric(pred,target,mask):
    e=pred[mask].astype('float64')-target[mask].astype('float64')
    return dict(rmse_c=float(np.sqrt(np.mean(e**2))),mae_c=float(np.mean(abs(e))),bias_c=float(e.mean()))

def main():
    complete=read(NEW/'complete.json'); migration=read(BASE/'resume_migration.json')
    assert complete['training_records']==5217544 and complete['updates']==migration['schedule_total_steps']==30573
    e1=read(NEW/'validation_2608772.json');e2=read(NEW/'validation_5217544.json')
    nfe_rows=[]; main_rows=[]
    for title,path in [('Old DiT epoch1 (best)',OLD/'validation_2608772.json'),('Old DiT epoch2 (final)',OLD/'validation_5217544.json'),('MMDiT epoch1',NEW/'validation_2608772.json'),('MMDiT epoch2 (best=final)',NEW/'validation_5217544.json')]:
        m=read(path)
        assert m['records']==141711 and m['valid_pixels']==e2['valid_pixels'] and m['nfe']==50
        for k in ('lst','hls'):assert m['validation_fingerprint'][k]==e2['validation_fingerprint'][k]
        main_rows.append(dict(model=title,rmse_c=m['rmse_c'],mae_c=m['mae_c'],nfe=50,records=m['records']))
    for n in [50,20,10,5]:
        m=read(NEW/f'final_nfe{n}.json');best=read(NEW/f'best_nfe{n}.json');old=read(OLD/f'best_nfe{n}.json')
        assert m['records']==141711 and m['validation_fingerprint']==e2['validation_fingerprint']
        assert np.isfinite([m['rmse_c'],m['mae_c']]).all()
        assert m['rmse_c']==best['rmse_c'] and m['mae_c']==best['mae_c']
        nfe_rows.append(dict(nfe=n,rmse_c=m['rmse_c'],mae_c=m['mae_c'],old_best_rmse_c=old['rmse_c'],old_best_mae_c=old['mae_c'],sampling_seconds=m['sampling_seconds_including_encoder']))
    arrays={'epoch1':np.load(NEW/'validation_2608772.npz'),'mmdit50':np.load(NEW/'final_nfe50.npz'),
            'mmdit5':np.load(NEW/'final_nfe5.npz'),'old50':np.load(OLD/'final_nfe50.npz'),
            'unet_eps':np.load(UNET/'epsilon.npz'),'unet_x0':np.load(UNET/'x0.npz')}
    a=arrays['mmdit50'];info=read(ROOT/'output/hlc_hls_context_20261013/manifest.json')['cases']
    for label,z in arrays.items():
        for k in ['index','record_id','month','target_c','mask']:
            assert np.array_equal(z[k][:8],a[k][:8]),(label,k)
        np.testing.assert_allclose(z['condition_c'][:8],a['condition_c'][:8],rtol=0,atol=2e-5)

    # Snapshot loss logs are sparse. Never describe these as all-step averages.
    logs=sorted({r['step']:r for r in map(json.loads,(NEW/'metrics.jsonl').read_text().splitlines())}.values(),key=lambda r:r['step'])
    assert logs[-1]['presented']==5217544 and logs[-1]['step']==30573 and logs[-1]['update_records']==132
    full=[r for r in logs if r['update_records'] in (128,256)]
    x=np.array([r['presented']/2608772 for r in full]); y=np.array([r['loss'] for r in full])
    means=[];centers=[]
    for lo in np.arange(0,5217544,128000):
        sel=(x*2608772>=lo)&(x*2608772<lo+128000)
        if sel.any():centers.append(float(x[sel].mean()));means.append(float(y[sel].mean()))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax in axes[0]:
        ax.plot(x,y,color='#aec4da',lw=.6,label='Logged full-batch loss')
        ax.plot(centers,means,color='#165c9b',lw=2,label='Log mean / 128k presentations')
        ax.axvline(1,color='#444444',ls='--',lw=1);ax.set_xlabel('Complete epochs presented');ax.set_ylabel('v-pred MSE');ax.grid(alpha=.18)
    axes[0,0].set_yscale('log');axes[0,0].set_title('Training loss: full range (log scale)');axes[0,0].legend(fontsize=8)
    axes[0,1].set_xlim(.25,2.02);axes[0,1].set_ylim(0,.13);axes[0,1].set_title('Later training: linear scale')
    axes[1,0].plot([r['presented']/2608772 for r in logs],[r['lr'] for r in logs],color='#165c9b')
    axes[1,0].axvline(1,color='#444444',ls='--');axes[1,0].set(xlabel='Complete epochs presented',ylabel='Learning rate',title='Sample-based schedule');axes[1,0].grid(alpha=.18)
    axes[1,1].plot([1,2],[e1['rmse_c'],e2['rmse_c']],marker='o',color='#165c9b')
    for ep,m in [(1,e1),(2,e2)]:axes[1,1].annotate(f"{m['rmse_c']:.4f} C",(ep,m['rmse_c']),xytext=(0,10),textcoords='offset points',ha='center')
    axes[1,1].set(xlabel='Completed epoch',ylabel='Full-val RMSE (C)',title='EMA validation | 141,711 records | 50 NFE',xticks=[1,2],xlim=(.8,2.2),ylim=(1.49,1.59));axes[1,1].grid(alpha=.18)
    fig.suptitle('HLC raw-HLS MMDiT | epoch1 batch128 -> epoch2 batch256\nSparse training logs; tail batches excluded from loss smoothing',fontsize=14)
    fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key in zip(axes,['rmse_c','mae_c']):
        ax.plot([r['nfe'] for r in nfe_rows],[r[key] for r in nfe_rows],'-o',color='#165c9b',label='MMDiT best (epoch2)')
        ax.plot([r['nfe'] for r in nfe_rows],[r['old_best_'+key] for r in nfe_rows],'--s',color='#ac7525',label='Old DiT best (epoch1)')
        ax.set(xlabel='Denoising function evaluations (NFE)',ylabel=key.replace('_c','').upper()+' (C)',xticks=[5,10,20,50]);ax.grid(alpha=.18);ax.legend(fontsize=9)
    fig.suptitle('Full validation | best checkpoints selected at 50 NFE | different architectures and batch schedules')
    fig.savefig(OUT/'nfe_comparison.png',dpi=160);plt.close(fig)

    # Boundary statistics: both adjacent pixels must be valid; all 16 saved cases.
    boundary={}
    for label,img in [('reference',a['target_c']),('old50',arrays['old50']['prediction_c']),('epoch1',arrays['epoch1']['prediction_c']),('mmdit50',a['prediction_c']),('mmdit5',arrays['mmdit5']['prediction_c'])]:
        image=img[:,0].astype('float64');mask=a['mask'][:,0];sums=np.zeros(16);counts=np.zeros(16)
        for axis in [1,2]:
            d=abs(np.diff(image,axis=axis));v=(mask[:,1:]&mask[:,:-1]) if axis==1 else (mask[:,:,1:]&mask[:,:,:-1])
            for phase in range(16):
                sel=np.arange(1,224)%16==phase;dd=d[:,sel,:] if axis==1 else d[:,:,sel];vv=v[:,sel,:] if axis==1 else v[:,:,sel]
                sums[phase]+=dd[vv].sum();counts[phase]+=vv.sum()
        avg=sums/counts;other=sums[1:].sum()/counts[1:].sum()
        boundary[label]=dict(boundary_c=float(avg[0]),interior_c=float(other),ratio=float(avg[0]/other),phase_means=avg.tolist())
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for label,color,style in [('reference','#555555','--'),('epoch1','#ac7525','-'),('mmdit50','#165c9b','-'),('mmdit5','#75559c',':')]:
        ax.plot(range(16),boundary[label]['phase_means'],style,marker='o',color=color,label=label)
    ax.set(xlabel='Pixel-edge position modulo 16 (0 = patch boundary)',ylabel='Mean absolute neighbor difference (C)',xticks=range(16),title='Patch boundary diagnostic | same 16 saved validation cases');ax.grid(alpha=.18);ax.legend()
    fig.savefig(OUT/'patch_boundary.png',dpi=160);plt.close(fig)

    hls=open_dataset('lstsr_tb','hls','val',root=ROOT,prepared=ROOT/'processed_data/v1')
    cmap=plt.get_cmap('inferno').copy();cmap.set_bad('#d4d4d4')
    case_rows=[]
    for i in range(8):
        rid=int(a['record_id'][i]);assert rid==info[i]['record_id']
        j=int(np.searchsorted(hls.indices[:,0],rid));assert hls.indices[j,0]==rid
        hs=hls[j];reflectance=hls.denormalize(hs['image'])[:,16:240,16:240];hm=hs['reference_mask'][0,16:240,16:240]
        rgb=np.moveaxis(np.clip(reflectance[[2,1,0]]/.3,0,1),0,-1)**(1/2.2);rgb[~hm]=.83
        target=a['target_c'][i,0];mask=a['mask'][i,0];lo,hi=np.percentile(target[mask],[0,100])
        fig,axes=plt.subplots(3,3,figsize=(11,11.7))
        fig.subplots_adjust(left=.025,right=.875,bottom=.035,top=.90,wspace=.045,hspace=.21)
        axes[0,0].imshow(rgb,interpolation='nearest');axes[0,0].set_title(f"HLS RGB | {info[i]['hls_date']}")
        panels=[(0,1,arrays['unet_eps']['synthetic_coarse_c'][i,0],'Synthetic coarse 7x7',False,None),
                (0,2,target,'Reference fine LST',True,None),
                (1,0,arrays['unet_eps']['prediction_c'][i,0],'Historical UNet epsilon | E1 | 50 NFE',True,'unet_eps'),
                (1,1,arrays['old50']['prediction_c'][i,0],'Old TimeGeo DiT | E2 | 50 NFE',True,'old50'),
                (1,2,a['prediction_c'][i,0],'Raw-HLS MMDiT | E2 | 50 NFE',True,'mmdit50'),
                (2,0,arrays['mmdit5']['prediction_c'][i,0],'Raw-HLS MMDiT | E2 | 5 NFE',True,'mmdit5')]
        for row,col,v,title,masked,key in panels:
            im=axes[row,col].imshow(np.ma.masked_where(~mask,v) if masked else v,cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
            if key:
                m=metric(v,target,mask);title+=f"\nRMSE {m['rmse_c']:.3f} C | bias {m['bias_c']:+.3f} C"
            axes[row,col].set_title(title,fontsize=10)
        for col,label in [(1,'mmdit50'),(2,'mmdit5')]:
            err=arrays[label]['prediction_c'][i,0]-target
            imerr=axes[2,col].imshow(np.ma.masked_where(~mask,err),cmap='RdBu_r',vmin=-5,vmax=5,interpolation='nearest')
            axes[2,col].set_title(f'{label}: prediction - reference\nFixed scale -5 to +5 C',fontsize=10)
        for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,cax=fig.add_axes([.90,.40,.018,.40]),label='Temperature (C)')
        fig.colorbar(imerr,cax=fig.add_axes([.90,.065,.018,.23]),label='Error (C)')
        fig.suptitle(f"HLC case {i+1} | record {rid} | LST {info[i]['lst_date']}\nSame record / target / mask; historical UNet is not a controlled architecture comparison",fontsize=12)
        fig.savefig(OUT/f'case_{i+1:02d}.png',dpi=150);plt.close(fig)
        row=dict(case=i+1,index=int(a['index'][i]),record_id=rid,month=int(a['month'][i]),hls_date=info[i]['hls_date'],lst_date=info[i]['lst_date'])
        row['coarse']=metric(a['condition_c'][i,0],target,mask)
        for label,z in arrays.items():row[label]=metric(z['prediction_c'][i,0],target,mask)
        case_rows.append(row)
    summary=dict(main_table_50nfe=main_rows,nfe_table=nfe_rows,cases=case_rows,boundary_16_saved_cases=boundary,
        training=dict(records=5217544,updates=30573,epochs=2,batches=[128,256],epoch_updates=[20382,10191],tails=[4,132],
                      trainable_parameters=217195328,last20_logged_full_batch_loss=float(y[-20:].mean()),last_loss=logs[-1]['loss'],
                      peak_allocated_GB=max(r.get('peak_allocated_GB',0) for r in logs)),
        gpu_stats=read(BASE/'fetch_status.json')['job_stats'],best_epoch=2,
        caveats=['All headline scores are synthetic-degradation full validation, not test.',
                 'UNet comparisons cover eight fixed cases only; objectives, crop, HLS preprocessing, RNG and budgets differ.',
                 'MMDiT architecture, encoder and fusion change together; epoch2 batch also changes.',
                 'Training loss is sparse logged v-pred MSE; validation is pixel-weighted Celsius error.',
                 'Best checkpoint selected using 50 NFE; 5 NFE is a separate sampling comparison.',
                 'HLS acquisition dates differ from LST dates; RGB display is clipped, model valid reflectance is not.'])
    (OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (OUT/'main_table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(main_rows[0]));w.writeheader();w.writerows(main_rows)
    with (OUT/'case_metrics.csv').open('w',newline='') as f:
        fields=['case','record_id','model','rmse_c','mae_c','bias_c'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in case_rows:
            for label in ['coarse',*arrays]:w.writerow(dict(case=c['case'],record_id=c['record_id'],model=label,**c[label]))
    sources=[NEW/'complete.json',NEW/'metrics.jsonl',NEW/'provenance.json',BASE/'resume_migration.json',OLD/'complete.json',UNET/'audit.json']
    (OUT/'sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(dict(output=str(OUT),training=summary['training'],main_table=main_rows,cases=len(case_rows)),indent=2))

if __name__=='__main__':main()
