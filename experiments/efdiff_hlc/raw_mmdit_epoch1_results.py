"""Render local, matched first-epoch validation results; no training or remote calls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[2]/'output'
NEW=BASE/'efdiff_hlc_raw_mmdit_b256_v_20261014/train/v/validation_2608772'
OLD=BASE/'efdiff_hlc_dit_timegeo_v_20261004/train/v/validation_2608772'
OUT=BASE/'efdiff_hlc_raw_mmdit_b256_v_20261014/epoch1_results'


def main():
    OUT.mkdir(exist_ok=True)
    new=json.loads(NEW.with_suffix('.json').read_text())
    old=json.loads(OLD.with_suffix('.json').read_text())
    for key in ('records','valid_pixels','nfe','presented','step'):
        assert new[key]==old[key]
    assert new['records']==141711 and new['nfe']==50 and new['presented']==2608772
    for modality in ('lst','hls'):
        assert new['validation_fingerprint'][modality]==old['validation_fingerprint'][modality]
    a=np.load(NEW.with_suffix('.npz')); b=np.load(OLD.with_suffix('.npz'))
    for key in ('target_c','condition_c','mask','index','record_id','month'):
        assert np.array_equal(a[key],b[key])
    selected=np.linspace(0,len(a['index'])-1,8,dtype=int).tolist()
    cases=[]
    for i in selected:
        mask=a['mask'][i,0]
        target=a['target_c'][i,0]
        row=dict(capture_position=i,index=int(a['index'][i]),record_id=int(a['record_id'][i]),month=int(a['month'][i]))
        for name,pred in [('coarse',a['condition_c'][i,0]),('old_dit',b['prediction_c'][i,0]),('mmdit',a['prediction_c'][i,0])]:
            e=(pred-target)[mask].astype('float64')
            row[name]=dict(rmse_c=float(np.sqrt(np.mean(e**2))),mae_c=float(np.mean(abs(e))),bias_c=float(np.mean(e)))
        cases.append(row)

    def draw(ids,filename):
        fig,axes=plt.subplots(len(ids),6,figsize=(18,len(ids)*3.1),layout='constrained',squeeze=False)
        titles=['Coarse LST','Reference LST','Old TimeGeo DiT','Raw-HLS MMDiT','Old DiT error','MMDiT error']
        for r,i in enumerate(ids):
            mask=a['mask'][i,0]; target=a['target_c'][i,0]
            values=[a['condition_c'][i,0],target,b['prediction_c'][i,0],a['prediction_c'][i,0]]
            low,high=np.percentile(np.concatenate([v[mask] for v in values]),[1,99])
            for col,v in enumerate(values):
                im=axes[r,col].imshow(np.ma.masked_where(~mask,v),cmap='inferno',vmin=low,vmax=high)
                if col!=1:
                    rmse=np.sqrt(np.mean((v[mask].astype('float64')-target[mask])**2))
                    axes[r,col].set_xlabel(f'Case RMSE {rmse:.2f} C')
            for col,pred in [(4,values[2]),(5,values[3])]:
                err=axes[r,col].imshow(np.ma.masked_where(~mask,pred-target),cmap='RdBu_r',vmin=-5,vmax=5)
                axes[r,col].set_xlabel(f'Mean error {(pred-target)[mask].mean():+.2f} C')
            axes[r,0].set_ylabel(f'Val index {a["index"][i]}\nMonth {a["month"][i]}')
            for col,ax in enumerate(axes[r]):
                ax.set_xticks([]);ax.set_yticks([])
                if r==0:ax.set_title(titles[col],fontsize=13)
            fig.colorbar(im,ax=list(axes[r,:4]),fraction=.013,pad=.008,label='C')
            fig.colorbar(err,ax=list(axes[r,4:]),fraction=.027,pad=.008,label='Prediction - reference (C)')
        fig.suptitle('HLC | matched epoch 1, batch128, 50 NFE\nFixed evenly spaced captures; temperature scale shared within each row; errors clipped to +/-5 C',fontsize=15)
        fig.savefig(OUT/filename,dpi=140)
        plt.close(fig)
    draw(selected,'eight_cases.png')
    draw(selected[:2],'preview_two_cases.png')
    summary=dict(new={k:new[k] for k in ('records','valid_pixels','nfe','presented','step','rmse_c','mae_c')},
                 old={k:old[k] for k in ('records','valid_pixels','nfe','presented','step','rmse_c','mae_c')},
                 reductions_percent={k:100*(old[k]-new[k])/old[k] for k in ('rmse_c','mae_c')},
                 full_validation_members_verified=True,captured_inputs_exactly_equal=True,cases=cases,
                 caveat='Architecture, raw/trainable HLS encoder and fusion change together; not a single-factor ablation. First epoch uses batch128; batch256 starts in epoch2.')
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
