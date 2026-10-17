"""One paired example for inline inspection; full 16-pair atlas remains the main output."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.paper1.codebook1024.more_predictions import DEST, read, region_metrics


def main(destination=DEST, model_keys=None, column_titles=None, caption=None):
    DEST = Path(destination)
    # Fixed illustrative record 240814; explicitly identified rather than presented as an aggregate.
    i = 11
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    keys = model_keys or [(4096, 'clean'), (4096, 'noisy'), (1024, 'clean'), (1024, 'noisy')]
    titles = column_titles or ['原始参考']+[f'{k} 码字\n'+('干净训练' if n=='clean' else '加噪训练') for k, n in keys]
    fig, axes = plt.subplots(2, 5, figsize=(13.4, 6.6), layout='constrained')
    for row, m in enumerate(('hls', 'lst')):
        with np.load(DEST/f'{m}_inputs.npz') as z:
            mean, std = z['mean'], z['std']
            target = (z['target'][i]*std+mean)/(100 if m=='lst' else 1)
            mask = z['reference_mask'][i, 0].astype(bool)
        sample = read(DEST/f'{m}_samples.json')[i]
        assert sample['record_id']==240814
        crop = sample['crop']
        y, x = crop['y'], crop['x']
        bounds = np.quantile(target[[2,1,0]][:,mask], [.01,.99]) if m=='hls' else sample['display_limits']
        predictions = [target]
        for k, n in keys:
            with np.load(DEST/f'{k}_{m}_{n}_predictions.npz') as z:
                predictions.append((z['clean_prediction'][i]*std+mean)/(100 if m=='lst' else 1))
        for col, pred in enumerate(predictions):
            value = pred[:, y:y+64, x:x+64]
            support = mask[y:y+64, x:x+64]
            ax = axes[row,col]
            ax.set_xticks([]); ax.set_yticks([])
            if m=='hls':
                rgb = np.clip((value[[2,1,0]].transpose(1,2,0)-bounds[0])/(bounds[1]-bounds[0]),0,1)**(1/2.2)
                rgb[~support] = .35
                ax.imshow(rgb,interpolation='nearest')
            else:
                cmap=plt.get_cmap('inferno').copy(); cmap.set_bad('#595959')
                ax.imshow(np.ma.masked_where(~support,value[0]),cmap=cmap,vmin=bounds[0],vmax=bounds[1],interpolation='nearest')
            if row==0: ax.set_title(titles[col],fontsize=11)
            if col==0: ax.set_ylabel(m.upper()+'\n局部 64×64',fontsize=11)
            else:
                r = region_metrics(value,target[:,y:y+64,x:x+64],support)
                unit = '°C' if m=='lst' else '反射率'
                ax.set_xlabel(f"局部 RMSE {r['rmse']:.4f} {unit}",fontsize=9)
    fig.suptitle('样本 12 · prithvi:val:240814 · T50SLD · 所有模型均接收干净输入\n'+(caption or '对照加噪训练与干净训练：局部放大，不做锐化或平滑'),fontsize=12)
    fig.supxlabel('各行统一采用参考图 1%-99% 显示范围；HLS 使用 1/2.2 伽马，RMSE 计算六波段；灰色为无效像素。',fontsize=9)
    fig.savefig(DEST/'paired_sample12_clean_detail.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
