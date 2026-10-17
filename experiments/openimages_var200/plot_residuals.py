"""Comparable residual maps of all three cases where bicubic has lower RMSE."""
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from experiments.openimages_var200.prepare import OUT, save


def main():
    folder = OUT/'bicubic_8x8'
    dest = folder/'residuals_all3'
    dest.mkdir(exist_ok=True)
    cases = json.loads((folder/'var_higher_rmse_all3.json').read_text(encoding='utf-8'))
    source = np.load(OUT/'reconstruction_arrays.npz', allow_pickle=False)
    baseline = np.load(folder/'arrays.npz', allow_pickle=False)
    assert source['image_ids'].tolist() == baseline['image_ids'].tolist()
    idx = [source['image_ids'].tolist().index(r['image_id']) for r in cases]
    target = source['input_rgb_u8'][idx]
    truth = target.astype(np.float32)/255
    var = source['reconstruction_rgb_float32'][idx]
    bicubic = baseline['reconstruction_rgb_float32'][idx]
    # Identical float32 targets and float64 subtraction as in the evaluated RMSE.
    residual = np.stack([bicubic.astype(np.float64)-truth, var.astype(np.float64)-truth], axis=1)*255
    signed = residual @ np.array([.2126, .7152, .0722])
    magnitude = np.sqrt(np.mean(residual**2, axis=-1))
    for i, r in enumerate(cases):
        for method, key in enumerate(['bicubic_rmse_255', 'var_rmse_255']):
            assert math.isclose(float(np.sqrt(np.mean(magnitude[i, method]**2))), r[key], abs_tol=1e-9)
    signed_limit = float(math.ceil(np.abs(signed).max()/10)*10)
    magnitude_limit = float(math.ceil(magnitude.max()/10)*10)
    plt.rcParams['font.family'] = FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    plt.rcParams['axes.unicode_minus'] = False

    def plot(indices, path):
        fig, ax = plt.subplots(len(indices), 5, figsize=(16, 3.5*len(indices)+1.5), squeeze=False)
        fig.subplots_adjust(left=.035, right=.985, top=.89, bottom=.14 if len(indices)==1 else .105, wspace=.075, hspace=.22)
        fig.suptitle('残差 = 重建 − 原图；所有图片、两种方法共用色标', fontsize=18, y=.98)
        columns = ['原图', 'bicubic：亮度加权残差', 'VQ-VAE：亮度加权残差', 'bicubic：RGB误差幅度', 'VQ-VAE：RGB误差幅度']
        for row, i in enumerate(indices):
            ax[row, 0].imshow(target[i])
            for method in range(2):
                signed_image = ax[row, method+1].imshow(signed[i, method], cmap='RdBu_r', vmin=-signed_limit, vmax=signed_limit, interpolation='nearest')
                magnitude_image = ax[row, method+3].imshow(magnitude[i, method], cmap='magma', vmin=0, vmax=magnitude_limit, interpolation='nearest')
            for col in range(5):
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([])
                heading = columns[col] if row == 0 else ''
                if col == 0:
                    heading += f"\n第{cases[i]['sample_number']}张 · {cases[i]['image_id']}"
                elif col in (3, 4):
                    heading += f"\n整体RMSE {cases[i]['bicubic_rmse_255' if col == 3 else 'var_rmse_255']:.2f}"
                ax[row, col].set_title(heading.strip(), fontsize=10)
        bottom = .052 if len(indices)>1 else .055
        colorbar1 = fig.colorbar(signed_image, cax=fig.add_axes([.24, bottom, .32, .016 if len(indices)>1 else .027]), orientation='horizontal')
        colorbar1.set_label('蓝：重建偏暗 ← 0 → 红：重建偏亮（0–255单位）', fontsize=10)
        colorbar2 = fig.colorbar(magnitude_image, cax=fig.add_axes([.635, bottom, .32, .016 if len(indices)>1 else .027]), orientation='horizontal')
        colorbar2.set_label('每像素 √[(ΔR²+ΔG²+ΔB²)/3]；越亮误差越大', fontsize=10)
        fig.savefig(path, dpi=160, facecolor='white')
        plt.close(fig)

    plot(list(range(3)), dest/'residuals_all3.png')
    for i, r in enumerate(cases):
        plot([i], dest/f"residual_{r['sample_number']:03d}_{r['image_id']}.png")
    np.savez_compressed(dest/'residual_arrays.npz', image_ids=np.array([r['image_id'] for r in cases]),
                        methods=np.array(['bicubic', 'var']), rgb_residual_255=residual,
                        luma_weighted_residual_255=signed, pixel_rgb_rmse_255=magnitude)
    result = {'images': [r['image_id'] for r in cases], 'signed_limits': [-signed_limit, signed_limit],
              'magnitude_limits': [0, magnitude_limit], 'color_clipping': False,
              'definition': 'prediction minus target, RGB 0-255 units; weighted brightness is 0.2126R+0.7152G+0.0722B',
              'rmse_recomputed_matches_previous_results': True}
    save(dest/'metadata.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
