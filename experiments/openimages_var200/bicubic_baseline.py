"""Paired 8x8 image baseline against the saved official VAR reconstructions."""
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from experiments.openimages_var200.prepare import OUT, save, sha


def main():
    torch.set_num_threads(4)
    dest = OUT/'bicubic_8x8'
    dest.mkdir(exist_ok=True)
    arrays = np.load(OUT/'reconstruction_arrays.npz', allow_pickle=False)
    targets = arrays['input_rgb_u8']
    reference = targets.astype(np.float32)/255
    var = arrays['reconstruction_rgb_float32']
    ids = arrays['image_ids'].tolist()
    assert reference.shape == var.shape == (200, 256, 256, 3)
    x = torch.from_numpy(reference).permute(0, 3, 1, 2).contiguous()
    # A 32x32 block average per output pixel: no aliased nearest-neighbor reduction.
    small = F.interpolate(x, size=(8, 8), mode='area')
    expanded = F.interpolate(small, size=(256, 256), mode='bicubic', align_corners=False)
    clipped_fraction = float(((expanded < 0) | (expanded > 1)).float().mean())
    pred = expanded.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    expected_small = reference.reshape(200, 8, 32, 8, 32, 3).mean(axis=(2, 4))
    assert np.allclose(small.permute(0, 2, 3, 1).numpy(), expected_small, atol=2e-6)
    baseline_mse, var_mse, rows = [], [], []
    for i, image_id in enumerate(ids):
        b = float(np.mean((pred[i].astype(np.float64)-reference[i])**2))
        v = float(np.mean((var[i].astype(np.float64)-reference[i])**2))
        baseline_mse.append(b)
        var_mse.append(v)
        rows.append({'image_id': image_id, 'bicubic_rmse_255': math.sqrt(b)*255,
                     'var_rmse_255': math.sqrt(v)*255, 'var_rmse_reduction_percent': (1-math.sqrt(v/b))*100})
    b_rmse, v_rmse = math.sqrt(np.mean(baseline_mse))*255, math.sqrt(np.mean(var_mse))*255
    old = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    assert math.isclose(v_rmse, old['rmse_255'], abs_tol=1e-9)
    summary = {'images': 200, 'input_resolution': [256, 256], 'low_resolution': [8, 8],
        'downsample': 'float32 area average (32x32 blocks)',
        'upsample': 'torch bicubic, align_corners=False, clamp to [0,1]',
        'bicubic_clipped_fraction': clipped_fraction,
        'bicubic_rmse_255': b_rmse, 'var_rmse_255': v_rmse,
        'absolute_rmse_reduction_255': b_rmse-v_rmse,
        'relative_rmse_reduction_percent': (1-v_rmse/b_rmse)*100,
        'relative_mse_reduction_percent': (1-(v_rmse/b_rmse)**2)*100,
        'var_lower_rmse_images': sum(v < b for v, b in zip(var_mse, baseline_mse)),
        'metric': 'pooled all 200 images, RGB pixels; float outputs before PNG rounding',
        'source_arrays_sha256': sha(OUT/'reconstruction_arrays.npz'),
        'torch_version': torch.__version__}
    save(dest/'summary.json', summary)
    (dest/'per_image.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n', encoding='utf-8')
    np.savez_compressed(dest/'arrays.npz', image_ids=arrays['image_ids'],
                        low_rgb_float32=small.permute(0, 2, 3, 1).numpy(), reconstruction_rgb_float32=pred)
    pngs = dest/'reconstructions_256'
    pngs.mkdir(exist_ok=True)
    pred_u8, var_u8 = np.rint(pred*255).astype(np.uint8), np.rint(var*255).astype(np.uint8)
    for i, image_id in enumerate(ids):
        Image.fromarray(pred_u8[i]).save(pngs/f'{image_id}.png')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    title = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    for page in range(2):
        canvas = Image.new('RGB', (804, 1270), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), 'OpenImages：原图 / 8×8 → bicubic / VAR VQ-VAE', font=title, fill='black')
        draw.text((12, 42), '固定抽样顺序；RMSE 使用 0–255 单位；每格 256×256', font=font, fill='black')
        for slot in range(4):
            i = page*4+slot
            y = 80+slot*295
            draw.text((12, y), f"{i+1:03d} {ids[i]}    bicubic RMSE {rows[i]['bicubic_rmse_255']:.2f}    VAR {rows[i]['var_rmse_255']:.2f}", font=font, fill='black')
            for col, a in enumerate((targets[i], pred_u8[i], var_u8[i])):
                canvas.paste(Image.fromarray(a), (12+col*264, y+28))
        canvas.save(dest/f'comparison_first8_{page+1}.png')
    report = f'''# 8×8 图像插值与 VAR 重建对照

沿用同一批固定抽样的200张 OpenImages 图片、相同256×256 RGB裁剪输入。先将每个32×32区域平均得到8×8 RGB图像，再用bicubic恢复256×256；align_corners=False。两种方法均将浮点输出裁剪至[0,1]，在保存PNG前计算误差。RMSE先汇总所有图片、像素和通道的平方误差再开平方。

| 方法 | 整体RMSE（0–255） |
|---|---:|
| 8×8 区域平均 → bicubic | {b_rmse:.4f} |
| 官方 VAR VQ-VAE，完整680索引 | {v_rmse:.4f} |

VQ-VAE的RMSE减少 **{b_rmse-v_rmse:.4f}**，相对降低 **{summary['relative_rmse_reduction_percent']:.2f}%**。MSE相对降低 **{summary['relative_mse_reduction_percent']:.2f}%**，注意这是不同指标。200张中有 **{summary['var_lower_rmse_images']}张** 的VQ-VAE逐图RMSE更低。

这是对粗分辨率图像插值的基线比较，不是等码率比较：8×8 RGB有192个通道数值，而完整VAR含680个离散索引并依赖预训练解码器。结果也不能直接外推到LST温度误差。

全部200张bicubic重建在 `reconstructions_256/`，逐图配对指标在 `per_image.jsonl`，低分辨率图像和浮点重建在 `arrays.npz`。以下只显示抽样顺序中的前8张，不按效果筛选。

![前四张](comparison_first8_1.png)

![第五至八张](comparison_first8_2.png)
'''
    (dest/'REPORT_ZH.md').write_text(report, encoding='utf-8')
    # Check saved representation, not only the in-memory calculations.
    stored = np.load(dest/'arrays.npz', allow_pickle=False)
    assert stored['image_ids'].tolist() == ids and np.array_equal(stored['reconstruction_rgb_float32'], pred)
    for i, image_id in enumerate(ids):
        with Image.open(pngs/f'{image_id}.png') as im:
            assert np.array_equal(np.asarray(im), pred_u8[i])
    save(dest/'verification.json', {'status': 'passed', 'images': 200, 'var_rmse_matches_previous_run': True,
                                    'area_downsample_matches_block_average': True, 'all_saved_pngs_match': True})
    save(dest/'artifact_hashes.json', {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
