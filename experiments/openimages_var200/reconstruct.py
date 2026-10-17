"""Official VAR RGB VQ-VAE, 200 fixed OpenImages images, no training or AR generator."""
from pathlib import Path
import json
import math
import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
import torch
from torchvision import transforms
from experiments.openimages_var200.prepare import OUT, ROOT, sha, save

SCALES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


def entropy(counts):
    p = counts[counts > 0]/counts.sum()
    h = float(-(p*np.log2(p)).sum())
    return {'used_codes': int((counts > 0).sum()), 'perplexity': 2**h, 'entropy_bits': h,
            'vocabulary': len(counts), 'tokens': int(counts.sum())}


def galleries(images, predictions, rows, out):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    titlefont = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    dest = out/'gallery'
    dest.mkdir(exist_ok=True)
    filenames = []
    for page in range(25):
        canvas = Image.new('RGB', (1060, 1228), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 6), f'VAR 官方 VQ-VAE · OpenImages 固定抽样 · {page+1}/25', font=titlefont, fill='black')
        draw.text((16, 38), '每组：左 原图输入 / 右 重建；256×256 原生像素显示', font=font, fill='#444444')
        for slot in range(8):
            idx = page*8+slot
            x, y = 12+(slot%2)*524, 72+(slot//2)*286
            row = rows[idx]
            draw.text((x, y), f"{idx+1:03d}  {row['image_id']}  PSNR {row['psnr_db']:.2f} dB", font=font, fill='black')
            canvas.paste(Image.fromarray(images[idx]), (x, y+25))
            canvas.paste(Image.fromarray(predictions[idx]), (x+260, y+25))
        name = f'comparison_{page+1:02d}.png'
        canvas.save(dest/name)
        filenames.append(name)
    # A location fixed before observing errors; nearest-neighbor magnification adds no detail.
    detail = Image.new('RGB', (1070, 1235), 'white')
    draw = ImageDraw.Draw(detail)
    draw.text((14, 8), '前4张固定样本：原图 / 重建 / 原图中央96×96 / 重建相同区域', font=titlefont, fill='black')
    draw.text((14, 42), '局部采用最近邻放大；框选位置固定，不按重建效果选择', font=font, fill='#444444')
    for idx in range(4):
        y = 78+idx*284
        draw.text((14, y), rows[idx]['image_id'], font=font, fill='black')
        a, b = Image.fromarray(images[idx]), Image.fromarray(predictions[idx])
        ac, bc = a.crop((80, 80, 176, 176)), b.crop((80, 80, 176, 176))
        for im in (a, b):
            ImageDraw.Draw(im).rectangle((80, 80, 175, 175), outline='#00ff00', width=1)
        for col, im in enumerate((a, b, ac.resize((256, 256), Image.Resampling.NEAREST), bc.resize((256, 256), Image.Resampling.NEAREST))):
            detail.paste(im, (12+col*264, y+24))
    detail.save(dest/'details_first4.png')
    return filenames


@torch.inference_mode()
def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.manual_seed(20260915)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    checkpoint = json.loads((OUT/'checkpoint.json').read_text(encoding='utf-8'))
    source = json.loads((OUT/'upstream_manifest.json').read_text(encoding='utf-8'))
    downloads = json.loads((OUT/'download_manifest.json').read_text(encoding='utf-8'))
    selection = json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    assert checkpoint['verified'] and sha(checkpoint['path']) == checkpoint['sha256']
    for name, digest in source['sha256'].items():
        assert sha(OUT/'upstream'/name) == digest
    sys.path.insert(0, str(OUT/'upstream'))
    from models.vqvae import VQVAE
    device = torch.device('cuda')
    free, total = torch.cuda.mem_get_info()
    if free < 5*2**30:
        raise RuntimeError('Need at least 5 GiB free before this small inference run')
    torch.cuda.set_per_process_memory_fraction(3*2**30/total)
    model = VQVAE(vocab_size=4096, z_channels=32, ch=160, share_quant_resi=4,
                  v_patch_nums=SCALES, test_mode=True).eval()
    state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
    # All checkpoint keys, including the scale-use buffer, must match exactly.
    assert set(model.state_dict()) == set(state)
    assert all(model.state_dict()[k].shape == v.shape for k, v in state.items())
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    transform = transforms.Compose([transforms.Resize(288, interpolation=transforms.InterpolationMode.LANCZOS),
                                    transforms.CenterCrop(256), transforms.ToTensor()])
    images = []
    for r in downloads['images']:
        path = OUT/'images'/f"{r['image_id']}.jpg"
        assert sha(path) == r['sha256']
        with Image.open(path) as im:
            images.append(transform(im.convert('RGB')))
    assert [r['image_id'] for r in downloads['images']] == [r['ImageID'] for r in selection['records']]
    x = torch.stack(images)
    targets = x.permute(0, 2, 3, 1).numpy()
    predictions, indices = [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for begin in range(0, 200, 2):
        inputs = x[begin:begin+2].to(device)*2-1
        ids = model.img_to_idxBl(inputs)
        reconstruction = model.idxBl_to_img(ids, same_shape=True, last_one=True)
        if begin == 0:
            direct = model.img_to_reconstructed_img(inputs, last_one=True)
            difference = float((direct-reconstruction).abs().max())
            # Upstream paths use different tensor layouts/contiguity during lookup
            # and convolution. Allow FP32 roundoff, at most 0.01275 in 8-bit units.
            assert difference < 1e-4, difference
            del direct
        predictions.append(((reconstruction+1)*.5).permute(0, 2, 3, 1).cpu().numpy())
        indices.append(torch.cat(ids, dim=1).cpu().numpy())
        if (begin+2) % 20 == 0:
            free, _ = torch.cuda.mem_get_info()
            print(f'Reconstructed {begin+2}/200; peak allocated {torch.cuda.max_memory_allocated()/2**30:.3f} GiB; free {free/2**30:.2f} GiB', flush=True)
            if free < 2*2**30:
                raise RuntimeError('Less than 2 GiB free; leave other GPU jobs intact')
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    pred = np.concatenate(predictions)
    ids = np.concatenate(indices)
    assert pred.shape == targets.shape == (200, 256, 256, 3) and ids.shape == (200, 680)
    assert np.isfinite(pred).all() and ids.min() >= 0 and ids.max() < 4096
    target_u8 = np.rint(targets*255).astype(np.uint8)
    pred_u8 = np.rint(pred*255).astype(np.uint8)
    rows = []
    for i, item in enumerate(downloads['images']):
        error = pred[i].astype(np.float64)-targets[i]
        mse = float(np.mean(error**2))
        dx, dy = error[:, 1:]-error[:, :-1], error[1:]-error[:-1]
        rows.append({'image_id': item['image_id'], 'mse_01': mse, 'rmse_255': math.sqrt(mse)*255,
            'mae_255': float(np.mean(np.abs(error)))*255, 'psnr_db': -10*math.log10(mse),
            'ssim': float(structural_similarity(targets[i], pred[i], data_range=1., channel_axis=-1,
                          gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
            'gradient_rmse_255': math.sqrt((np.mean(dx**2)+np.mean(dy**2))/2)*255,
            'finest_grid_unique_codes': int(len(np.unique(ids[i, -256:]))),
            'finest_grid_mode_fraction': float(np.bincount(ids[i, -256:], minlength=4096).max()/256)})
        for subdir, array in [('inputs_256', target_u8[i]), ('reconstructions_256', pred_u8[i])]:
            folder = OUT/subdir
            folder.mkdir(exist_ok=True)
            Image.fromarray(array).save(folder/(item['image_id']+'.png'))
    with (OUT/'per_image.jsonl').open('w', encoding='utf-8') as stream:
        for r in rows:
            stream.write(json.dumps(r)+'\n')
    np.savez_compressed(OUT/'reconstruction_arrays.npz', image_ids=np.array([r['image_id'] for r in rows]),
                        input_rgb_u8=target_u8, reconstruction_rgb_float32=pred, indices=ids, scales=SCALES)
    histogram = np.bincount(ids.ravel(), minlength=4096)
    by_scale = []
    start = 0
    for scale in SCALES:
        count = np.bincount(ids[:, start:start+scale**2].ravel(), minlength=4096)
        by_scale.append({'grid': scale, **entropy(count)})
        start += scale**2
    np.save(OUT/'code_counts.npy', histogram, allow_pickle=False)
    mean_mse = float(np.mean([r['mse_01'] for r in rows]))
    summary = {'status': 'complete', 'images': 200, 'resolution': 256,
        'rmse_255': math.sqrt(mean_mse)*255, 'mean_image_psnr_db': float(np.mean([r['psnr_db'] for r in rows])),
        'pooled_psnr_db': -10*math.log10(mean_mse), 'mean_ssim': float(np.mean([r['ssim'] for r in rows])),
        'mean_mae_255': float(np.mean([r['mae_255'] for r in rows])),
        'median_finest_unique_codes': float(np.median([r['finest_grid_unique_codes'] for r in rows])),
        'median_finest_mode_fraction': float(np.median([r['finest_grid_mode_fraction'] for r in rows])),
        'tokens': entropy(histogram), 'tokens_by_scale': by_scale,
        'parameter_count': sum(p.numel() for p in model.parameters()), 'precision': 'float32; TF32 disabled; no AMP',
        'inference_seconds': elapsed, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
        'weight_sha256': checkpoint['sha256'], 'upstream_revision': source['revision'],
        'strict_checkpoint_load': True, 'roundtrip_max_abs_difference': difference,
        'preprocessing': 'official validation transform: RGB, shorter edge 288 Lanczos, center crop 256; input in [-1,1]',
        'metric_definition': 'float reconstruction clamped to [0,1], RGB all pixels; SSIM Gaussian sigma1.5 population covariance; PSNR both mean per-image and pooled',
        'training': False, 'noise': False, 'fid_computed': False,
        'dataset_caveat': 'fixed 200 OpenImages validation images; exact official tokenizer training membership not published/verified; not claimed as untouched benchmark'}
    summary['gallery_pages'] = galleries(target_u8, pred_u8, rows, OUT)
    save(OUT/'summary.json', summary)
    lines = ['# VAR 官方 VQ-VAE：OpenImages 200 张重建测试', '',
        f"200/200 张图片完成重建。平均逐图 PSNR **{summary['mean_image_psnr_db']:.2f} dB**，平均 SSIM **{summary['mean_ssim']:.4f}**，汇总 RGB RMSE **{summary['rmse_255']:.3f}/255**。", '',
        '从官方 OpenImages validation 的 41,620 张名单中，按 SHA256("20260915:"+ImageID) 排序取前 200 张。名单先于推理固定，不按类别、质量或重建效果筛选；下载失败不自动换样本。保留原始 JPEG、许可证与作者等元数据及 SHA256。下载的是 CVDF 最长边不超过 1024 的图片版本。', '',
        '加载官方 vae_ch160v4096z32.pth，核对 Hugging Face LFS SHA256 和全部参数名、形状；采用固定上游提交的原始推理代码。4096 码字、32 维、基础通道 160；每图 680 个多尺度索引。只做图像编码和解码，无训练、无加噪、无自回归生成。', '',
        '预处理沿用上游验证流程：RGB，短边缩放到 288（Lanczos），中心裁剪 256×256，输入映射到 [-1,1]。报告比较的是裁剪后输入与其重建，不是完整原始 JPEG 的逐像素还原。FP32 推理，关闭 TF32 和 AMP；输出按官方方式裁剪，再换算到 [0,1]。', '',
        f"汇总 MSE 对应 PSNR 为 {summary['pooled_psnr_db']:.2f} dB；逐图 PSNR 均值与其定义不同。MAE 为 {summary['mean_mae_255']:.3f}/255。SSIM 为 RGB 通道平均、Gaussian sigma=1.5、population covariance。图像输出为 PNG，指标在保存成 8-bit PNG 前的浮点输出上计算。", '',
        f"全部尺度合计使用 {summary['tokens']['used_codes']}/4096 个码字，困惑度 {summary['tokens']['perplexity']:.2f}；最细 16×16 网格每图不同码字数中位数 {summary['median_finest_unique_codes']:.0f}，单图最常用码字占比中位数 {summary['median_finest_mode_fraction']:.1%}。这些是本次 200 张图上的统计，不能与遥感实验直接作同条件比较。", '',
        f"推理循环耗时 {elapsed:.1f} 秒（含首批往返一致性检查）；峰值张量分配 {summary['peak_allocated_gib']:.3f} GiB。这是单次运行耗时，没有进行独占环境下的重复性能测试。", '',
        '本次是小样本重建诊断，不计算或声称复现论文 ImageNet rFID≈0.90。官方 tokenizer 的精确训练成员未经核实，也不声称此集合完全未被它见过。', '',
        '## 固定位置细节对照', '', '![前四张中央局部](gallery/details_first4.png)', '',
        '## 全部 200 张对比', '', '每页 8 张，每组左侧为输入、右侧为重建；保留抽样顺序，无挑图。', '']
    for page, name in enumerate(summary['gallery_pages'], 1):
        lines.extend([f'### 第 {page} 页', '', f'![第{page}页](gallery/{name})', ''])
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    save(OUT/'artifact_hashes.json', {str(p.relative_to(OUT)): sha(p) for p in [OUT/'summary.json', OUT/'per_image.jsonl',
         OUT/'reconstruction_arrays.npz', OUT/'REPORT_ZH.md', *sorted((OUT/'gallery').glob('*.png'))]})
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
