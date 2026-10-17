"""Reuse audited full-train moments, verify membership, freeze a single A06 arm."""
from pathlib import Path
import copy
import json
import numpy as np
from eo_data import open_dataset
from eo_data.core import json_write, sha256_file
from .data import DatasetV2, PREPARED, COUNTS

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT/'runs/paper1/lst_residual_scale_20260918'
BASE = ROOT/'runs/paper1/lst_v2_5epoch_20260915'
BASE_RUN = BASE/'full/full_lst_v2_residual_spatial4_seed17'
NAME = 'full_lst_v2_residual_scaled_spatial4_seed17'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original = read(BASE_RUN/'config.json')
    assert read(BASE_RUN/'status.json')['status'] == 'complete'
    moments_root = ROOT/'runs/openimages_hlp_color_matched_20260917'
    moment_path = moments_root/'results/hlp_per_image_moments.npz'
    hashes = read(moments_root/'delivery_hashes.json')
    assert sha256_file(moment_path) == hashes[str(moment_path.relative_to(ROOT))]
    protocol = read(moments_root/'protocol.json')
    ds = open_dataset('prithvi', 'lst', 'train', root=ROOT, prepared=ROOT/PREPARED)
    assert ds.fingerprint == protocol['lst_fingerprint']
    with np.load(moment_path) as z:
        ids, cov = z['ids'], z['lst_cov'][:, 0, 0]
        assert len(ids) == COUNTS['train'] == len(set(ids.tolist()))
        mapping = dict(ds._connect().execute('SELECT id,key FROM records'))
        expected = np.array([mapping[int(r)] for r in ds.indices[:, 0]])
        assert np.array_equal(expected, ids), 'Cached moments must match every ordered train member'
        assert np.isfinite(cov).all() and (cov >= 0).all()
        # Independent public-reader checks on 16 deterministic positions, all valid pixels.
        checks = []
        for index in np.random.default_rng(20260918).choice(len(ds), 16, replace=False):
            sample = ds[int(index)]
            valid = sample['reference_mask'].astype(bool)
            celsius = ds.denormalize(sample['image'])/100.
            actual = float(np.var(celsius[valid].astype(np.float64)))
            assert np.isclose(actual, cov[index], rtol=2e-5, atol=1e-6)
            checks.append(dict(position=int(index), cached_variance=float(cov[index]), reader_variance=actual))
        residual_std_celsius = float(np.sqrt(cov.mean()))
    global_std_celsius = ds.stats['std'][0]/100.
    scale = residual_std_celsius/global_std_celsius
    assert 0 < scale < 1
    stat = dict(fit_split='train', samples=len(ds), method='sqrt(equal-image mean valid-pixel population variance)',
        residual_std_celsius=residual_std_celsius, global_std_celsius=global_std_celsius,
        normalized_scale=scale, input_gain=1/scale, source_moments=str(moment_path),
        source_moments_sha256=sha256_file(moment_path), data_fingerprint=ds.fingerprint, normalization=ds.stats,
        ordered_member_match=True, public_reader_checks=checks, per_image_scale=False)
    ds.close()
    json_write(OUTPUT/'residual_scale.json', stat)
    configs = {}
    for phase in ('smoke', 'full'):
        c = copy.deepcopy(original)
        c['name'] = NAME.replace('full_', phase+'_', 1)
        c['experiment'].update(id='A06', phase=phase, output_root=str(OUTPUT))
        c['residual_scaling'] = {k:stat[k] for k in ('normalized_scale','residual_std_celsius','fit_split','source_moments_sha256')}
        if phase == 'smoke':
            c['data'].update(train_limit=67, val_limit=8, test_limit=8)
            c['experiment']['epochs'] = 2
            c['train'].update(steps=4, warmup=1, eval_every=2, save_every=1, log_every=1)
            c['eval']['bootstrap_replicates'] = 50
        p = HERE/'configs'/phase/(c['name']+'.json')
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            assert read(p) == c
        else:
            json_write(p, c)
        configs[str(p.relative_to(HERE))] = sha256_file(p)
    full = read(HERE/'configs/full'/f'{NAME}.json')
    compare = copy.deepcopy(full)
    compare.pop('residual_scaling')
    compare['name'] = original['name']
    compare['experiment'] = original['experiment']
    assert compare == original, 'Only residual scaling and experiment identity may change'
    prints, members = {}, {}
    for split in COUNTS:
        d = DatasetV2(full, split, .1)
        prints[split] = d.fingerprint
        members[split] = set(d.record_ids.tolist())
        assert d.sources['lst'].stats == stat['normalization']
        if split == 'train':
            assert d.fingerprint == read(BASE_RUN/'data_fingerprint.json')
        else:
            assert d.fingerprint == read(BASE_RUN/f'final_{split}_noisy10/metrics.json')['data_fingerprint']
        d.close()
    assert all(not members[a]&members[b] for a,b in [('train','val'),('train','test'),('val','test')])
    json_write(OUTPUT/'data_preflight.json', dict(status='passed', fingerprints=prints, configs=configs,
        unchanged_training_and_loss_configuration=True, scale=stat,
        historical_baseline=str(BASE_RUN), baseline_checkpoint_sha256=sha256_file(BASE_RUN/'last.pt')))
    (OUTPUT/'PROTOCOL_ZH.md').write_text(f'''# LST 残差尺度消融 A06

仅新增一组：残差缩放＋空间损失4倍。历史对照为 A05 残差预测＋空间损失4倍；不重复训练对照。
两组均 seed=17、从零训练，HLP v2 train 218,975 条、5 完整 epoch、有效 batch=64、17,110 更新。
沿用相同样本顺序、微批4/累积16、约10%缺失输入、split-local mask bank、归一化、优化器和学习率。
每轮末更新31条，不补齐或丢弃。val=11,684、test=11,757；每轮全量val，固定最后第5轮检查点，test不选模。

输入原有归一化值 x，基准 mu 仅来自实际可见输入，r=(x-mu)/s；输出 xhat=mu+s*rhat。
s=sqrt(mean_i Var_valid_pixels(T_i))/sigma_train={scale:.12g}；图内尺度 {residual_std_celsius:.8f}°C，输入残差放大 {1/scale:.6f} 倍。
尺度使用全部训练记录的冻结全像素统计，逐条成员核验及16条独立读取复核；未使用val/test，未按单张图标准差拉伸。
缺失扰动在原输入空间构造；全部图像损失在逆缩放后的原归一化单位计算。
总损失仍为 reconstruction MSE + 3*去均值空间MSE + 0.1*gradient MSE + 原VQ损失；不额外乘除s改动损失系数。
缩放会改变网络内部梯度与VQ优化动态，不能称为内部实际损失比完全相同。
码率仍为6800 bit索引+32 bit逐图均值；s为一个共享float32模型常量，不增加逐图码率。

先做67记录×2轮技术检查，检查尾批、断点恢复、s=1等价性、尺度往返和真实token解码，再启动正式训练。
最终完整val/test分别评估干净/缺失输入，各2个实现；干净输入是缺失训练模型的泛化测试。
报告总体/均值/空间/梯度/缺失区RMSE、码字困惑度、固定8例预测与误差图、GPU-hours和显存峰值。
训练和评估阶段耗时分别记录，训练阶段含每轮val和检查点I/O；不将墙钟小时称为纯GPU计算小时。
单种子实验，test有历史查看记录。HLS、Prithvi均不在本轮范围。
''', encoding='utf-8')
    print(json.dumps(dict(status='passed', scale=scale, input_gain=1/scale), ensure_ascii=False))


if __name__ == '__main__':
    main()
