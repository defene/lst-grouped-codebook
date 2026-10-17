"""Re-evaluate four frozen A04 checkpoints on complete HLP v2 val/test."""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eo_data import open_dataset
from eo_data.core import json_write, sha256_file
from eo_data.locking import build_lock
from eo_denoise import engine
from eo_denoise.data import PairedDataset, EpochSampler
from eo_denoise.metrics import EvaluationMetrics
from experiments.paper1.lst_residual.runtime import install, ExperimentModel
from experiments.paper1.lst_residual.prepare import ARMS
from experiments.paper1.lst_full_eval.data import FullDataset, PREPARED, COUNTS
from experiments.paper1.lst_full_eval.mask_bank_v2 import build_bank

OUTPUT = ROOT/'runs/paper1/lst_full_eval_20260914'
OLD = ROOT/'runs/paper1/lst_residual_20260913/pilot'
LABELS = ['原始预测', '原始预测＋空间4倍', '残差预测', '残差预测＋空间4倍']


def utc():
    return datetime.now(timezone.utc).isoformat()


def record_status(output, **fields):
    json_write(output/'status.json', {'utc': utc(), 'pid': os.getpid(), **fields})


class UnitAdapter(torch.nn.Module):
    """Keep dataset in v2 units, map visible inputs/output to frozen checkpoint units."""
    def __init__(self, inner, old_stats, new_stats):
        super().__init__()
        self.inner = inner
        for name, values in [('old_mean', old_stats['mean']), ('old_std', old_stats['std']),
                             ('new_mean', new_stats['mean']), ('new_std', new_stats['std'])]:
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32)[None, :, None, None])

    @property
    def config(self):
        return self.inner.config

    @property
    def quantizer(self):
        return self.inner.quantizer

    def rate(self):
        return self.inner.rate()

    def forward(self, image, input_mask):
        old_image = (image.float()*self.new_std+self.new_mean-self.old_mean)/self.old_std
        old_image = torch.where(input_mask.bool(), old_image, 0)
        output = self.inner(old_image, input_mask)
        output['reconstruction'] = (output['reconstruction'].float()*self.old_std+self.old_mean-self.new_mean)/self.new_std
        return output


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    members = {}
    tiles = {}
    norms = {}
    for split in COUNTS:
        ds = open_dataset('prithvi', 'lst', split, root=ROOT, prepared=ROOT/PREPARED)
        members[split] = set(map(int, ds.indices[:, 0]))
        assert len(members[split]) == COUNTS[split]
        # Tile IDs are read from the authoritative split manifests below.
        norms[split] = ds.stats
        ds.close()
        tiles[split] = set((ROOT/f'data_inspection/hlp_split_20260914/{split}_tiles.txt').read_text().splitlines())
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not members[a] & members[b]
        assert not tiles[a] & tiles[b]
    assert norms['train'] == norms['val'] == norms['test']
    for split in ('val', 'test'):
        if not (output/f'mask_banks/{split}.json').exists():
            record_status(output, status='preparing', stage='mask_bank', split=split)
            build_bank(ROOT, output/'mask_banks', split)
    return members


def summarize_report(output):
    rows = []
    for arm, _, _ in ARMS:
        for split in ('val', 'test'):
            for condition in ('clean', 'noisy10'):
                path = output/arm/split/condition/'metrics.json'
                if path.exists():
                    r = json.loads(path.read_text(encoding='utf-8'))
                    assert r['evaluated_pairs'] == COUNTS[split]
                    rows.append(r)
    json_write(output/'results.json', {'status': 'complete' if len(rows) == 16 else 'running', 'rows': rows})
    lines = ['# 现有四组 LST 模型：完整 val / test 补测', '',
        '本次不重新训练：四个 A04 检查点均从零训练 2,000 步、128,000 次记录呈现，尚不是新的 5 epoch 实验。固定使用 last.pt；本次 val/test 分数不用于选检查点或调参。', '',
        '读取 HLP v2：val 11,684 对，test 11,757 对，均全量、不抽样。旧训练流实际见过其中 6,563 条 val 记录，因此 val 只作历史模型复查，不能称为独立验证；test 不在这些训练流中，但历史实验曾反复监测它，不能声称它从未被查看。', '',
        '数据读取与误差统计使用新 train 的归一化；模型边界转换到旧检查点所需单位，输出再转回 v2，最终报告摄氏度。该兼容转换不修改权重或重新估计任何统计量。新正式训练仍须从零使用 v2 归一化。', '',
        '两种输入：干净、约 10% 已知掩码缺失；每条记录两个固定实现。QA 形状库按 v2 的 val/test 各自重新建立并保存来源，所有模型使用同一记录、形状库与噪声种子。test 使用明确的 test 种子标识，因此噪声实现可能不同于历史旧 val 名称下的实现。', '',
        'RMSE 先在每条记录的有效像素计算 MSE、平均两个实现，再对记录等权平均后开方。均值与空间 RMSE 来自逐图偏差平方及剩余 MSE；空间项包含大尺度变化，不能单独代表高频细节。', '',
        '| 集合 | 模型 | 输入 | 对数 | RMSE °C | 均值 RMSE | 空间 RMSE | 缺失区 RMSE | 保留区 RMSE | 困惑度 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        regions = r['regions']
        missing = regions.get('lst/corrupt', {}).get('rmse')
        lines.append(f"| {r['split']} | {r['label']} | {r['condition']} | {r['evaluated_pairs']} | {regions['lst/all']['rmse']:.4f} | {r['mean_rmse']:.4f} | {r['spatial_rmse']:.4f} | {f'{missing:.4f}' if missing is not None else '—'} | {regions['lst/retained']['rmse']:.4f} | {r['tokens']['lst']['perplexity']:.2f} |")
    lines += ['', f'进度：{len(rows)}/16 个完整评估单元。各单元目录含指标、逐条误差、成员与掩码摘要以及数据和检查点哈希。GPU 时间仅统计本次推理，不混入历史训练 GPU-hours。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')


@torch.inference_mode()
def main(args):
    output = args.output.resolve()
    if args.smoke:
        output = output/'smoke'
    output.mkdir(parents=True, exist_ok=True)
    install()
    torch.set_num_threads(4)
    with build_lock(output):
        start = time.perf_counter()
        record_status(output, status='preparing')
        members = prepare(output)
        baseline_config = json.loads((OLD/'pilot_lst_absolute_seed17/config.json').read_text())
        original_ds = PairedDataset(baseline_config, 'train')
        legacy_fingerprint = original_ds.fingerprint
        old_stats = copy.deepcopy(original_ds.sources['lst'].stats)
        consumed = np.fromiter((idx for _, idx in EpochSampler(len(original_ds), 17)), dtype=np.int64)[:128000]
        seen_ids = set(map(int, original_ds.record_ids[consumed]))
        assert len(seen_ids & members['val']) == 6563
        assert not seen_ids & members['test']
        original_ds.close()
        json_write(output/'protocol.json', {'prepared': PREPARED, 'counts': COUNTS, 'old_checkpoint_stats': old_stats,
            'training_steps': 2000, 'records_presented': 128000, 'actually_seen_val_records': 6563,
            'actually_seen_test_records': 0, 'historical_test_monitoring': True, 'new_training': False,
            'checkpoint_selection': 'fixed last.pt; no selection using this evaluation',
            'script_hashes': {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
            'scope': 'smoke only' if args.smoke else 'full val and test', 'started_utc': utc()})
        digests = {}
        for arm_idx, (arm, _, _) in enumerate(ARMS):
            checkpoint_path = OLD/f'pilot_lst_{arm}_seed17/last.pt'
            checkpoint = engine.read_checkpoint(checkpoint_path, data_fingerprint=legacy_fingerprint)
            assert checkpoint['step'] == 2000
            config = checkpoint['config']
            engine.seed_everything(config['seed'])
            device = engine.device_for(config)
            model = ExperimentModel(config).to(device).eval()
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            checkpoint_hash = sha256_file(checkpoint_path)
            native = model.codecs['lst']
            for split in ('val', 'test'):
                for condition, coverage in [('clean', 0), ('noisy10', .1)]:
                    destination = output/arm/split/condition
                    if (destination/'metrics.json').exists():
                        previous = json.loads((destination/'metrics.json').read_text())
                        assert previous['checkpoint_sha256'] == checkpoint_hash
                        assert previous['evaluated_pairs'] == (8 if args.smoke else COUNTS[split])
                        old_digest = digests.setdefault((split, condition), previous['input_digest'])
                        assert old_digest == previous['input_digest']
                        continue
                    destination.mkdir(parents=True, exist_ok=True)
                    ds = FullDataset(config, split, coverage, output/'mask_banks')
                    model.codecs['lst'] = UnitAdapter(native, old_stats, ds.sources['lst'].stats).to(device).eval()
                    subset = Subset(ds, range(8)) if args.smoke else ds
                    loader = DataLoader(subset, batch_size=4, num_workers=2, shuffle=False, pin_memory=True,
                        generator=torch.Generator().manual_seed(20260910))
                    metrics = EvaluationMetrics(ds.sources, model)
                    processed, mean_squared_sum = 0, 0.0
                    input_digest = hashlib.sha256()
                    unit_start = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats(device)
                    for batch in loader:
                        if (output/'STOP').exists():
                            raise RuntimeError('STOP requested; completed evaluation units remain resumable')
                        for name in ('image', 'input_mask', 'reference_mask', 'corruption_mask'):
                            input_digest.update(batch['lst'][name].numpy().tobytes())
                        input_digest.update(batch['record_id'].numpy().astype('<i8').tobytes())
                        moved = engine.move_batch(batch, device, model.codecs)
                        with engine.amp_context(config, device):
                            predictions = model(engine.inputs_only(moved, model.codecs))
                        metrics.add('lst', predictions['lst'], moved['lst'], batch['sample_id'], batch['tile_id'])
                        processed += len(batch['sample_id'])
                        if processed == len(batch['sample_id']) or processed % 128 == 0 or processed == len(subset):
                            mem = engine.memory_stats(device)
                            record_status(output, status='running', model=arm, split=split, condition=condition,
                                processed=processed, total=len(subset), elapsed_seconds=time.perf_counter()-unit_start,
                                completed_units=arm_idx*4+(0 if split=='val' else 2)+(condition=='noisy10'), memory=mem)
                            print(f'{utc()} {arm} {split}/{condition} {processed}/{len(subset)} {mem}', flush=True)
                            if mem['device_free_gib'] < 2:
                                raise RuntimeError('Less than 2 GiB free GPU memory')
                    digest = input_digest.hexdigest()
                    assert digests.setdefault((split, condition), digest) == digest, 'Models must receive identical input streams'
                    all_rows = [r for r in metrics.rows if r['region'] == 'all']
                    actual_ids = {int(r['sample_id'].rsplit(':', 1)[1]) for r in all_rows}
                    expected_ids = set(map(int, ds.record_ids[:8])) if args.smoke else members[split]
                    assert processed == len(subset) and actual_ids == expected_ids
                    assert len(all_rows) == 2*processed
                    mean_mse = float(np.mean([r['bias_physical']**2 for r in all_rows]))
                    total_mse = float(np.mean([r['mse_physical'] for r in all_rows]))
                    result = metrics.summary(50 if args.smoke else 500)
                    result.pop('selection_score', None)
                    result.pop('selection_definition', None)
                    result.update(model=arm, label=LABELS[arm_idx], split=split, condition=condition,
                        evaluated_pairs=processed, realizations=2, coverage=coverage,
                        mean_rmse=mean_mse**.5, spatial_rmse=max(0, total_mse-mean_mse)**.5,
                        checkpoint_sha256=checkpoint_hash, checkpoint_path=str(checkpoint_path), checkpoint_step=2000,
                        data_fingerprint=ds.fingerprint, input_digest=digest, old_checkpoint_stats=old_stats,
                        rates={'lst': model.codecs['lst'].rate()}, memory=engine.memory_stats(device),
                        elapsed_seconds=time.perf_counter()-unit_start, completed_utc=utc(),
                        independent_validation=False, historical_test_monitoring=True)
                    with (destination/'samples.jsonl').open('w', encoding='utf-8') as stream:
                        for row in metrics.rows:
                            stream.write(json.dumps(row, allow_nan=False)+'\n')
                    np.save(destination/'record_ids.npy', sorted(actual_ids), allow_pickle=False)
                    result['samples_sha256'] = sha256_file(destination/'samples.jsonl')
                    json_write(destination/'metrics.json', result)
                    ds.close()
                    del loader, predictions, moved, metrics
                    if not args.smoke:
                        summarize_report(output)
            del model, native
            gc.collect()
            torch.cuda.empty_cache()
        record_status(output, status='complete', completed_units=16, elapsed_seconds=time.perf_counter()-start,
                      inference_job_wall_hours=(time.perf_counter()-start)/3600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        out = args.output.resolve()/('smoke' if args.smoke else '')
        out.mkdir(parents=True, exist_ok=True)
        record_status(out, status='failed', error=repr(error))
        raise
