"""Inference-only new cases for the complete A06/A07 grouped-codebook comparison."""
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from eo_data.core import sha256_file, json_write
from eo_denoise import engine
from experiments.paper1.lst_grouped_vq.runtime import ExperimentModel
from experiments.paper1.lst_grouped_vq.data import DatasetV2
from experiments.paper1.lst_grouped_vq.train import source_hashes
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/'runs/paper1/lst_grouped_vq_20260924'
NEW = ROOT/'runs/paper1/lst_grouped_vq8_20260928'
OUTPUT = ROOT/'runs/paper1/lst_grouped_summary_20261002'
ARMS = ['single1024','g2k32','g4k256','g8k16','g8k256']
LABELS = ['单码本1024\n6832 bit/图','2组×32\n6832 bit/图','4组×256\n21792 bit/图','8组×16\n21792 bit/图','8组×256\n43552 bit/图']
from experiments.paper1.lst_grouped_vq.io import read


@torch.inference_mode()
def main():
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(4)
    assert source_hashes() == read(NEW/'manifest.json')['sources']
    OUTPUT.mkdir(exist_ok=True)
    excluded = [231773,232129,232414,232565,232620,233073,233356,234070]
    selection_source = OLD/'full/full_lst_grouped_single1024_seed17/final_test_clean/record_ids.npy'
    pool = [int(x) for x in np.load(selection_source) if int(x) not in excluded]
    salt = 'grouped-vq-case-20261002:'
    ids = sorted(pool,key=lambda x:(hashlib.sha256((salt+str(x)).encode()).hexdigest(),x))[:8]
    selection_path = OUTPUT/'selection.json'
    selection = dict(record_ids=ids,excluded_previous_ids=excluded,pool_size=len(pool),
        rule='First eight by (SHA256(salt + decimal record ID), record ID); no metric or image selection',
        salt=salt,source=str(selection_source),source_sha256=sha256_file(selection_source))
    if selection_path.exists(): assert read(selection_path)==selection
    else: json_write(selection_path,selection)
    dest = OUTPUT/'gallery'
    dest.mkdir(exist_ok=True)
    pictures, input_hashes, checks, checkpoint_hashes = {}, {}, [], {}
    for arm in ARMS:
        run_root = OLD if arm in ARMS[:3] else NEW
        run = run_root/'full'/f'full_lst_grouped_{arm}_seed17'
        state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
        assert state['step'] == 17110 and state['source_hashes'] == source_hashes()
        config = state['config']
        checkpoint_hashes[arm] = sha256_file(run/'last.pt')
        engine.seed_everything(config['seed'])
        device = engine.device_for(config)
        model = ExperimentModel(config).to(device).eval()
        model.load_state_dict(state['model'], strict=True)
        del state
        for condition, coverage in [('clean', 0), ('noisy10', .1)]:
            ds = DatasetV2(config, 'test', coverage)
            pos = {int(r): i for i, r in enumerate(ds.record_ids)}
            entries = [ds[pos[r]] for r in ids]
            digest = hashlib.sha256()
            for entry in entries:
                for field in ('image', 'input_mask', 'target', 'reference_mask', 'corruption_mask'):
                    digest.update(entry['lst'][field].tobytes())
            assert input_hashes.setdefault(condition, digest.hexdigest()) == digest.hexdigest()
            pred_parts = []
            for start in range(0, len(ids), 4):
                batch = entries[start:start+4]
                image = torch.from_numpy(np.stack([x['lst']['image'][0] for x in batch])).to(device)
                mask = torch.from_numpy(np.stack([x['lst']['input_mask'][0] for x in batch])).to(device)
                with engine.amp_context(config, device):
                    pred = model.codecs['lst'](image, mask)['reconstruction'].float()
                pred_parts.append(pred.cpu().numpy())
            preds = np.concatenate(pred_parts)
            std = np.asarray(ds.sources['lst'].stats['std'], np.float32)[:, None, None]
            mean = np.asarray(ds.sources['lst'].stats['mean'], np.float32)[:, None, None]
            target = np.stack([x['lst']['target'] for x in entries])
            valid = np.stack([x['lst']['reference_mask'][0] for x in entries]).astype(bool)
            samples = {}
            for line in (run/f'final_test_{condition}/samples.jsonl').read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                rid = int(row['sample_id'].rsplit(':', 1)[1])
                if rid in ids and row['region'] == 'all' and row['view'] == 0:
                    samples[rid] = row
            for i, rid in enumerate(ids):
                mse = float(np.mean((((preds[i]-target[i])*std/100)[:, valid[i]]).astype(np.float64)**2))
                expected = samples[rid]['mse_physical']
                assert np.isclose(mse, expected, rtol=.002, atol=1e-7), (arm, condition, rid, mse, expected)
                checks.append(dict(model=arm, condition=condition, record_id=rid, view=0,
                                   gallery_mse=mse, full_evaluation_mse=expected))
            pictures[f'{arm}_{condition}'] = (preds*std+mean)/100
            if arm == ARMS[0]:
                pictures['target'] = (target*std+mean)/100
                pictures['reference_mask'] = valid
                pictures[condition+'_input'] = (np.stack([x['lst']['image'][0] for x in entries])*std+mean)/100
                pictures[condition+'_mask'] = np.stack([x['lst']['input_mask'][0, 0] for x in entries]).astype(bool)
            ds.close()
        del model, pred, image, mask
        gc.collect()
        torch.cuda.empty_cache()
    np.savez_compressed(dest/'predictions_8_records.npz', record_ids=ids, **pictures)
    font = 'C:/Windows/Fonts/msyh.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(), 'axes.unicode_minus': False})
    artifacts = []
    for condition in ('clean', 'noisy10'):
        for page in range(2):
            fig, axes = plt.subplots(4, 7, figsize=(20, 11.8), layout='constrained')
            for row, i in enumerate(range(page*4, page*4+4)):
                valid = pictures['reference_mask'][i]
                target = pictures['target'][i, 0]
                lo, hi = np.quantile(target[valid], [.01, .99])
                if hi-lo < .01:
                    lo, hi = lo-.5, hi+.5
                values = [target, pictures[condition+'_input'][i, 0]]+[pictures[f'{a}_{condition}'][i, 0] for a in ARMS]
                masks = [valid, pictures[condition+'_mask'][i]]+[valid]*len(ARMS)
                for col, (value, support) in enumerate(zip(values, masks)):
                    cmap = plt.get_cmap('inferno').copy()
                    cmap.set_bad('#777777')
                    im = axes[row, col].imshow(np.ma.masked_where(~support, value), vmin=lo, vmax=hi,
                                               cmap=cmap, interpolation='nearest')
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    if row == 0:
                        axes[row, col].set_title((['参考温度', '实际输入']+LABELS)[col], fontsize=10)
                    if col >= 2:
                        check = next(x for x in checks if x['model']==ARMS[col-2] and x['condition']==condition and x['record_id']==ids[i])
                        axes[row, col].set_xlabel(f"RMSE {np.sqrt(check['full_evaluation_mse']):.3f}°C", fontsize=9)
                    if col == 0:
                        axes[row, col].set_ylabel(f'记录 {ids[i]}', fontsize=9)
                fig.colorbar(im, ax=axes[row, :], shrink=.8, pad=.008, label='°C')
            fig.suptitle(f'HLP v2 · 组合码本全部5配置 · 新8个case · 固定epoch5 · {"干净输入" if condition == "clean" else "约10%缺失输入"}', fontsize=14)
            fig.supxlabel('同一行共用参考温度1%–99%色标（超出范围截色）；灰色为无效或缺失。固定哈希抽样，排除旧8条；view0。图下RMSE仅对应该记录，非全量指标。', fontsize=9)
            path = dest/f'predictions_{condition}_{page+1}.png'
            fig.savefig(path, dpi=160)
            plt.close(fig)
            artifacts.append(dict(path=path.name, sha256=sha256_file(path)))
    json_write(dest/'review.json', dict(status='passed', image_record_ids=ids, image_view=0,
        selection_source=str(selection_path), selection_sha256=sha256_file(selection_path),
        input_hashes=input_hashes, checkpoint_hashes=checkpoint_hashes, checks=checks, artifacts=artifacts,
        predictions_sha256=sha256_file(dest/'predictions_8_records.npz'),
        started_utc=started_utc, completed_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-started,
        timing_scope='Additional inference/plotting wall time; separate from formal training/evaluation GPU hours.'))
    print(json.dumps(dict(status='complete', destination=str(dest), records=ids), ensure_ascii=False))


if __name__ == '__main__':
    main()
