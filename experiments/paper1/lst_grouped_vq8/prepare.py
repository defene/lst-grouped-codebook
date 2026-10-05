"""Freeze three exact-budget arms and reuse fingerprint-checked A05 noise banks."""
from pathlib import Path
import copy
import math
import numpy as np
from eo_data.core import sha256_file
from experiments.paper1.lst_grouped_vq.io import read, json_write
from experiments.paper1.lst_grouped_vq.data import DatasetV2, COUNTS
from experiments.paper1.lst_grouped_vq.train import source_hashes

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT/'runs/paper1/lst_grouped_vq8_20260928'
ARMS = ['g8k16','g8k256']
SETTINGS = [(8,16),(8,256)]
LABELS = ['8组×16码字（与g4同码率）','8组×256码字（两倍索引码率）']


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = ROOT/'runs/paper1/lst_v2_5epoch_20260915'
    old_manifest = read(previous/'manifest.json')
    current = source_hashes()
    baseline = ROOT/'runs/paper1/lst_grouped_vq_20260924'
    assert current == read(baseline/'manifest.json')['sources'], 'A06 frozen core changed'
    assert read(baseline/'completion_review.json')['status']=='passed'
    assert all(current[k] == v for k,v in old_manifest['sources'].items()), 'Historical frozen core changed'
    old = read(previous/'full/full_lst_v2_residual_spatial4_seed17/config.json')
    configs, prints, member_sets, tiles = {}, {}, {}, {}
    for phase in ('smoke','full'):
        for arm,(g,k) in zip(ARMS, SETTINGS):
            c = copy.deepcopy(old)
            c['name'] = f'{phase}_lst_grouped_{arm}_seed17'
            c['model'].update(groups=g,codes_per_group=k,quantizer='var_vq' if g==1 else 'grouped_var_vq')
            # codebook_size stays 1024 during native network construction for matched backbone initialization.
            c['data'].update(train_limit=67 if phase=='smoke' else None,
                             val_limit=8 if phase=='smoke' else None,test_limit=8 if phase=='smoke' else None)
            n,epochs = (67,2) if phase=='smoke' else (COUNTS['train'],5)
            c['train'].update(steps=math.ceil(n/64)*epochs,warmup=1 if phase=='smoke' else 500,
                              log_every=1 if phase=='smoke' else 20,save_every=1 if phase=='smoke' else 250,
                              eval_every=math.ceil(n/64))
            c['eval']['bootstrap_replicates'] = 50 if phase=='smoke' else 500
            c['experiment'].update(id='A07',phase=phase,epochs=epochs,output_root=str(OUTPUT),
                                   codebook_ablation='channel_group_product_quantization_before_full_channel_Phi',
                                   primary_comparison='A06_g4k256_vs_g8k16_same_rate_and_g8k256_expanded_rate')
            path = HERE/'configs'/phase/(c['name']+'.json')
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                assert read(path) == c
            else:
                json_write(path,c)
            configs[str(path.relative_to(HERE))] = sha256_file(path)
            for split in COUNTS:
                ds = DatasetV2(c,split,.1)
                sample = ds[0]['lst']
                assert np.isfinite(sample['image']).all()
                assert not sample['image'][~sample['input_mask'].astype(bool)].any()
                if phase=='full':
                    assert len(ds)==COUNTS[split]
                    assert prints.setdefault(split,ds.fingerprint)==ds.fingerprint
                    member_sets[split] = set(map(int,ds.record_ids))
                ds.close()
    for split in COUNTS:
        tiles[split] = set((ROOT/f'data_inspection/hlp_split_20260914/{split}_tiles.txt').read_text().splitlines())
        assert prints[split]['stats'] == prints['train']['stats']
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not member_sets[a]&member_sets[b] and not tiles[a]&tiles[b]
    old_prints = read(baseline/'data_preflight.json')['fingerprints']
    assert prints == old_prints, 'A06 input/normalization fingerprint changed'
    banks = {p.name:sha256_file(p) for p in (previous/'mask_banks').iterdir() if p.is_file()}
    json_write(OUTPUT/'data_preflight.json',dict(status='passed',counts=COUNTS,
        fingerprints=prints,disjoint_ids_and_tiles=True,configs=configs,
        reused_A05_split_local_mask_banks=banks,normalization=prints['train']['stats'],
        formal_records_per_model=1094875,formal_updates=17110,tail_records=31))
    print('PASS: two eight-group arms, HLP v2 train-only normalization, exact splits and A05 noise banks.')


if __name__=='__main__':
    main()
