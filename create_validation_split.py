"""Create a reproducible tile-disjoint validation split from canonical train."""
from pathlib import Path
import csv
import hashlib
import json
import os
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719'
OUT = BASE / 'splits'
SEED = 20260905
FRACTION = 0.05
FILES = ['phase4_dataset_manifest.csv', 'phase3_tile_variant_months_merged.csv',
         'hls_l8_conditions_2019_2025_lc08/hls_condition_manifest.csv']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    report_path = OUT / 'split_summary.json'
    if report_path.exists():
        raise RuntimeError('Split already created; inspect splits/split_summary.json before changing it.')
    a = pd.read_csv(BASE / FILES[0], usecols=['sequence_id','mgrs_tile','split','available_month_count'])
    assert set(a.split) == {'train','test'}
    assert a.sequence_id.is_unique
    assert a.groupby('mgrs_tile').split.nunique().max() == 1
    eligible = sorted(a.loc[a.split == 'train','mgrs_tile'].unique())
    ranked = sorted(eligible, key=lambda t: (hashlib.sha256(f'{SEED}:{t}'.encode()).hexdigest(), t))
    nval = round(len(eligible) * FRACTION)
    val_tiles = set(ranked[:nval])
    old_test = set(a.loc[a.split == 'test','sequence_id'])
    old_train = set(a.loc[a.split == 'train','sequence_id'])
    original = a.copy()
    a.loc[(a.split == 'train') & a.mgrs_tile.isin(val_tiles),'split'] = 'val'
    assert set(a.loc[a.split == 'test','sequence_id']) == old_test
    assert set(a.loc[a.split.isin(['train','val']),'sequence_id']) == old_train
    assert a.groupby('mgrs_tile').split.nunique().max() == 1
    summary = a.groupby('split').agg(sequences=('sequence_id','size'),tiles=('mgrs_tile','nunique'),
                                   usable_months=('available_month_count','sum')).to_dict('index')
    prepared = []
    for relative in FILES:
        p = BASE / relative
        temp = p.with_suffix('.csv.split_pending')
        count = 0
        with p.open('r',encoding='utf-8-sig',newline='') as src, temp.open('w',encoding='utf-8',newline='') as dst:
            reader = csv.reader(src); writer = csv.writer(dst)
            header = next(reader); writer.writerow(header)
            ti, si = header.index('mgrs_tile'), header.index('split')
            for row in reader:
                if row[si] == 'train' and row[ti] in val_tiles:
                    row[si] = 'val'; count += 1
                writer.writerow(row)
        # Verify every field, including multiline metadata, before replacing any manifest.
        with p.open(encoding='utf-8-sig',newline='') as src, temp.open(encoding='utf-8',newline='') as dst:
            left, right = csv.reader(src), csv.reader(dst)
            assert next(left) == next(right)
            for before in left:
                after = next(right)
                expected = before.copy()
                if before[si] == 'train' and before[ti] in val_tiles:
                    expected[si] = 'val'
                assert after == expected
            assert next(right,None) is None
        prepared.append({'file':relative,'renamed_rows':count,'before_sha256':sha(p),'after_sha256':sha(temp)})
    b = pd.read_csv((BASE / FILES[2]).with_suffix('.csv.split_pending'),usecols=['sequence_id','split'])
    pd.testing.assert_series_equal(a.set_index('sequence_id').split.sort_index(),b.set_index('sequence_id').split.sort_index())
    c = pd.read_csv((BASE / FILES[1]).with_suffix('.csv.split_pending'),usecols=['mgrs_tile','split'])
    assert c.groupby('mgrs_tile').split.nunique().max() == 1
    cm = c.drop_duplicates('mgrs_tile').set_index('mgrs_tile').split
    am = a.drop_duplicates('mgrs_tile').set_index('mgrs_tile').split
    assert (cm.reindex(am.index) == am).all()
    OUT.mkdir(exist_ok=True)
    for split in ['train','val','test']:
        subset = a[a.split == split]
        subset[['sequence_id','mgrs_tile','available_month_count']].sort_values('sequence_id').to_csv(OUT/f'{split}_sequences.csv',index=False)
        (OUT/f'{split}_tiles.txt').write_text('\n'.join(sorted(subset.mgrs_tile.unique()))+'\n',encoding='utf-8')
    for record in prepared:
        p = BASE / record['file']; os.replace(p.with_suffix('.csv.split_pending'),p)
    report = {'date':'2026-09-05','method':'Rank sorted original-train MGRS tile IDs by SHA256(seed:tile); select first round(0.05*N).',
              'seed':SEED,'target_train_tile_fraction':FRACTION,'eligible_train_tiles':len(eligible),
              'selected_val_tiles':nval,'counts':summary,'test_membership_unchanged':True,
              'all_non_split_csv_fields_unchanged':True,'pairwise_tile_id_overlap':0,
              'spatial_footprint_overlap_checked':False,'files':prepared}
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    config_path = BASE / 'hls_l8_conditions_2019_2025_lc08/hls_condition_config.json'
    config = json.loads(config_path.read_text())
    for key,record in zip(['patch_manifest_sha256','tile_variant_months_sha256'],prepared):
        config['source_files'][key] = record['after_sha256']
    config['internal_validation_split'] = {'seed':SEED,'method':'sha256_tile_ranking','train_tile_fraction':FRACTION,'summary':'splits/split_summary.json'}
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    profile_path = ROOT / 'data_inspection/inspection.json'
    profile = json.loads(profile_path.read_text())
    profile['splits'] = {s:{'sequences':v['sequences'],'tiles':v['tiles'],'months':v['usable_months']} for s,v in summary.items()}
    mapping = a.set_index('sequence_id').split
    for sample in profile.get('array_samples',[]): sample['split'] = mapping[sample['sequence_id']]
    profile['internal_validation_split'] = report
    profile_path.write_text(json.dumps(profile,indent=2),encoding='utf-8')
    lines = ['# Dataset split protocol','',f'Seed: {SEED}. Select {nval} of {len(eligible)} original-train MGRS tiles (approximately 5%) using SHA256(seed:tile) ranking. All sequences and months from a selected tile belong to val. Existing test membership stays fixed.','',
             '| Split | Sequences | MGRS tiles | Usable months |','|---|---:|---:|---:|']
    for split in ['train','val','test']:
        v=summary[split]; lines.append(f"| {split} | {v['sequences']:,} | {v['tiles']:,} | {v['usable_months']:,} |")
    lines += ['', 'The three manifests contain the current split labels. Membership lists and hashes are stored alongside this file. Use train for fitting, val for model selection, and test for final evaluation. Tile IDs are disjoint; neighboring-tile geographic footprint overlap has not been assessed.','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    index = BASE/'DATASET_INDEX.md'
    text=index.read_text(encoding='utf-8')
    text+='\n## Internal validation split (2026-09-05)\n\nThe current train/val/test split supersedes the two-way counts above. See [split protocol](splits/README.md) and [exact membership](splits/). Test membership is unchanged.\n'
    index.write_text(text,encoding='utf-8')
    plan=ROOT/'ICLR_CROSS_VIEW_DISCRETE_TOKENIZATION_PLAN.md'
    text=plan.read_text(encoding='utf-8')
    v=summary['val'];t=summary['train']
    text=text.replace('- 内部验证划分另行确定，不将原始 test 用作内部验证；',
        f"- 从原始 train 按 MGRS tile 划出 val：{v['sequences']:,} 个 sequence、{v['usable_months']:,} 个 image-month、{v['tiles']} 个 tile；\n- 划分后实际 train：{t['sequences']:,} 个 sequence、{t['usable_months']:,} 个 image-month、{t['tiles']} 个 tile；")
    text=text.replace('### 2.4 数据划分\n','### 2.4 数据划分\n\n当前划分已落实到三个 manifest 的 `split` 列。固定 seed 为 `20260905`，对原始 train tile 按 `SHA256(seed:tile)` 排序，取前 118 个 tile 作为 val（约 5%）。同一 tile 的所有 sequence 和月份归入同一 split；原始 test 成员不变。名单与校验记录见数据目录 `splits/`。\n')
    plan.write_text(text,encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
