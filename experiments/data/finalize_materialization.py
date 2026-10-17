"""Summarize completed, independently verified full builds."""
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eo_data.core import json_write, sha256_file


def main():
    validation = json.loads((ROOT/'experiments/data/materialization_validation.json').read_text())
    assert validation['status'] == 'passed'
    audit = json.loads((ROOT/'experiments/data/source_audit.json').read_text())
    cleanup = json.loads((ROOT/'experiments/data/storage_cleanup.json').read_text(encoding='utf-8-sig'))
    assert cleanup['status'] == 'complete'
    archive_path = ROOT/'artifacts/prithvi_data_v1_20260909.zip'
    archive = json.loads(archive_path.with_suffix('.zip.verification.json').read_text())
    assert archive['status'] == 'passed' and archive_path.stat().st_size == archive['size']
    output = ROOT/'processed_data/v1'
    summary = {'status': 'ready', 'scope': 'full', 'training_scope': 'independent_datasets',
               'training_started': False, 'synthetic_noise': False, 'datasets': {}}
    table = ['| 数据源 / 模态 | train | val | test |', '|---|---:|---:|---:|']
    for name in ('prithvi', 'lstsr_tb'):
        directory = output/name
        bundle = json.loads((directory/'bundle.json').read_text())
        assert bundle['scope'] == 'full' and bundle['status'] == 'ready'
        info = json.loads((directory/'source_info.json').read_text())
        assert info['record_counts'] == audit['record_counts'][name]
        stats = json.loads((directory/'normalization.json').read_text())
        with sqlite3.connect((directory/'index.sqlite').as_uri()+'?mode=ro&immutable=1', uri=True) as db:
            rejected = {f'{split}/{mod}': n for split, mod, n in db.execute(
                'SELECT records.split,rejected.modality,COUNT(*) FROM rejected JOIN records ON records.id=rejected.record_id GROUP BY records.split,rejected.modality')}
        for key, count in info['candidate_samples'].items():
            assert bundle['samples'].get(key, {}).get('samples', 0) + rejected.get(key, 0) == count
        for mod in ('hls', 'lst_hr'):
            assert stats[mod]['samples'] == bundle['samples'][f'train/{mod}']['samples']
            counts = [str(bundle['samples'][f'{split}/{mod}']['samples']) if f'{split}/{mod}' in bundle['samples'] else '—'
                      for split in ('train', 'val', 'test')]
            table.append(f'| {name} / {mod} | '+ ' | '.join(counts)+' |')
        total_bytes = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
        summary['datasets'][name] = {'records': bundle['records'],
            'samples': {k: v['samples'] for k, v in bundle['samples'].items()},
            'rejected': rejected, 'referenced_source_bytes': bundle['source_bytes'],
            'prepared_directory_bytes': total_bytes, 'elapsed_seconds': bundle['elapsed_seconds'],
            'normalization': str(directory/'normalization.json'),
            'bundle_sha256': sha256_file(directory/'bundle.json'),
            'source_manifest_hashes': bundle['source_manifest_hashes']}
        if name == 'prithvi':
            assert bundle['all_packed_records_byte_verified']
            assert all(s['verified'] for s in bundle['shards'])
            assert sum(s['records'] for s in bundle['shards']) == bundle['records']
            original_bytes = bundle['source_bytes'] + sum(v['size'] for v in bundle['auxiliary'].values())
            working_bytes = total_bytes + archive['size']
            summary['compression'] = {'original_directory_bytes': original_bytes,
                'packed_frames_bytes': bundle['packed_bytes'], 'prepared_directory_bytes': total_bytes,
                'archive_path': str(archive_path),'archive_bytes':archive['size'],'archive_sha256':archive['sha256'],
                'working_bytes_including_archive_and_indices':working_bytes,
                'reduction_percent_including_indices': 100*(1-working_bytes/original_bytes),
                'shards': len(bundle['shards']), 'verified_records': bundle['records'],
                'original_directory_retained': (ROOT/'prithvi_data').exists(),
                'original_zip_retained': (ROOT/'prithvi_data.zip').exists(),
                'original_zip_bytes': cleanup['old_zip_bytes'],
                'old_original_plus_zip_bytes':original_bytes+cleanup['old_zip_bytes'],
                'net_logical_saved_bytes':original_bytes+cleanup['old_zip_bytes']-working_bytes,
                'original_data_deleted': not (ROOT/'prithvi_data').exists(),
                'loose_shards_deleted':not (directory/'packs').exists()}
            assert summary['compression']['original_data_deleted'] and not summary['compression']['original_zip_retained']
            assert summary['compression']['loose_shards_deleted']
    summary['validation_report'] = str(ROOT/'experiments/data/materialization_validation.json')
    summary['environment'] = json.loads((output/'environment.json').read_text())
    summary['frozen_preprocessing_source_sha256'] = sha256_file(output/'pipeline_source.zip')
    temporary = ROOT/'processed_data/restore_check'
    temporary_files = [p for p in temporary.rglob('*') if p.is_file()] if temporary.exists() else []
    summary['post_validation_temporary_restore'] = {
        'path':str(temporary),'files':len(temporary_files),'bytes':sum(p.stat().st_size for p in temporary_files),
        'removal_status':'blocked_by_automatic_approval' if temporary_files else 'absent',
        'tool_reason':'blocked by policy' if temporary_files else None}
    json_write(ROOT/'experiments/data/materialization_summary.json', summary)
    c = summary['compression']
    text = '# 全量预处理结果\n\n'
    text += '两套数据已分别完成全量预处理与读取验收，共用相同处理方法；未合并数据、未加入人工噪声、未启动模型训练。\n\n'
    text += '\n'.join(table)+'\n\n'
    text += (f"Prithvi 原目录逻辑体积 {c['original_directory_bytes']/2**30:.3f} GiB；"
             f"新单文件 ZIP {c['archive_bytes']/2**30:.3f} GiB，"
             f"含外部读取索引的完整工作集 {c['working_bytes_including_archive_and_indices']/2**30:.3f} GiB，"
             f"减少 {c['reduction_percent_including_indices']:.2f}%。"
             f"共有 {c['shards']} 个压缩分片，全部 {c['verified_records']:,} 条记录已逐条解压并核对原始字节 SHA-256。\n\n")
    text += (f"已按用户授权删除原始 Prithvi 解压目录、旧 ZIP 和独立分片；保留最新 ZIP 与必要的读取索引。"
             f"相对原目录和旧 ZIP 合计体积，净减少 {c['net_logical_saved_bytes']/2**30:.3f} GiB。"
             'LSTSR 复用现有 NPZ，不另存整套 float32 数组。\n\n')
    text += f"最新压缩包：`{archive_path}`。整个文件 SHA-256：`{c['archive_sha256']}`。压缩包内逐文件 SHA-256 校验与原目录变更检查均通过。\n\n"
    text += '使用入口见 [eo_data/README.md](../../eo_data/README.md)，一键查看四个训练 profile：`python examples/read_eo_data.py`。\n\n'
    text += '四组归一化参数分别位于两个数据集的 `normalization.json`，仅使用各自 train。HLS 按波段；LST 按原始整数编码空间，温度物理单位尚未完全确认。普通 VAE 与 Prithvi 架构 VAE 应共用同一 profile 的样本索引、参数和 mask。\n\n'
    text += '验收包括每个 split/modality 的真实样本读取与反归一化、Windows spawn 进程读入一致性及 8 条 Prithvi 原文件恢复。当前环境未安装 PyTorch，提供的 PyTorch DataLoader 适配器尚未在本环境执行。详细结果见 `materialization_validation.json`。\n'
    if temporary_files:
        text += f"\n最后一轮恢复测试新生成的临时目录仍保留（{summary['post_validation_temporary_restore']['bytes']/2**20:.2f} MiB）；自动审批拒绝清理，工具只返回 `blocked by policy`。此目录不参与训练。\n"
    (ROOT/'experiments/data/MATERIALIZATION_RESULT.md').write_text(text, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
