"""Publishable source/report snapshot; excludes datasets, checkpoints and environments."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'artifacts/grouped_codebook_repository_20261004'
ALLOW = {'.py','.json','.md','.txt','.yaml','.yml','.toml','.ps1','.sh','.cfg','.ini'}
SKIP = {'__pycache__','.git','node_modules','.pytest_cache'}
copied = {}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy(rel, target=None):
    src = ROOT/rel
    dstrel = target or rel
    dst = DEST/dstrel
    if not src.is_file():
        raise FileNotFoundError(src)
    if src.is_symlink() or src.name.startswith('.env'):
        raise ValueError(src)
    if src.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors','.eoz','.npz','.npy','.sqlite','.zip','.gz'}:
        raise ValueError(f'Non-report payload: {src}')
    if dst.exists() and sha(src) == sha(dst):
        return
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    copied[str(dstrel).replace('\\','/')] = dict(source=str(rel).replace('\\','/'),sha256=sha(dst),bytes=dst.stat().st_size)


def main():
    # Preserve the original A06/A07 export bytes, including its older AGENTS document.
    archive=DEST/'docs/historical_export_20261004/AGENTS.md'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        shutil.copy2(DEST/'AGENTS.md',archive)
    for tree in ('eo_data','eo_denoise','experiments','tests','tools','docs','examples'):
        for folder, dirs, files in os.walk(ROOT/tree):
            dirs[:]=[d for d in dirs if d not in SKIP and not (Path(folder)/d).is_symlink()]
            for filename in files:
                p=Path(folder)/filename
                if p.suffix.lower() not in ALLOW:
                    continue
                rel=p.relative_to(ROOT)
                # The repository's registry is a frozen A06/A07 subset, not the full workspace registry.
                if rel.as_posix()=='experiments/paper1/registry.json':
                    copy(rel,Path('docs/workspace_registry_20261017.json'))
                else:
                    copy(rel)
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in {'.py','.ps1','.md'} and p.name!='README.md':
            copy(p.relative_to(ROOT))
    copy(Path('README.md'),Path('docs/WORKSPACE_README_20261017.md'))
    for rel in ('data_inspection/FINAL_DATASET_SUMMARY.md','data_inspection/HLP_SOURCE_PROVENANCE.md',
                'data_inspection/hlp_split_20260914/README.md','data_inspection/hlp_split_20260914/split_summary.json'):
        copy(Path(rel))

    report_sets = {
        'torch_experiments_20261013':['RESULTS.txt','results_index.json','preview_clean_G.png','preview_pixel10_G.png','download_verification.json'],
        'loss_diagnostics_20261014':['ALL_39_LOSS_SUMMARY.txt','FINDINGS.txt','diagnostics.json','codebook_weight_scales.json','mechanism_probes.json','anomalies_overview.png','grouped_g8k256_loss_components.png','sd_loss_components.png','train_vs_validation.png','vqgan_pretrained_loss_components.png','vq_drifting_loss_components.png','vq_loss_components.png'],
        'grouped_vs_vq_20261014':['COMPARISON.txt','comparison.json','bottleneck_curves.png','G_per_scale_usage.png','pixel_curves.png'],
        'training_lowpoints_20261014':['ALL_39_LOWPOINTS.txt','server_checkpoint_inventory.json','training_lowpoints.json'],
        'local_cases50_all39_20261014':['RESULTS.txt','summary.csv','summary.json','family_best_on_50_cases.json','metric_formula_crosscheck.json','per_case_metrics.csv','preview_case01_family_best.png'],
        'ema_local_pilot_20261014_attempt01':['RESULTS.txt','comparison.json','results.json','analysis_verification.json','selected_checkpoints.json','curves.png','grouped_perplexity.png'],
        'ema_k512_checks_20261015':['tests_verified.xml'],
        'ema_full_checks_20261015':['tests_passed.xml'],
    }
    for folder,names in report_sets.items():
        for name in names:
            copy(Path('outputs')/folder/name)
    # Configurations and validation receipts are evidence; all checkpoints and step logs stay local.
    for folder in ('ema_local_full_20261015_attempt01','ema_local_k512_full_20261015_attempt01'):
        base=ROOT/'outputs'/folder
        for name in ('queue.json','queue_status.json','scheduler_status.json','manifest.json','cpu_model_checks.json','README.txt','STOP'):
            if (base/name).is_file():copy((base/name).relative_to(ROOT))
        for sub in ('configs','preflight'):
            for p in (base/sub).glob('*.json'):copy(p.relative_to(ROOT))
        for run in (base/'runs').glob('*'):
            if not run.is_dir():continue
            for name in ('config.json','status.json','selection.json','training_complete.json','data_fingerprint.json','source_hashes.json','model.json'):
                if (run/name).is_file():copy((run/name).relative_to(ROOT))
            for p in (run/'epoch_validation').glob('epoch_*.json'):copy(p.relative_to(ROOT))

    # Generate a compact index from the committed, full-validation receipts.
    formal=ROOT/'outputs/ema_local_full_20261015_attempt01'
    rows=[]
    evidence=[]
    for name in ('vq_G_baseline','vq_G_ema','vq_G_ema_maint'):
        run=formal/'runs'/name
        s=json.loads((run/'selection.json').read_text())
        p=run/'epoch_validation'/f"epoch_{s['best']['epoch']:02d}.json"
        receipt=json.loads(p.read_text())
        cells={c['profile']['kind']:c for c in receipt['cells']}
        usage=cells['pixel10']['code_usage']['groups'][0]['overall']
        complete=(run/'training_complete.json').exists()
        rows.append(f"| {name} | {'5/5，已完成' if complete else '1/5，已停止'} | {s['best']['epoch']} | {cells['clean']['raw']['pixel_weighted_rmse']:.6f} | {cells['pixel10']['raw']['pixel_weighted_rmse']:.6f} | {cells['pixel10']['raw']['pixel_weighted_mae']:.6f} | {usage['used_codes']}/1024 |")
        evidence.append(dict(run=name,complete=complete,selection=s['best'],receipt=p.relative_to(ROOT).as_posix(),receipt_sha256=sha(p)))
    report=DEST/'reports/local_vq_ema_20261017'
    report.mkdir(parents=True,exist_ok=True)
    (report/'README.md').write_text('''# 本地 VQ / Grouped / EMA 实验状态（2026-10-17）

全部训练已按用户要求停止；本次提交只同步源码、配置和现有报告，不启动或恢复任何实验。

## 已有正式结果

HLP v2：train 218,975，val 11,684；G 表示，LST + mask，oracle clean mean 条件。
每轮完整 val，clean/pixel10 各两个固定视图；按 pixel10 raw 有效像素加权摄氏度 RMSE 选 best，平分取早轮次。未用 test 选模。
正式预算为 5 个完整 epoch、有效 batch64（micro8），每轮3422次更新、尾批31条。

| 模型 | 已验证预算及状态 | best epoch | clean RMSE °C | pixel10 RMSE °C | pixel10 MAE °C | pixel10 已用码 |
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

原始 VQ 与纯 EMA 均完成 5 轮，best 都是最终第5轮。EMA+K-means+死码重置仅完成第1轮验证；不能与5轮结果作等预算结论。
同为第1轮，纯EMA的pixel10 RMSE为0.789700°C，初始化+死码重置组为0.883844°C；后者同时增加两个机制，不能据此单独归因。
Grouped K256 三组未开始正式训练。K512 的 VQ 与 Grouped 各三组只完成配置与CPU检查，GPU预检和正式训练均未开始。
Grouped K512 表示8组、每组512码；内部旧路由名 grouped_g8k256 不代表实际K。它与普通VQ不是等码率比较。

## 报告入口

- [正式实验逐轮指标、选模与配置](../../outputs/ema_local_full_20261015_attempt01/)
- [K512配置与停止状态](../../outputs/ema_local_k512_full_20261015_attempt01/)
- [EMA六组短程pilot](../../outputs/ema_local_pilot_20261014_attempt01/RESULTS.txt)：128更新、2048训练样本呈现、64条验证子集，不能与完整预算结果混用。
- [39模型历史结果索引](../../outputs/torch_experiments_20261013/results_index.json)：历史最终轮次评估，不能重新标记为逐轮验证选出的最优模型。
- [39模型loss审查](../../outputs/loss_diagnostics_20261014/FINDINGS.txt)
- [Grouped与普通VQ对照](../../outputs/grouped_vs_vq_20261014/COMPARISON.txt)
- [训练低点及checkpoint存留审查](../../outputs/training_lowpoints_20261014/ALL_39_LOWPOINTS.txt)
- [本地50例报告](../../outputs/local_cases50_all39_20261014/RESULTS.txt)：子集诊断，不是完整独立验证。
- [数据集报告](../../data_inspection/FINAL_DATASET_SUMMARY.md)

## 发布边界

新增内容包含源代码、研究配置、文本/表格/图表报告和验证汇总；不包含原始数据、prepared数据、预测数组、模型权重、训练断点、完整训练日志、虚拟环境或缓存。
既有A06/A07仓库文件保持历史语义。原始导出清单保留；旧AGENTS字节另存于 docs/historical_export_20261004，verify_export.py验证这一历史副本。
仓库原有A06/A07 registry保持冻结；当前工作区完整registry另存为 docs/workspace_registry_20261017.json，里面的旧数据口径按历史版本解释。

这是审计与源码同步，不是迁移后可直接续训的包。配置保留原始Windows绝对路径和哈希，依赖数据及checkpoint不在本次上传中。
服务器实际部署需另行绑定路径、环境与数据，重新做预检；不要直接运行旧启动命令。
两个本机队列的 STOP 标记保留。未提交SSH作业，也未恢复本地训练。
''',encoding='utf-8')
    (report/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    manifest=dict(scope='Current source additions and reports only; no datasets/checkpoints; all queues stopped',files=copied)
    (DEST/'UPDATE_MANIFEST_20261017.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(copied_files=len(copied),mib=sum(x['bytes'] for x in copied.values())/1024**2),indent=2))


if __name__=='__main__':main()
