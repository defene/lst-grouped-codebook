"""Export A06/A07 code and auditable results without changing historical runs."""
from pathlib import Path
import argparse, gzip, hashlib, json, shutil, ast

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); args=ap.parse_args()
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()): raise RuntimeError('Destination must be empty')
    out.mkdir(parents=True,exist_ok=True)
    entries=[]
    def copy(p, compress=False):
        rel=p.relative_to(ROOT); dst=out/rel
        if compress: dst=dst.with_name(dst.name+'.gz')
        dst.parent.mkdir(parents=True,exist_ok=True)
        raw=p.read_bytes()
        if compress: dst.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
        else: dst.write_bytes(raw)
        entries.append({'path':dst.relative_to(out).as_posix(),'source_path':rel.as_posix(),
          'source_sha256':hashlib.sha256(raw).hexdigest(),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),
          'bytes':dst.stat().st_size,'encoding':'gzip' if compress else 'identity'})
    folders=['eo_data','eo_denoise','experiments/paper1/lst_grouped_vq',
             'experiments/paper1/lst_grouped_vq8','experiments/paper1/lst_grouped_summary',
             'experiments/paper1/lst_v2_5epoch','experiments/paper1/lst_residual']
    for folder in folders:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.json','.md','.txt'}: copy(p)
    for name in ['experiments/paper1/codebook1024/run_experiment.py',
                 'experiments/paper1/codebook1024/make_configs.py',
                 'experiments/paper1/var_vs_prithvi_ae/run_suite.py',
                 'AGENTS.md','experiments/paper1/training_budget.json','experiments/data/dataset_catalog.json',
                 'data_inspection/FINAL_DATASET_SUMMARY.md','docs/MIGRATION.md',
                 'tests/test_lst_grouped_vq.py','tests/test_lst_grouped_vq8.py',
                 'tests/test_lst_v2_epochs.py','tests/test_lst_residual.py',
                 'tools/export_grouped_codebook.py',
                 'output/pdf/grouped_codebook_A06_A07_summary_20261002.pdf']:
        copy(ROOT/name)
    for p in sorted((ROOT/'data_inspection/hlp_split_20260914').iterdir()):
        if p.is_file(): copy(p, p.suffix=='.csv')
    for name in ['experiments/__init__.py','experiments/paper1/__init__.py',
                 'experiments/paper1/codebook1024/__init__.py','experiments/paper1/var_vs_prithvi_ae/__init__.py']:
        if (ROOT/name).exists():copy(ROOT/name)
    run_names=['lst_grouped_vq_20260924','lst_grouped_vq8_20260928','lst_grouped_summary_20261002']
    for name in run_names:
        base=ROOT/'runs/paper1'/name
        for p in sorted(base.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            rel=p.relative_to(base)
            if p.suffix in {'.json','.md','.png','.npy','.npz','.xml'}: copy(p)
            elif p.suffix=='.jsonl' and (len(rel.parts)==1 or p.name=='train.jsonl' or any(x.startswith('final_') for x in rel.parts)):
                copy(p,True)
    # Original registry is broader than this release; retain only the two trial records.
    reg=json.loads((ROOT/'experiments/paper1/registry.json').read_text(encoding='utf-8'))
    selected={k:v for k,v in reg.items() if k in {'lst_grouped_vq_trial','lst_grouped_vq8_trial'}}
    (out/'experiments/paper1/registry.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    (out/'.gitignore').write_text('__pycache__/\n*.py[cod]\n.pytest_cache/\n.venv*/\n.env\n.env.*\n/data/\n/models/\n/processed_data/\n*.pt\n*.pth\n*.ckpt\n*.safetensors\n*.eoz\n/tmp/\n',encoding='utf-8')
    (out/'requirements.txt').write_text('torch\ntorchvision\ntimm\neinops\nnumpy\nscipy\nmatplotlib\nzstandard\nPyYAML\npytest\n',encoding='utf-8')
    (out/'README.md').write_text('''# LST 组合码本实验：A06 / A07

本快照包含 single1024、g2k32、g4k256、g8k16、g8k256 五个已完成配置的实验框架和结果；没有启动新训练。

- [完整中文汇总](runs/paper1/lst_grouped_summary_20261002/REPORT_ZH.md)
- [9页PDF](output/pdf/grouped_codebook_A06_A07_summary_20261002.pdf)
- [A06原始报告](runs/paper1/lst_grouped_vq_20260924/REPORT_ZH.md) / [A07原始报告](runs/paper1/lst_grouped_vq8_20260928/REPORT_ZH.md)
- [A06协议](experiments/paper1/lst_grouped_vq/PROTOCOL_ZH.md) / [A07协议](experiments/paper1/lst_grouped_vq8/PROTOCOL_ZH.md)

## 范围与主要结论

HLP v2，train 218975 / val 11684 / test 11757；仅 LST + mask。seed17，从零5完整epoch，17110更新/模型，固定epoch5权重，两视图完整评估。A07复用的A06 g4结果不重复计数，合计20个唯一评估单元。

| 配置 | bit/图含均温 | test clean RMSE °C | test noisy10 RMSE °C |
|---|---:|---:|---:|
| single1024 | 6832 | 0.948149 | 0.985072 |
| g2k32 | 6832 | 0.805381 | 0.861624 |
| g4k256 | 21792 | 0.693595 | 0.757358 |
| g8k16 | 21792 | 0.625777 | 0.699742 |
| g8k256 | 43552 | 0.623212 | 0.696104 |

同码率分组改善总体误差；缺失区收益较小。g8k256索引位数翻倍而总体误差仅小幅下降。单seed结果，不能宣称多种子稳定优势；token多样性不等于正确恢复细节。

## 框架与复现边界

`lst_grouped_vq` 是冻结训练、量化、评估核心；`lst_grouped_vq8` 是A07配置和调度；`lst_grouped_summary` 是五配置报告与新case生成器。A04/A05和少量旧调度代码是依赖，不代表额外实验纳入本报告。eo_data/eo_denoise及上游许可证原样保留。框架中的Prithvi类是公共依赖，五个模型均未使用Prithvi。

安装适合本机的PyTorch/CUDA，再安装 requirements.txt。此文件是依赖清单，不是跨平台锁定环境。源码、配置和历史manifest按字节保存；配置保留原机器绝对路径，监测工具部分针对Windows。**这是一份可审计历史快照，不是迁移后直接启动的训练包。**

只跑单元检查（不启动训练）：
```sh
python -m pytest tests/test_lst_grouped_vq.py tests/test_lst_grouped_vq8.py tests/test_lst_v2_epochs.py tests/test_lst_residual.py -q
```

未包含原始影像、prepared数据本体、A05 QA形状库或训练权重（单检查点约1.3GB）。成员清单、数据统计证据、输入摘要和检查点SHA256保留。复现预测须另行部署数据、QA库及对应epoch5检查点，并核对哈希。迁移/再训练必须另建run与配置，显式指定HLP v2、新train归一化、split-local QA和准确epoch边界；不得覆盖本快照历史结果或直接执行旧supervise启动队列。

## 原始证据

包括最终20单元metrics、逐样本误差、逐组逐尺度token频数、记录ID、每轮验证汇总、训练日志、门禁、完成审查、恢复事件记录、GPU时长和历史/新case图。体积较大的JSONL和划分CSV使用无损gzip，解压后字节SHA256与原文件相同。每轮验证的逐样本日志、权重和原始数据未打包；其他外部依赖/未附文件由历史manifest描述。源报告保留原始路径，缺少外部资产时不能直接重跑report/gallery。

`EXPORT_MANIFEST.json` 记录所有复制文件的来源与SHA256。可执行 `python verify_export.py` 验证上传副本；不加载模型、不写历史结果。历史registry只导出A06/A07两条记录。
''',encoding='utf-8')
    (out/'verify_export.py').write_text('''from pathlib import Path
import gzip,hashlib,json
root=Path(__file__).resolve().parent
m=json.loads((root/'EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
for e in m['files']:
    b=(root/e['path']).read_bytes()
    assert hashlib.sha256(b).hexdigest()==e['sha256'],e['path']
    if e['encoding']=='gzip':b=gzip.decompress(b)
    assert hashlib.sha256(b).hexdigest()==e['source_sha256'],e['path']
print('PASS:',len(m['files']),'source files; exact source bytes verified')
''',encoding='utf-8')
    for p in out.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    manifest={'scope':'A06/A07 five unique arms; 20 final evaluation units','files':entries,
              'omitted':'weights, raw/prepared imagery, QA banks, epoch validation sample logs, environments and caches',
              'frozen_sources_modified':False,'copied_bytes':sum(e['bytes'] for e in entries)}
    (out/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'files':len(entries),'MB':manifest['copied_bytes']/1e6}))

if __name__=='__main__':main()
