"""Build a source-only GitHub export; never copies data, weights or environments."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Export destination must be empty')
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {}

    def copy(relative):
        src = root / relative
        dst = out / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        source_hashes[relative.as_posix()] = hashlib.sha256(src.read_bytes()).hexdigest()

    for folder in ('eo_data', 'eo_denoise', 'experiments/paper1/lst_restoration'):
        for src in sorted((root / folder).rglob('*')):
            if src.is_file() and '__pycache__' not in src.parts and src.suffix in ('.py', '.md', '.txt', '.json'):
                copy(src.relative_to(root))
    for item in ('AGENTS.md', 'experiments/paper1/training_budget.json',
                 'experiments/data/dataset_catalog.json', 'data_inspection/FINAL_DATASET_SUMMARY.md',
                 'data_inspection/hlp_split_20260914/README.md',
                 'data_inspection/hlp_split_20260914/split_summary.json',
                 'runs/paper1/lst_residual_scale_20260918/residual_scale.json',
                 'runs/paper1/lst_colour_mapping_check_20260927/mapping_checks.json'):
        copy(Path(item))
    # Distribution-only path portability. Scientific settings remain identical.
    cfg = out / 'experiments/paper1/lst_restoration/config.py'
    source = cfg.read_text(encoding='utf-8')
    assert source.count('"root": str(PROJECT)') == 1
    cfg.write_text(source.replace('"root": str(PROJECT)', '"root": "."'), encoding='utf-8')
    (out / '.gitattributes').write_text('# Preserve exact bytes for source/asset SHA256 audits across platforms.\n* -text\n', encoding='utf-8')
    (out / '.gitignore').write_text('''__pycache__/
*.py[cod]
.venv*/
.env
.env.*
!.env.example
.pytest_cache/
/data/
/models/
/processed_data/
/output/
/tmp/
/artifacts/
*.pt
*.pth
*.ckpt
*.safetensors
*.zip
*.eoz
*.npz
*.npy
*.sqlite
*.log
/runs/*
!/runs/paper1/
/runs/paper1/*
!/runs/paper1/lst_residual_scale_20260918/
!/runs/paper1/lst_colour_mapping_check_20260927/
/runs/paper1/lst_residual_scale_20260918/*
!/runs/paper1/lst_residual_scale_20260918/residual_scale.json
/runs/paper1/lst_colour_mapping_check_20260927/*
!/runs/paper1/lst_colour_mapping_check_20260927/mapping_checks.json
''', encoding='utf-8')
    (out / 'requirements.txt').write_text('''# Install a compatible PyTorch/torchvision build for your platform separately.
# Source environment: torch 2.11.0, torchvision 0.26.0; not a portability guarantee.
timm==1.0.29
einops==0.8.2
numpy>=1.24
zstandard>=0.22
scipy>=1.10
PyYAML>=6
diffusers==0.35.2
safetensors==0.8.0
matplotlib>=3.10
Pillow>=11
# EO-VAE and TerraMind require separately bound audited upstream sources/dependencies.
''', encoding='utf-8')
    (out / 'THIRD_PARTY_NOTICES.md').write_text('''# Third-party notices

This repository contains adapted or reference upstream code. Preserve their original notices and licenses:

- FoundationVision/VAR: `eo_denoise/VAR_LICENSE.txt`; VAR reference/codec/quantizer modules.
- Prithvi-EO-2.0: `eo_denoise/PRITHVI_LICENSE.txt`; Prithvi reference/spatial modules.
- SatMAE++: `experiments/paper1/lst_restoration/geo_sources/satmae_LICENSE.txt`; adapted model and position code.
- EO-VAE: `experiments/paper1/lst_restoration/geo_sources/eo_LICENSE.txt`; reviewed source excerpts.
- TerraMind/TerraTorch: `experiments/paper1/lst_restoration/geo_sources/tm_LICENSE.txt`; reviewed source excerpts.

Source revisions and hashes are recorded in `geo_sources/manifest.json` and `model_sources.md`.
No new blanket license is assigned by this export. Model weights and datasets are not distributed.
''', encoding='utf-8')
    plan_path = 'experiments/paper1/lst_tables_v1_20260927_r1'
    result = subprocess.run([sys.executable, '-X', 'utf8', '-m',
        'experiments.paper1.lst_restoration', 'table-plan', '--output', plan_path],
        cwd=out, check=True, capture_output=True, text=True, encoding='utf-8')
    plan_summary = json.loads(result.stdout)
    (out / 'README.md').write_text('''# LST Restoration Benchmark — 实验框架

用离散表示恢复受合成传感器噪声、随机像素缺失和云形遮盖污染的 LST；比较原生数值、残差及颜色表示，并保留自然图像预训练迁移方向。

**状态：实验设计和实现阶段。166份候选配置、6张主表＋4张补表；全部 `execution.enabled=false`。未启动本轮训练，尚无本轮正式模型结果。**

## 先看这些

- [实验顺序与填表协议](experiments/paper1/lst_restoration/EXPERIMENT_ORDER_ZH.md)
- [框架说明](experiments/paper1/lst_restoration/README_ZH.md)
- [166组计划与待补对照](experiments/paper1/lst_tables_v1_20260927_r1/manifest.json)
- [十表索引](TABLES.md)
- [颜色映射公式与CPU合成检查](experiments/paper1/lst_restoration/COLOUR_MAPPING_ZH.md)
- [模型来源与适配范围](experiments/paper1/lst_restoration/model_sources.md)
- [第三方许可证](THIRD_PARTY_NOTICES.md)

## 使用

在仓库根目录运行，标准库即可生成计划，不初始化模型：

```bash
python -m experiments.paper1.lst_restoration --help
python -m experiments.paper1.lst_restoration table-plan --output output/my_plan
python -m experiments.paper1.lst_restoration validate --config experiments/paper1/lst_tables_v1_20260927_r1/configs/vq_R_clean_seed17.json
```

计划生成拒绝覆盖非空目录。上传配置的 `data.root` 使用相对路径 `.`；需在仓库根目录执行。迁移到别的位置或修改配置后，重新绑定预检摘要。

数学颜色映射检查需安装 PyTorch，可单独执行，不读取数据、不启动训练：

```bash
python -m experiments.paper1.lst_restoration.check_colour --output output/colour_check
```

模型运行另外需要平台适配的 PyTorch/torchvision 和 `requirements.txt`。这些依赖来自源码环境记录，未承诺任意机器均已验证。EO-VAE/TerraMind 需单独绑定官方源码和依赖；自然图像模型需要合法取得的本地官方权重与摘要，运行器不自动下载。

## 数据与正式运行

本仓库不包含影像、压缩数据、训练检查点、虚拟环境或大型缓存。数据须单独部署并维持 HLP v2 冻结划分，参见 [数据约定](AGENTS.md) 和 [数据报告](data_inspection/FINAL_DATASET_SUMMARY.md)。统计与划分说明文件不等于数据集本体，读取器需实际归档和完整索引。

正式训练统一5完整epoch、218,975条训练记录、1,094,875次呈现、有效batch64及17,110次更新。完整val/test为11,684/11,757条。主线为明确给定均温的残差条件，其他均值条件单独比较。

先做数据、组件、模型、预算、参数/码率和资源验收，再绑定证据并显式开启单个配置。静态检查不能代替运行验证。本仓库没有自动提交全部166组的后台任务或GitHub Actions训练任务。

尚待接入：无均温绝对温度组、自然KL-VAE同架构随机初始化、自然图像预训练离散模型，以及若干统计汇总接口。不要把当前scratch VQ/FSQ/LFQ/BSQ标为官方预训练模型。

## 导出说明

本仓库是本地项目实验框架的独立源码快照。原有实验与数据保持不变；仅将导出版本的默认数据根改为相对路径并重新生成配置与摘要。`runs/`仅保留两份明确列出的轻量统计/数学检查证据，其历史路径与指纹属于原始记录，不代表本仓库拥有对应数据或已经通过新代码模型预检。

参考文档中的历史实验/数据外链文件不一定随源码分发；当前可用入口以上述相对链接为准。未知结果保持null，未运行不填零。
''', encoding='utf-8')
    registry = json.loads((out / plan_path / 'table_registry.json').read_text(encoding='utf-8'))
    lines = ['# 十张结果表', '', '均为空模板；同一job_id的结果供多表复用。', '', '|表|内容|模板|', '|---|---|---|']
    for key, item in registry.items():
        lines.append(f'|{key}|{item["title"]}|[Markdown]({plan_path}/tables/{key}.md) · [JSON]({plan_path}/tables/{key}.json)|')
    (out / 'TABLES.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for file in out.rglob('*.py'):
        ast.parse(file.read_text(encoding='utf-8-sig'))
    write = { 'status': 'source_export_static_only', 'plan': plan_summary,
        'source_sha256_before_export_path_adjustment': source_hashes,
        'adjustment': 'exported config.py data.root default str(PROJECT) -> .; plan regenerated',
        'datasets_included': False, 'weights_included': False, 'training_started': False}
    (out / 'EXPORT_MANIFEST.json').write_text(json.dumps(write, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(out), **plan_summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
