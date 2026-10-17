"""Assemble reviewed evidence and a Chinese report without rerunning models."""
import json
from datetime import datetime, timezone
from pathlib import Path
from experiments.openimages_var200.prepare import OUT, save, sha


def main():
    dest = OUT/'integrated_report'
    dest.mkdir(exist_ok=True)
    read = lambda p: json.loads((OUT/p).read_text(encoding='utf-8'))
    s, b = read('summary.json'), read('bicubic_8x8/summary.json')
    cases = read('bicubic_8x8/var_higher_rmse_all3.json')
    paired = [json.loads(line) for line in (OUT/'bicubic_8x8/per_image.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(paired) == 200 and sum(r['var_rmse_255'] < r['bicubic_rmse_255'] for r in paired) == 197
    assert {r['image_id'] for r in paired if r['var_rmse_255'] > r['bicubic_rmse_255']} == {r['image_id'] for r in cases}
    assert abs((1-b['var_rmse_255']/b['bicubic_rmse_255'])*100-b['relative_rmse_reduction_percent']) < 1e-10
    sources = ['summary.json','per_image.jsonl','bicubic_8x8/summary.json','bicubic_8x8/per_image.jsonl',
               'bicubic_8x8/var_higher_rmse_all3.json','checkpoint.json','upstream_manifest.json',
               'selection.json','verification.json','bicubic_8x8/verification.json','bicubic_8x8/residuals_all3/metadata.json']
    provenance = {p: sha(OUT/p) for p in sources}
    save(dest/'source_hashes.json', provenance)
    sections = []
    def add(id, title, body, queries=None, image=None, caption=None):
        sections.append(dict(id=id, title=title, body='## '+title+'\n\n'+body.strip(), queries=queries or [], image=image, caption=caption))
    add('summary', '核心发现', '''
- **整体像素误差明显降低。** 同一批200张图，8×8图像经bicubic恢复后的RMSE为36.64，官方VAR VQ-VAE为21.77，降低40.59%；197/200张的逐图RMSE更低。
- **视觉质量与像素误差出现局部矛盾。** 剩余3张的VQ-VAE重建在观察中更清晰，但RMSE高出1.76%–7.66%。这些案例说明，单独用RMSE排序不足以评价自然图像重建的视觉质量；并不说明RMSE失去数值误差的测量作用。
- **本批码本没有明显使用坍缩。** 全部尺度共使用4096/4096个码字，困惑度3958.31；最细网格每图不同码字数中位数为245/256。该结果不能确定遥感模型的最优码本大小。
- **评价方案需要同时覆盖数值与感知。** 自然图像建议补测LPIPS、DISTS；LST仍应保留摄氏度RMSE，并分开检查图像均值与空间变化。LPIPS、DISTS本次尚未计算。
''', ['overall','codebook','exceptions'])
    add('protocol', '实验使用同一批200张、256×256 RGB图像', '''
本报告整合本轮OpenImages重建、8×8插值对照、三张反例及残差可视化。它不是历史HLP/HLC全部训练实验的汇总，也没有新增训练结果。

| 项目 | 本轮设置 |
|---|---|
| 数据 | OpenImages validation的41,620张名单中固定抽取200张 |
| 抽样 | 按SHA256("20260915:" + ImageID)排序取前200；名单先于推理固定，无按效果筛图 |
| 下载版本 | CVDF最长边不超过1024像素的JPEG；200/200下载和解码校验通过 |
| 输入处理 | RGB，短边Lanczos缩放到288，再中心裁剪256×256；模型输入映射至[-1,1] |
| 权重 | 官方vae_ch160v4096z32.pth，SHA256校验通过，全部参数严格匹配 |
| 模型 | 108,948,355参数；4096个32维码字；基础通道160；FP32，无AMP、关闭TF32 |
| 操作 | 只进行编码→离散索引→解码；没有加噪、训练或自回归生成 |
| 插值基线 | 256×256按32×32区域平均得到8×8，再bicubic恢复256×256；align_corners=False |
| 误差范围 | 两种输出裁剪到[0,1]，保存PNG前以浮点输出计算；报告RGB 0–255单位 |

每图680个索引来自1、2、3、4、5、6、8、10、13、16各尺度网格的格点总和。索引选择码本向量，并由解码器还原图像；它不是一个原始RGB像素。**本次使用的是预训练官方4096码字模型，不是我们从零训练的1024码字LST模型。**

官方tokenizer的精确训练成员未经核实；不能宣称这200张完全未被它见过，也不能根据OpenImages全集规模断言该权重使用了约900万张训练图。
''', ['overall'])
    add('overall', 'VQ-VAE相对8×8插值的整体RMSE降低40.59%', f'''
| 指标 | 8×8 → bicubic | 官方VAR VQ-VAE |
|---|---:|---:|
| 整体RGB RMSE（0–255） | {b['bicubic_rmse_255']:.4f} | {s['rmse_255']:.4f} |
| 整体RGB RMSE（0–1） | {b['bicubic_rmse_255']/255:.5f} | {s['rmse_255']/255:.5f} |
| 平均逐图PSNR（dB） | 未汇总 | {s['mean_image_psnr_db']:.4f} |
| 整体MSE对应PSNR（dB） | 未汇总 | {s['pooled_psnr_db']:.4f} |
| 平均SSIM | 未计算 | {s['mean_ssim']:.4f} |
| 平均MAE（0–255） | 未汇总 | {s['mean_mae_255']:.4f} |

VQ-VAE的整体RMSE绝对减少{b['absolute_rmse_reduction_255']:.4f}，相对减少{b['relative_rmse_reduction_percent']:.2f}%。对应MSE减少{b['relative_mse_reduction_percent']:.2f}%，这两个百分比不可混用。逐图胜出比例为197/200，即98.5%。

整体RMSE先对全部200张、所有像素和RGB通道的平方误差求平均，再开平方；它不是逐图RMSE的平均值。平均逐图PSNR与整体MSE对应PSNR也不是同一聚合方式。

**这是一项粗分辨率插值基线，不是等码率压缩评估。** 8×8 RGB含192个通道数值，而VAR含680个离散索引并依赖预训练解码器；两者的编码精度、先验和可用信息不同。它也不是把VAR只截断到8×8 token网格后的结果。
''', ['overall'], 'bicubic_8x8/comparison_first8_1.png', '固定抽样顺序的前4张。每行左：原图；中：8×8插值；右：VQ-VAE。')
    table = '| 抽样序号 | ImageID | bicubic RMSE | VQ-VAE RMSE | RMSE增加 |\n|---|---|---:|---:|---:|\n'
    for c in cases:
        table += f"| {c['sample_number']} | {c['image_id']} | {c['bicubic_rmse_255']:.4f} | {c['var_rmse_255']:.4f} | {-c['var_rmse_reduction_percent']:.2f}% |\n"
    add('exceptions', '三张反例：VQ-VAE更清晰，逐像素误差却更大', table+'''

第103张具有密集亮点纹理；第127张为草地与树干；第160张为密集细长枝叶。上表是全部3张反例，不是挑选出的部分反例。

**目视观察：** VQ-VAE保留了更多主体、边缘和纹理，bicubic呈现大范围模糊。这里的“更清晰”来自样例观察，不是已完成的盲测或感知指标结论；清晰也不等于原始细节逐像素正确。
''', ['exceptions'], 'bicubic_8x8/var_higher_rmse_all3.png', '全部3张反例，左：原图；中：bicubic；右：VQ-VAE。')
    add('residuals', '残差图揭示了两种不同的误差形态', '''
残差定义为“重建−原图”。中间两列显示0.2126ΔR+0.7152ΔG+0.0722ΔB的亮度加权残差，蓝色表示偏暗、红色表示偏亮；这是对RGB差值的加权显示，不是温度残差，也不是图内去均值残差。

右侧两列显示每像素的RGB误差幅度：√[(ΔR²+ΔG²+ΔB²)/3]，越亮误差越大。对该幅度图平方取均值再开平方，可恢复该图的RGB RMSE，已与逐图指标校验一致。

所有三张、两种方法共用色标：有符号残差[-170,170]，RGB幅度[0,180]，没有通过裁剪极值增强对比。

**观察：** bicubic残差保留了明显草叶和枝条轮廓，体现结构被平滑移除；VQ-VAE残差更细碎。VQ-VAE的残差并没有在这些案例中获得更低的累计平方误差。

**解释边界：** 纹理位置或明暗不完全匹配可解释这种现象，但本次没有估计位移场、做对齐干预或分解具体成因，因此不能断言误差全部来自几像素的平移。
''', ['exceptions'], 'bicubic_8x8/residuals_all3/residuals_all3.png', '同尺度残差对照。右侧两列与RMSE直接对应；亮度加权残差可能发生通道抵消。')
    add('interpretation', 'RMSE衡量数值准确，不能单独代替视觉质量', '''
当重建纹理的亮暗位置与原图不一致时，清晰重建可能产生较大的平方误差。模糊图输出局部平均值，虽然丢失结构，却可能减小平均平方偏差。

| 教学示例（不是实验数据） | 像素序列 | 相对原图RMSE |
|---|---|---:|
| 原图 | [0, 100, 0, 100] | — |
| 清晰但完全错位 | [100, 0, 100, 0] | 100 |
| 完全模糊 | [50, 50, 50, 50] | 50 |

这说明“细节存在”与“细节在正确的位置、具有正确的值”是不同要求。对于同一图像、同一动态范围，PSNR=20log10(MAX/RMSE)，改用PSNR不会改变与RMSE对应的优劣排序。跨图片采用不同聚合方式可能改变整体排序，不代表新增了感知信息。

因此，应将当前结论表述为：**RMSE单独评价自然图像感知质量存在局限；它对数值保真仍然有效。** 不能为了让视觉上更清晰的结果获胜而事后只保留有利指标。
''')
    add('codebook', '码本使用充分，但尚不能确定遥感最优码本大小', f'''
| 统计范围 | 已测结果 |
|---|---:|
| 全部尺度总索引数 | 136,000 |
| 全部尺度使用码字数 | {s['tokens']['used_codes']}/4096 |
| 全部尺度困惑度 | {s['tokens']['perplexity']:.2f} |
| 最细16×16尺度跨200张使用码字数 | {s['tokens_by_scale'][-1]['used_codes']}/4096 |
| 最细尺度困惑度 | {s['tokens_by_scale'][-1]['perplexity']:.2f} |
| 最细网格每张不同码字数中位数 | 245/256 |
| 最细网格每张最常用码字占比中位数 | 0.78125% |

困惑度定义为2^H，H=−Σp(k)log₂p(k)。本批观测没有显示明显的码字使用坍缩；全尺度和单尺度统计应分开看，不能将全尺度100%使用率写成每张图都使用全部码字。

高使用率和高困惑度不证明码字的语义、空间结构或任务相关性都理想。它们也不建立“训练图片数量→码本大小”的直接比例关系。自然RGB、预训练过程、网络和量化配置均与我们的遥感模型不同，不能根据这200张推断HLP应采用4096码字，或认定1024已是最优。
''', ['codebook'])
    add('metrics', '建议补充感知评价，并保留像素指标', '''
| 指标 | 主要衡量内容 | 趋势 | 本轮状态 |
|---|---|---|---|
| RMSE / PSNR | 逐像素数值误差 | RMSE↓，PSNR↑ | RMSE已对比；VAR PSNR已汇总 |
| SSIM / MS-SSIM | 局部结构、亮度、对比度 | ↑ | 仅VAR SSIM已测；其余未测 |
| LPIPS | 深度特征的感知差异 | ↓ | 未测 |
| DISTS | 结构与纹理相似性，对纹理重采样更宽容 | ↓ | 未测 |
| rFID | 原图与重建图的集合特征分布 | ↓ | 未测；不能评价单张对应关系 |

SSIM提出结构相似的评价方式；LPIPS研究深度特征与人类感知判断的联系；DISTS明确考虑了纹理重采样的容忍性，因此特别适合检验这次草地、枝叶案例中的矛盾。相关来源：[SSIM](https://ieeexplore.ieee.org/document/1284395)、[LPIPS](https://arxiv.org/abs/1801.03924)、[DISTS](https://arxiv.org/abs/2004.07728)。[FID](https://arxiv.org/abs/1706.08500)比较集合分布，不能替代逐图对应评价。

**建议的下一项验证：** 对同一200张、同一裁剪和两种方法补测LPIPS、DISTS及SSIM，固定模型版本与输入规范，报告全体结果和三张反例的配对结果。结果未知，不能提前声称感知指标一定支持VQ-VAE。200张的小样本不宜将rFID作为主要结论；本报告没有复现或引用论文rFID作为本地实测。
''')
    add('remote', '对HLP温度实验的启示：分开评价均值与空间变化', '''
以下为评价方案建议，尚未通过本轮自然图像测试在LST/HLS上验证，也未改变现有训练配置。

对LST，保留摄氏度RMSE和平均偏差，并分别报告图像均值误差、图内去均值残差RMSE、梯度误差及局部对比图。在相同有效mask上，令r=T−mean(T)、r̂=T̂−mean(T̂)，则每张图有：

**MSE(T̂,T) = [mean(T̂)−mean(T)]² + MSE(r̂,r)。**

该分解可区分整图温度水平与图内变化，适合检验“模型只学到图片之间差异”的假设。汇总时必须按有效像素数加权，且不能把RMSE直接相加。去均值残差和梯度误差仍对错位敏感，不能把它们当作天然解决感知问题的指标。

LST使用有效mask，摄氏度读取为 `ds.denormalize(sample["image"]) / 100.0`；归一化image不能直接除以100。fine LST是卫星反演参考产品。自然RGB的0–255 RMSE不能与摄氏度RMSE直接比较。

HLS应保留逐波段数值误差和空间结构检查。LPIPS/DISTS基于自然RGB的特征，不能未经验证直接作为多光谱HLS或LST主指标；伪彩色图上的感知得分也不能替代全部波段的物理量评价。

后续正式HLP实验遵守既定v2划分和5个完整epoch预算，使用新train归一化、val选模、test最终评估；不使用test选择码本大小。Prithvi版本保持连续瓶颈的既定方向。本报告只给出评价建议，没有启动这些训练。
''')
    checkpoint = read('checkpoint.json')
    add('reproducibility', '验证记录、复现入口与限制', f'''
已核对200张输入、重建数组、逐图指标和PNG的一致性，重新计算RMSE与已存结果一致；8×8降采样与32×32分块均值一致；三张残差图的幅度可恢复其逐图RMSE。已观察固定前四张细节、第25页及全部三张反例。没有声称逐张完成200张的人工质量评级。

官方VAR本轮推理循环为{s['inference_seconds']:.2f}秒，含首批往返一致性检查；峰值张量显存{s['peak_allocated_gib']:.3f} GiB，峰值保留显存{s['peak_reserved_gib']:.3f} GiB。这是单次推理记录，不是训练GPU hours，也不是重复测量的独占吞吐量。

权重SHA256：`{checkpoint['sha256']}`。上游代码提交：`{s['upstream_revision']}`。样本、权重和脚本来源均已保留，本报告另存来源文件哈希。

限制包括：只有200张固定样本；官方训练成员未经核实；输入经过中心裁剪；基线不等码率；感知指标尚未计算；本轮没有证明具体纹理误差的因果来源，也不能确定遥感码本大小。

主要外部来源：[VAR官方仓库](https://github.com/FoundationVision/VAR)、[官方权重](https://huggingface.co/FoundationVision/var/blob/main/vae_ch160v4096z32.pth)、[OpenImages下载说明](https://storage.googleapis.com/openimages/web/download_v7.html)、[CVDF下载版本](https://github.com/cvdfoundation/open-images-dataset)。
''', ['overall'])
    save(dest/'sections.json', sections)
    lines = ['# VAR重建更清晰，但RMSE不能单独代表视觉质量', '', '实验批次：OpenImages 200张（20260915）；本文整合现有结果，不增加训练。', '']
    for section in sections:
        lines.extend([section['body'], ''])
        if section['image']:
            lines.extend([f"![{section['caption']}](../{section['image']})", '', section['caption'], ''])
    lines.extend(['## 本地证据索引', '', '[全部200张VAR对照](../REPORT_ZH.md) · [插值基线报告](../bicubic_8x8/REPORT_ZH.md) · [逐图配对指标](../bicubic_8x8/per_image.jsonl) · [来源哈希](source_hashes.json)', '',
                  '项目根目录下的脚本为 `experiments/openimages_var200/` 中的 prepare.py、reconstruct.py、verify.py、bicubic_baseline.py、show_bicubic_exceptions.py 和 plot_residuals.py；报告汇编脚本为 prepare_findings_report.py。', ''])
    (dest/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    def query(rows, files, definition, components):
        return {'rows': rows, 'source': {'label': '本地已验证实验输出', 'files': files,
                 'filters': ['固定OpenImages验证集200张；256×256 RGB；无训练、无加噪'],
                 'metricDefinitions': [{'label': definition.split('：')[0], 'definition': definition, 'componentIds': components}],
                 'evidenceFlow': [{'title': '读取现有实验输出', 'detail': '从固定实验目录读取JSON/JSONL；报告生成不重跑模型。来源文件SHA256保存在source_hashes.json。'}]},
                'methods': [{'language': 'text', 'code': definition}]}
    snapshot = {'surface': 'report', 'title': 'VAR重建更清晰，但RMSE不能单独代表视觉质量',
                'generatedAt': datetime.now(timezone.utc).isoformat(), 'status': 'reviewed', 'buildStatus': 'creating',
                'report': {'asOf': '2026-09-15'}, 'filters': [], 'queries': {
        'overall': query([{**b, 'mean_image_psnr_db':s['mean_image_psnr_db'], 'mean_ssim':s['mean_ssim']}],
                         ['summary.json','bicubic_8x8/summary.json'],
                         '整体RMSE：先对全部200张RGB像素平方误差求平均，再开平方；单位0–255。相对改善=(1−VAR_RMSE/bicubic_RMSE)×100%。',
                         ['summary','protocol','overall','overall-figure','reproducibility']),
        'paired': query(paired, ['bicubic_8x8/per_image.jsonl'], '逐图RMSE：每张256×256 RGB图像分别计算；全部200张，包含3张反例。', ['paired-table']),
        'exceptions': query(cases, ['bicubic_8x8/var_higher_rmse_all3.json'], '反例：逐图VAR_RMSE>bicubic_RMSE的全部3张；正向增加百分比为−var_rmse_reduction_percent。', ['summary','exceptions','exceptions-figure','residuals','residuals-figure']),
        'codebook': query([{'scope':'all', **s['tokens']}, *s['tokens_by_scale']], ['summary.json'], '困惑度：2^(−Σp(k)log₂p(k))；全尺度与各尺度分别计数，不代表语义质量。', ['summary','codebook'])}}
    save(dest/'reviewed_snapshot.json', snapshot)
    print(str(dest))


if __name__ == '__main__':
    main()
