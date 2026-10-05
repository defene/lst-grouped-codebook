"""One-page presentation handout; values read from frozen experiment outputs."""
import hashlib
import json
import math
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import ParagraphStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'output/pdf/LST_VQVAE_Prithvi_Comparison_ZH.pdf'
A05 = ROOT/'runs/paper1/lst_v2_5epoch_20260915'
A02 = ROOT/'runs/paper1/var_vs_prithvi_ae_full'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont('Yahei', 'C:/Windows/Fonts/msyh.ttc'))
    pdfmetrics.registerFont(TTFont('YaheiBold', 'C:/Windows/Fonts/msyhbd.ttc'))
    pdfmetrics.registerFontFamily('Yahei', normal='Yahei', bold='YaheiBold')
    source_files = [A05/'results.json']
    results = read(source_files[0])['rows']
    data = []
    arms = ['absolute', 'absolute_spatial4', 'residual', 'residual_spatial4']
    labels = ['VQ-VAE 原始预测', 'VQ-VAE 原始＋空间4倍', 'VQ-VAE 残差预测', 'VQ-VAE 残差＋空间4倍']
    for arm, label in zip(arms, labels):
        clean = next(r for r in results if (r['model'], r['split'], r['condition']) == (arm, 'test', 'clean'))
        noisy = next(r for r in results if (r['model'], r['split'], r['condition']) == (arm, 'test', 'noisy10'))
        data.append([label, '缺失', 'v2 / 5轮', clean['regions']['lst/all']['rmse'],
                     noisy['regions']['lst/all']['rmse'], noisy['regions']['lst/corrupt']['rmse'],
                     noisy['regions']['lst/retained']['rmse'], noisy['mean_rmse'], noisy['spatial_rmse'],
                     noisy['regions']['lst/all']['gradient_rmse']])
    for train in ['clean', 'noisy']:
        path = A02/f'full_lst_prithvi_{train}_seed17'
        clean_path, noisy_path, sample_path = path/'eval_clean/metrics.json', path/'eval_noisy10/metrics.json', path/'eval_noisy10/samples.jsonl'
        source_files.extend([clean_path, noisy_path, sample_path])
        clean, noisy = read(clean_path), read(noisy_path)
        samples = [json.loads(line) for line in sample_path.read_text(encoding='utf-8').splitlines()]
        samples = [r for r in samples if r['region'] == 'all']
        assert len(samples) == 11757*2
        mse = sum(r['mse_physical'] for r in samples)/len(samples)
        mean = sum(r['bias_physical']**2 for r in samples)/len(samples)
        assert math.isclose(mse, noisy['regions']['lst/all']['rmse']**2, rel_tol=1e-8)
        data.append(['Prithvi 连续 AE', '干净' if train == 'clean' else '缺失', 'v1 / 约3轮',
                     clean['regions']['lst/all']['rmse'], noisy['regions']['lst/all']['rmse'],
                     noisy['regions']['lst/corrupt']['rmse'], noisy['regions']['lst/retained']['rmse'],
                     mean**.5, max(mse-mean, 0)**.5, noisy['regions']['lst/all']['gradient_rmse']])
    width, height = landscape(A4)
    margin = 34
    content_width = width-2*margin
    c = canvas.Canvas(str(OUT), pagesize=(width, height))
    c.setTitle('LST 重建实验完整对比：VQ-VAE 与 Prithvi')
    c.setAuthor('HLP 实验项目')
    navy = colors.HexColor('#17334A')
    ink = colors.HexColor('#243747')
    muted = colors.HexColor('#596978')
    c.setFillColor(navy)
    c.rect(0, height-9, width, 9, fill=1, stroke=0)
    c.setFont('YaheiBold', 21)
    c.drawString(margin, height-48, 'LST 重建实验完整对比')
    c.setFont('Yahei', 10)
    c.setFillColor(muted)
    c.drawString(margin, height-70, 'HLS-LST-Pairs（HLP）｜本轮4组 VQ-VAE ＋ 历史2组 Prithvi｜汇报版 · 2026-09-18')
    normal = ParagraphStyle('body', fontName='Yahei', fontSize=9.4, leading=15, textColor=ink, wordWrap='CJK')
    small = ParagraphStyle('small', parent=normal, fontSize=8.4, leading=13, textColor=muted)
    def paragraph(text, top, style=normal):
        p = Paragraph(text, style)
        _, h = p.wrap(content_width, height)
        p.drawOn(c, margin, top-h)
        return top-h
    y = paragraph('<b>评估范围：</b>同一批11,757条记录，每条2个实现；所有模型均从零训练。误差单位为°C，越低越好。', height-92)
    y = paragraph('<b>阅读提示：</b>本轮4组可作等预算对照；与历史Prithvi的比较未完全对齐，只能作为参考。', y-4)
    header_style = ParagraphStyle('header', parent=small, fontSize=8.6, leading=12, textColor=colors.white, alignment=1)
    cell_style = ParagraphStyle('cell', parent=small, fontSize=9, leading=13, textColor=ink, alignment=1)
    label_style = ParagraphStyle('label', parent=cell_style, alignment=0)
    headers = ['模型', '训练<br/>输入', '数据版本<br/>训练预算', '干净输入<br/>总体RMSE', '缺失输入<br/>总体RMSE', '缺失区<br/>RMSE¹', '保留区<br/>RMSE¹', '均值<br/>RMSE¹', '空间<br/>RMSE¹', '梯度<br/>RMSE¹']
    table_data = [[Paragraph(x, header_style) for x in headers]]
    minimum = [min(r[j] for r in data) for j in range(3, 10)]
    for row in data:
        cells = []
        for j, value in enumerate(row):
            text = str(value) if j < 3 else f'{value:.4f}'
            if j >= 3 and value == minimum[j-3]:
                text = '<b>'+text+'</b>'
            cells.append(Paragraph(text, label_style if j == 0 else cell_style))
        table_data.append(cells)
    widths = [151, 45, 65]+[(content_width-261)/7]*7
    table = Table(table_data, colWidths=widths, rowHeights=[38]+[34]*6)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), navy),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F5F7')]),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#E6F0F5')),
        ('BACKGROUND', (0, 6), (-1, 6), colors.HexColor('#E6F0F5')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('LINEBELOW', (0, 0), (-1, 0), .6, navy),
        ('LINEABOVE', (0, 5), (-1, 5), 1, colors.HexColor('#90A7B6')),
        ('LINEBELOW', (0, -1), (-1, -1), .6, colors.HexColor('#B8C6D0')),
    ]))
    _, th = table.wrap(content_width, height)
    table_top = y-15
    table.drawOn(c, margin, table_top-th)
    y = table_top-th-10
    y = paragraph('¹ 缺失区、保留区及均值/空间/梯度指标均对应约10%缺失输入。粗体仅标出表内最低数值，不表示严格公平排名。', y, small)
    y = paragraph('<b>主要观察：</b>本轮残差＋空间4倍在4组VQ-VAE中总体误差最低；Prithvi缺失训练版的总体和保留区误差更低，而新版VQ-VAE的缺失区、梯度误差数值更低。细节过度平滑仍未解决。', y-10)
    y = paragraph('<b>比较限制：</b>VQ-VAE：v2训练218,975条，5轮/17,110次更新；Prithvi：旧训练230,659条，约3轮/10,813次更新。归一化、噪声库和遮挡实现不同，缺失区优势需同输入补测确认。两者参数量接近，但Prithvi无VQ、保留连续特征，表示容量未对齐。', y-7, small)
    y = paragraph('<b>指标口径：</b>先平均每条记录两个实现的有效像素MSE，再对记录等权平均后开方；总体MSE＝均值MSE＋空间MSE。“空间4倍”是损失权重，不是分辨率提升。旧报告称该集合为val，现为test；保留其历史监测事实。', y-6, small)
    assert y > 39, f'Footer collision: {y}'
    c.setStrokeColor(colors.HexColor('#D7DFE5'))
    c.line(margin, 31, width-margin, 31)
    c.setFont('Yahei', 7.3)
    c.setFillColor(muted)
    c.drawString(margin, 18, '来源：A05 results.json；A02 最终全量评估 metrics.json / samples.jsonl。均为seed17，未包含重复训练的随机性。')
    c.drawRightString(width-margin, 18, '1 / 1')
    c.save()
    reader = PdfReader(str(OUT))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    for row in data:
        for value in row[3:]:
            assert f'{value:.4f}' in text
    manifest = dict(pdf=str(OUT), pages=1, rows=data, source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                    pdf_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(), numeric_text_check='passed')
    (OUT.with_suffix('.sources.json')).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(pdf=str(OUT), pages=1, bottom_y=y)))


if __name__ == '__main__':
    main()
