"""Faithful print export of the existing REPORT_ZH.md, including all figures."""
import hashlib
import html
import json
import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'runs/paper1/lst_v2_5epoch_20260915/REPORT_ZH.md'
OUT = ROOT/'output/pdf/HLP_v2_5epoch_REPORT_ZH.pdf'


def main():
    source = SOURCE.read_text(encoding='utf-8')
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    pdfmetrics.registerFont(TTFont('ReportCN', 'C:/Windows/Fonts/msyh.ttc'))
    pdfmetrics.registerFont(TTFont('ReportCNBold', 'C:/Windows/Fonts/msyhbd.ttc'))
    pdfmetrics.registerFontFamily('ReportCN', normal='ReportCN', bold='ReportCNBold')
    w, h = landscape(A4)
    available = w-64
    body = ParagraphStyle('body', fontName='ReportCN', fontSize=9, leading=13.5, spaceAfter=8, wordWrap='CJK', textColor=colors.HexColor('#223444'))
    h1 = ParagraphStyle('h1', parent=body, fontName='ReportCNBold', fontSize=19, leading=26, spaceAfter=12, keepWithNext=True)
    h2 = ParagraphStyle('h2', parent=body, fontName='ReportCNBold', fontSize=14, leading=20, spaceBefore=5, spaceAfter=9, keepWithNext=True)
    cell = ParagraphStyle('cell', parent=body, fontSize=8.1, leading=10.8, spaceAfter=0, alignment=1)
    left = ParagraphStyle('left', parent=cell, alignment=0)
    head = ParagraphStyle('head', parent=cell, fontName='ReportCNBold', textColor=colors.white)
    caption = ParagraphStyle('caption', parent=body, fontName='ReportCNBold', fontSize=11, leading=15, spaceAfter=7)
    def inline(s):
        return re.sub(r'`([^`]+)`', lambda m: '<font name="ReportCN">'+m.group(1)+'</font>', html.escape(s))
    def p(s, style=body):
        return Paragraph(inline(s), style)
    blocks = re.split(r'\n\s*\n', source.strip())
    story, images, text_blocks = [], [], []
    for block in blocks:
        if block.startswith('!['):
            match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', block.strip())
            assert match, block
            alt, name = match.groups()
            image_path = Path(name)
            assert image_path.is_file()
            story.append(PageBreak())
            story.append(p(alt, caption))
            image = Image(str(image_path))
            max_h = 427 if 'clean_2.png' in name else 489
            scale = min(available/image.imageWidth, max_h/image.imageHeight)
            image.drawWidth = image.imageWidth*scale
            image.drawHeight = image.imageHeight*scale
            image.hAlign = 'CENTER'
            story.extend([image, Spacer(1, 7)])
            images.append(dict(path=name, sha256=hashlib.sha256(image_path.read_bytes()).hexdigest()))
        elif block.startswith('|'):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            rows = [[v.strip() for v in line.strip('|').split('|')] for line in lines]
            assert re.fullmatch(r'[|:\-\s]+', lines[1])
            rows.pop(1)
            n = len(rows[0])
            if n == 10:
                widths = [33, 141, 34, 70, 70, 70, 70, 83, 83, available-654]
            elif n == 7:
                widths = [45, 190, 60, 110, 110, 140, available-655]
            elif n == 4:
                widths = [315, 165, 185, available-665]
            elif n == 2:
                widths = [260, available-260]
            else:
                raise AssertionError(n)
            assert abs(sum(widths)-available)<.01
            formatted = [[p(value, head if i==0 else left if (j==1 and n in (7,10)) or (j==0 and n in (2,4)) else cell) for j,value in enumerate(row)] for i,row in enumerate(rows)]
            table = Table(formatted, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#19394F')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white,colors.HexColor('#F0F4F7')]),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
                ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#CCD7DF')),
            ]))
            story.extend([table,Spacer(1,9)])
            text_blocks.extend(value for row in rows for value in row)
        elif block.startswith('# '):
            title = block[2:].strip()
            story.append(p(title, h1))
            text_blocks.append(title)
        elif block.startswith('## '):
            heading = block[3:].strip()
            if heading in ['GPU 时间','码字使用分布摘要']:
                story.append(PageBreak())
            story.append(p(heading, h2))
            text_blocks.append(heading)
        else:
            story.append(p(block))
            text_blocks.append(block.replace('`',''))
    def footer(c, doc):
        c.saveState()
        c.setStrokeColor(colors.HexColor('#D1DAE2'))
        c.line(32,24,w-32,24)
        c.setFillColor(colors.HexColor('#607586'))
        c.setFont('ReportCN',7)
        c.drawString(32,13,'REPORT_ZH.md | HLP v2 五轮 LST 实验')
        c.drawRightString(w-32,13,f'{doc.page}')
        c.restoreState()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=(w,h), leftMargin=32,rightMargin=32,topMargin=26,bottomMargin=34,
                            title='HLP v2：四组 LST 模型各训练5个完整 epoch', author='HLP 实验项目')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    pdf = PdfReader(str(OUT))
    extracted = '\n'.join(page.extract_text() for page in pdf.pages)
    def normalize(s):
        return re.sub(r'\s+','',s)
    full = normalize(extracted)
    for text in text_blocks:
        assert normalize(text) in full, 'Missing text: '+text[:80]
    image_count = sum(len(page.images) for page in pdf.pages)
    assert image_count == len(images) == 4, image_count
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    check = dict(source=str(SOURCE), source_sha256=before, pdf=str(OUT), pages=len(pdf.pages), images=images,
                 all_text_blocks_verified=len(text_blocks), embedded_images=image_count, source_unchanged=True,
                 pdf_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest())
    OUT.with_suffix('.verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(pdf=str(OUT),pages=len(pdf.pages),text_blocks=len(text_blocks),images=image_count)))


if __name__=='__main__':
    main()
