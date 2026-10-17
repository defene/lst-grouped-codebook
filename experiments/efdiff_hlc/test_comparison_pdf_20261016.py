"""Create the audited full-test report from saved local artifacts."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/hlc_unet_mmdit_test_20261015'
FIG=BASE/'results'
data=json.loads((FIG/'verification.json').read_text())
audit=json.loads((FIG/'extended_audit.json').read_text())
protocol=json.loads((BASE/'test/protocol.json').read_text())
stats=json.loads((BASE/'latest_status.json').read_text())['job_stats']
assert data['complete_coverage'] and audit['record_month_pairs_unique']
assert all(stats[k]==0 for k in ['runners','running','backlog'])
OUT=ROOT/'output/pdf/hlc_unet_mmdit_fulltest_20261016.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
W,H=595.276,841.89
c=canvas.Canvas(str(OUT),pagesize=(W,H));c.setTitle('HLC Full Test - UNet and MMDiT');c.setAuthor('HLC experiment analysis')
body=ParagraphStyle('body',fontName='CN',fontSize=10,leading=16,textColor=HexColor('#233047'))
small=ParagraphStyle('small',parent=body,fontSize=8,leading=12)
page=0
def para(text,y,size=10):
    p=Paragraph(text,body if size==10 else small);_,h=p.wrap(W-76,700)
    assert y-h>42,('Text overflow',page,y,h)
    p.drawOn(c,38,y-h);return y-h-10
def start(title,sub='HLS-LST-CrossSensor | Full test | 2026-10-16'):
    global page
    page+=1;c.setFillColor(HexColor('#102942'));c.rect(0,H-9,W,9,fill=1,stroke=0)
    c.setFont('CN',20);c.drawString(38,H-54,title)
    c.setFont('CN',8);c.setFillColor(HexColor('#60738a'));c.drawString(38,H-73,sub)
    return H-94
def end():
    c.setFillColor(HexColor('#60738a'));c.setFont('CN',8)
    c.drawString(38,24,'HLC | Synthetic Wald32 degradation | Fixed checkpoints')
    c.drawRightString(W-38,24,str(page));c.showPage()
def table(rows,y,widths):
    t=Table([[Paragraph(str(v),small) for v in row] for row in rows],colWidths=widths)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e4edf5')),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#f4f7fa')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    _,h=t.wrap(W-76,700);assert y-h>42;t.drawOn(c,38,y-h);return y-h-15
def picture(name,y,maxheight):
    im=ImageReader(str(FIG/name));iw,ih=im.getSize();s=min((W-76)/iw,maxheight/ih)
    c.drawImage(im,(W-iw*s)/2,y-ih*s,width=iw*s,height=ih*s)
    return y-ih*s-12
m=data['metrics']
keys=['unet_x0','mmdit_5','unet_epsilon','mmdit_50','bicubic']
names={'unet_x0':'UNet x0','unet_epsilon':'UNet epsilon','mmdit_5':'MMDiT v-pred','mmdit_50':'MMDiT v-pred','bicubic':'Bicubic'}
gap=100*(m['mmdit_5']['rmse_c']/m['unet_x0']['rmse_c']-1)
gain=100*(1-m['mmdit_5']['rmse_c']/m['unet_epsilon']['rmse_c'])
gain50=100*(1-m['mmdit_5']['rmse_c']/m['mmdit_50']['rmse_c'])
y=start('完整测试结果：UNet 与 MMDiT')
y=para(f'141,524 条完整测试已完成。UNet x0 的总体误差最低；MMDiT 5 NFE 优于 UNet epsilon，但 RMSE 仍比 UNet x0 高 {gap:.2f}%。MMDiT 50 NFE 的误差高于 5 NFE。',y)
y=table([['模型','训练轮次','NFE','RMSE (°C)','MAE (°C)']]+[[names[k],'-' if k=='bicubic' else ('2' if k.startswith('mmdit') else '1'),m[k]['nfe'],f"{m[k]['rmse_c']:.4f}",f"{m[k]['mae_c']:.4f}"] for k in keys],y,[175,75,59,105,105])
y=para('所有行使用相同的 141,524 条 test 记录和 7,072,794,483 个有效像素。RMSE = sqrt(全体有效像素平方误差之和 / 有效像素数)；MAE 使用同一分母。不是逐图 RMSE 的平均值。',y)
y=para(f'MMDiT 5 NFE 相对 UNet epsilon 的 RMSE 低 {gain:.2f}%，相对自身 50 NFE 低 {gain50:.2f}%。这里的 NFE 表示模型采样评估次数，不代表实际耗时比例。两个 NFE 设置在测试前已固定；没有根据 test 重新选模或调参。',y)
y=para('这次结果支持“当前 UNet x0 配置在这套合成退化测试上更准确”，尚不支持“UNet 架构一定优于 MMDiT”。UNet 训练 1 轮，MMDiT 训练 2 轮；训练裁剪、损失、条件输入及预处理均有差异。',y)
y=para('本主表没有混入旧 512 条验证分数、完整 validation 分数或旧部分 test 分数。温度参考为卫星反演 fine LST；粗温度由参考图合成退化，不能视为真实粗温度观测恢复的最终结论。',y)
end()
y=start('逐记录比较与固定 case')
y=table([['比较：MMDiT 5 NFE','RMSE 更低的记录','占全部 test']]+[[label,f'{audit["per_record_wins"][k]:,}',f'{100*audit["per_record_wins"][k]/141524:.2f}%'] for k,label in [('mmdit_5_vs_unet_x0','对 UNet x0 / 15 NFE'),('mmdit_5_vs_unet_epsilon','对 UNet epsilon / 50 NFE'),('mmdit_5_vs_mmdit_50','对 MMDiT / 50 NFE')]],y,[259,140,120])
y=para('逐记录胜率与全局像素加权指标回答不同问题。记录之间可能存在空间或月份相关性，这里不把记录当独立样本做显著性宣称。',y,8)
y=table([['Case / record / 月','UNet x0','UNet ε','MMDiT 5','MMDiT 50']]+[[f'{r["case"]} / {r["record_id"]} / {r["month"]}']+[f'{r[k]["rmse_c"]:.3f}' for k in ['unet_x0','unet_epsilon','mmdit_5','mmdit_50']] for r in audit['fixed_cases']],y,[179,85,85,85,85])
y=para('上表为单图有效像素 RMSE（°C）。测试前预设 16 个等间隔索引保存预测，再固定取其中 8 个等间隔位置展示；未按误差挑选样本。这组 test case 与此前看的 validation case 不同。',y)
y=para('后续每页包含对应 HLS RGB、实际 7×7 合成粗温度、fine LST 参考、四种预测设置及 MMDiT 残差。每个 case 内温度图共享色标（参考有效像素第 1 至 99 百分位）；各 case 的温度范围可以不同。残差为预测减参考，统一 ±5°C，超范围颜色饱和。',y)
y=para('HLS RGB 使用固定显示范围及 gamma，仅用于可视化。HLS 获取日期可能不同于目标温度日期，因此不应期待每一条光谱边缘都对应同日温度边缘。无效 HLS 区域标灰；无效 LST 区域不计分并留白。',y,8)
end()
for r in audit['fixed_cases']:
    y=start(f'Case {r["case"]:02d} | record {r["record_id"]}',f'Test index {r["index"]} | target month {r["month"]} | center crop 224 x 224')
    y=picture(f'case_{r["case"]:02d}.png',y,650);end()
y=start('8 个 case：统一残差对照')
y=picture('residuals_8cases.png',y,650)
y=para('四列均为预测减参考，固定 ±5°C。bias 是该图有效像素的平均有符号误差；残差图只作定位诊断。',y,8);end()
y=start('评估协议与完成验收')
y=para('数据：HLC processed_data/v1 完整 test；固定中心 [16:240,16:240]，224×224，无增强。HLS、fine LST、mask 共用窗口；Wald32 合成粗温度和目标 / mask 对所有模型一致。归一化参数来自训练集。',y)
y=para('检查点：UNet x0 / epsilon 均为冻结 epoch 1 EMA；MMDiT 为完整 validation、50 NFE 指标选出的 epoch 2 best EMA，与 final 的 EMA 张量相同。UNet 使用各自原训练的 HLS 预处理及冻结 Prithvi；MMDiT 输入原始 HLS 和有效 mask。',y)
y=para('推理：batch 32，最后一批 20 条；512 条分块，最后一块 212 条。每批固定 seed = 20261015 + 有序 batch 起点，每个模型 / NFE 重置。BF16 模型推理，摄氏度 FP32，误差累积 FP64。每个 5400 秒段保存后续接；只运行一个 A100 任务。',y)
y=para('验收：277 个分块 CSV 的 SHA256 全部通过；索引连续、record ID 与月份逐条匹配源 test 索引，141,524 个成员无重漏；所有逐条误差有限，每模型有效像素数一致；逐记录平方误差与绝对误差累积匹配 complete.json。16 个保存 case 的索引符合预设。',y)
y=para('硬件：请求 A100-40GB，部分段实际分配 A100-SXM4-80GB，最后一段为 40GB；峰值 PyTorch allocated 为 10,379,727,872 bytes（10.38 GB），不等于整卡总占用。读取最终状态时 runners=0、running=0、backlog=0。',y)
y=para('证据目录：output/hlc_unet_mmdit_test_20261015<br/>test/complete.json、test/protocol.json、test/chunks/*.csv / *.json / *.npz<br/>results/verification.json、extended_audit.json、main_table.csv、per_record_metrics.csv、case_metrics.json',y,8)
y=para('运行：hlc-unet-mmdit-fulltest-20261015-v1<br/>评估脚本：experiments/efdiff_hlc/test_comparison_20261015.py<br/>本地核验及绘图：test_comparison_results_20261015.py、test_comparison_audit_20261016.py',y,8)
y=para('评估代码 SHA256:<br/>'+protocol['eval_code_sha256'][:32]+'<br/>'+protocol['eval_code_sha256'][32:],y,8)
end();c.save();print(json.dumps({'pdf':str(OUT),'pages':page}))
