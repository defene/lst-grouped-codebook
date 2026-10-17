"""Assemble the verified HLC MMDiT results and fixed-case atlas."""
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
BASE=ROOT/'output/efdiff_hlc_raw_mmdit_b256_v_20261014'
FIG=BASE/'final_results'
DATA=json.loads((FIG/'results.json').read_text())
OUT=ROOT/'output/pdf/hlc_mmdit_training_results_20261015.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
W,H=595.276,841.89
c=canvas.Canvas(str(OUT),pagesize=(W,H))
c.setTitle('HLC Raw-HLS MMDiT - Overall Training Results')
c.setAuthor('HLC experiment analysis')
body=ParagraphStyle('body',fontName='CN',fontSize=10,leading=16,textColor=HexColor('#233047'))
small=ParagraphStyle('small',parent=body,fontSize=8,leading=12)
page=0

def para(text,y,size=10):
    p=Paragraph(text,body if size==10 else small)
    _,h=p.wrap(W-76,700);p.drawOn(c,38,y-h)
    return y-h-10

def start(title,sub='HLS-LST-CrossSensor (HLC) | 2026-10-15'):
    global page
    page+=1;c.setFillColor(HexColor('#102942'));c.rect(0,H-9,W,9,fill=1,stroke=0)
    c.setFont('CN',20);c.drawString(38,H-54,title)
    c.setFont('CN',8);c.setFillColor(HexColor('#60738a'));c.drawString(38,H-73,sub)
    return H-94

def end():
    c.setFillColor(HexColor('#60738a'));c.setFont('CN',8)
    c.drawString(38,24,'HLC | Synthetic degradation | Validation, not test')
    c.drawRightString(W-38,24,str(page));c.showPage()

def table(rows,y,widths):
    formatted=[[Paragraph(str(v),small) for v in row] for row in rows]
    t=Table(formatted,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e4edf5')),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#f4f7fa')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,HexColor('#91a9bc'))]))
    _,h=t.wrap(W-76,700);t.drawOn(c,38,y-h);return y-h-15

def picture(name,y,maxheight):
    p=FIG/name;i=ImageReader(str(p));iw,ih=i.getSize();scale=min((W-76)/iw,maxheight/ih)
    dw,dh=iw*scale,ih*scale;c.drawImage(i,(W-dw)/2,y-dh,width=dw,height=dh);return y-dh-12

y=start('整体结果：Raw-HLS MMDiT')
y=para('两轮训练已完成。50 NFE 下，第二轮同时是验证最优和最终检查点；相对旧 TimeGeo DiT 的验证最优结果，RMSE 降低 24.3%。第二轮相对第一轮继续改善 2.68%，但局部细节与 patch 接缝仍有不足。',y)
y=table([['完整验证主表 / 50 NFE','RMSE (°C)','MAE (°C)']]+[[r['model'],f"{r['rmse_c']:.4f}",f"{r['mae_c']:.4f}"] for r in DATA['main_table_50nfe']],y,[299,110,110])
y=para('统一评估成员：141,711 条完整 validation 记录；有效像素总数 7,083,539,993。指标按有效像素加权，单位摄氏度。没有使用 test 选模。旧 DiT 与新模型的参考温度、mask 和验证成员已核对。',y)
y=table([['新 MMDiT / best = final','RMSE (°C)','MAE (°C)']]+[[f"{r['nfe']} NFE",f"{r['rmse_c']:.4f}",f"{r['mae_c']:.4f}"] for r in DATA['nfe_table']],y,[299,110,110])
y=para('5 NFE 在本次完整验证上的 RMSE/MAE 最低，但 best 检查点仍按预先指定的 50 NFE 选择。不能把较低 RMSE 等同于更好的细节或更少的平滑；采样步数结果需要与图像一起看。',y)
y=para('对照边界：新模型同时改变了条件编码器、融合方式与架构，第二轮 batch 也由 128 调至 256；这不是单独验证 MMDiT、adaLN 或移除 Prithvi 的消融。历史 UNet 的训练裁剪、损失、预处理和预算不同，仅作为同位置 8 case 参考，不混入这张全量主表。',y)
end()

y=start('训练预算与收敛')
y=table([['阶段','呈现记录数','更新次数','batch / 尾批'],['Epoch 1','2,608,772','20,382','128 / 4'],['Epoch 2','2,608,772','10,191','256 / 132'],['合计','5,217,544','30,573','混合 batch']],y,[112,140,110,157])
y=picture('training_curves.png',y,390)
y=para('训练 loss 为归一化残差的 v-pred MSE；验证分数为摄氏度误差，二者不能按数值直接比较。图中浅线为稀疏日志，平滑线按每 128,000 次样本呈现分箱，计算已记录 full-batch loss 的均值，不代表全部更新的精确均值。尾批不纳入平滑。',y,8)
y=para('最后 20 个已记录 full batch 的平均 loss = 0.055706。后期下降已明显放缓，但只有两个 epoch 的完整验证，尚不能证明继续训练一定无收益，也不能只靠 loss 判定提前收敛。',y,8)
end()

y=start('采样步数与局部接缝')
y=picture('nfe_comparison.png',y,235)
y=picture('patch_boundary.png',y,255)
y=para('接缝诊断使用保存的 16 个 case：仅在相邻两像素都有效时，计算 16 像素 patch 边界与内部的平均绝对温差。参考图比值约 0.99；新模型 50 NFE 从第一轮的 2.06 降到第二轮的 1.39，说明接缝减轻但未消失。5 NFE 比值约 1.47，绝对边界和内部变化均更小。',y,8)
y=para('这项统计是小样本诊断，不是全量验证指标；平滑也会改变分母。当前输出头仍是每个 token 线性输出 16×16 像素后拼接，没有卷积细化头、VAE 或梯度辅助损失。该结构是可能因素，尚未通过单变量实验确定因果。',y,8)
end()

y=start('8 个固定 case：逐图指标')
y=para('沿用之前查看的同一组 8 个 case，核对 record ID、参考图、mask 及粗温度；不重新挑选最好看的样本。以下是单图有效像素 RMSE，不能作为全量分数。',y)
y=table([['Case / record','粗温度¹','UNet ε','旧 DiT²','新 50','新 5']]+[[f"{r['case']} / {r['record_id']}"]+[f"{r[k]['rmse_c']:.3f}" for k in ['coarse','unet_eps','old50','mmdit50','mmdit5']] for r in DATA['cases']],y,[139,76,76,76,76,76])
y=para('¹ 粗温度分数使用上采样至 224×224 的输入。后续图展示实际 7×7 合成粗温度。<br/>² 旧 DiT 图使用 epoch 2 final、50 NFE；主表的旧模型 best 为 epoch 1，因此对应数值口径不同。UNet 图为历史 epsilon epoch 1、50 NFE。',y,8)
y=para('每个 case 页包含 HLS RGB、合成粗温度、参考 fine LST、历史 UNet、旧 DiT、新 MMDiT 50/5 NFE，以及新模型的两张残差图。温度图在单个 case 内共用色标；不同 case 可有不同温度范围。残差统一为预测减参考，色标固定 ±5°C。灰色为无效区域。',y)
y=para('HLS 是同一空间位置的数据，但获取日期可能与目标 LST 不同，不能把每条光谱边缘都当成同一天的温度边缘。每页标明 HLS 与 LST 日期。RGB 仅为展示而使用固定范围与 gamma，模型有效反射率输入未做该显示截断。',y)
end()

for r in DATA['cases']:
    y=start(f"Case {r['case']:02d} | record {r['record_id']}",f"HLS {r['hls_date']} | LST {r['lst_date']} | Fixed center crop 224 x 224")
    y=picture(f"case_{r['case']:02d}.png",y,650)
    end()

y=start('训练协议与产物核验')
y=para('数据：HLC processed_data/v1，固定中心裁剪 [16:240,16:240]，224×224；HLS、LST 和 mask 共用窗口，无随机翻转。合成粗温度由 fine LST 退化得到，不能等同于真实跨传感器粗温度恢复测试。',y)
y=para('模型：217,195,328 个可训练参数，约比旧 TimeGeo DiT 多 0.092%。原始 6 波段 HLS、粗温度与完整有效 mask 构成条件输入；patch16，hidden896，14 heads，4 个条件编码层与 6 个联合 MMDiT 层。月份与地理编码和扩散时间通过融合 MLP 注入。无 Prithvi、无 VAE、无 REPA。',y)
y=para('优化：Adam，峰值学习率 2e-5，按 320,000 次样本呈现暖启，余弦衰减至 1e-6；EMA 按样本数保持时间尺度。v-pred，LST 有效像素 MSE；每轮完整 validation、50 NFE 选模。第二轮 batch256 从第一轮检查点续训，并非从零重新训练。',y)
y=para('硬件与状态：A100-SXM4-40GB，日志记录峰值 allocated 显存 13.779 GB；这是 PyTorch 分配量，不是整卡总占用。最终检查 GPU job runners=0、running=0、backlog=0。',y)
verification=BASE/'weight_download_verified.json'
if verification.exists():
    weights=json.loads(verification.read_text())
    y=para('权重下载核验：best_ema.pt 与 final_ema.pt 已下载；本地文件大小与 SHA256 均与远程 CPU 计算结果一致。',y)
    for name,m in weights['files'].items():
        y=para(f"{name}: {m['bytes']:,} bytes<br/>SHA256: {m['sha256'][:32]}<br/>{m['sha256'][32:]}",y,8)
else:
    y=para('权重下载核验尚未完成；以同目录 weight_download_verified.json 为最终凭证。',y)
y=para('可复核文件：final_results/results.json（指标与预算）、main_table.csv、case_metrics.csv、sources.json（证据文件哈希）；train/v 下保留完成记录、逐轮验证、评估 JSON/NPZ 与下载的 EMA 权重。',y,8)
y=para('运行 ID：hlc-raw-mmdit-v-b256-fullval-epoch2-20261014-v2<br/>本地：output/efdiff_hlc_raw_mmdit_b256_v_20261014<br/>绘图脚本：experiments/efdiff_hlc/raw_mmdit_final_results.py',y,8)
end();c.save()
print(json.dumps({'pdf':str(OUT),'pages':page,'weights_verified':verification.exists()}))
