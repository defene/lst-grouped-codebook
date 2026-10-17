"""Show every image for which VAR has greater RMSE than the 8x8 baseline."""
import json
from PIL import Image, ImageDraw, ImageFont
from experiments.openimages_var200.prepare import OUT, save


def main():
    folder = OUT/'bicubic_8x8'
    rows = [json.loads(s) for s in (folder/'per_image.jsonl').read_text(encoding='utf-8').splitlines()]
    failures = [dict(r, sample_number=i+1, rmse_increase_255=r['var_rmse_255']-r['bicubic_rmse_255'])
                for i, r in enumerate(rows) if r['var_rmse_255'] > r['bicubic_rmse_255']]
    assert len(failures) == 3
    canvas = Image.new('RGB', (804, 965), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    title = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    draw.text((12, 8), 'VAR 的 RMSE 更高的全部3张：原图 / bicubic / VAR', font=title, fill='black')
    draw.text((12, 42), '按200张原始抽样顺序排列；误差单位为0–255', font=font, fill='black')
    for j, r in enumerate(failures):
        y = 76+295*j
        draw.text((12, y), f"{r['sample_number']:03d} {r['image_id']}   bicubic {r['bicubic_rmse_255']:.2f}   VAR {r['var_rmse_255']:.2f}   增加 {r['rmse_increase_255']:.2f}", font=font, fill='black')
        for col, path in enumerate([OUT/'inputs_256', folder/'reconstructions_256', OUT/'reconstructions_256']):
            with Image.open(path/(r['image_id']+'.png')) as im:
                assert im.size == (256, 256)
                canvas.paste(im, (12+264*col, y+28))
    canvas.save(folder/'var_higher_rmse_all3.png')
    save(folder/'var_higher_rmse_all3.json', failures)
    print(json.dumps(failures, indent=2))


if __name__ == '__main__':
    main()
