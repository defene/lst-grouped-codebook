"""Independent saved-artifact checks for the fixed 200-image reconstruction run."""
import json
import math
import numpy as np
from PIL import Image
from experiments.openimages_var200.prepare import OUT, sha, save


def main():
    selection = json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (OUT/'per_image.jsonl').read_text(encoding='utf-8').splitlines()]
    arrays = np.load(OUT/'reconstruction_arrays.npz', allow_pickle=False)
    ids = arrays['image_ids'].tolist()
    assert ids == [r['ImageID'] for r in selection['records']] == [r['image_id'] for r in rows]
    assert len(ids) == len(set(ids)) == 200
    pred, target, tokens = arrays['reconstruction_rgb_float32'], arrays['input_rgb_u8'], arrays['indices']
    assert pred.shape == target.shape == (200, 256, 256, 3)
    assert tokens.shape == (200, 680) and tokens.min() >= 0 and tokens.max() < 4096
    assert np.isfinite(pred).all() and pred.min() >= 0 and pred.max() <= 1
    recalculated = []
    for i, image_id in enumerate(ids):
        error = pred[i].astype(np.float64) - target[i].astype(np.float32)/255
        mse = float(np.mean(error**2))
        recalculated.append(mse)
        assert math.isclose(mse, rows[i]['mse_01'], rel_tol=1e-10, abs_tol=1e-12)
        assert math.isclose(-10*math.log10(mse), rows[i]['psnr_db'], abs_tol=1e-9)
        for folder, expected in [('inputs_256', target[i]), ('reconstructions_256', np.rint(pred[i]*255).astype(np.uint8))]:
            with Image.open(OUT/folder/f'{image_id}.png') as im:
                assert im.size == (256, 256) and np.array_equal(np.array(im), expected)
    assert math.isclose(math.sqrt(np.mean(recalculated))*255, summary['rmse_255'], abs_tol=1e-9)
    assert np.array_equal(np.bincount(tokens.ravel(), minlength=4096), np.load(OUT/'code_counts.npy'))
    assert summary['tokens']['tokens'] == 136000
    for scale in summary['tokens_by_scale']:
        assert scale['tokens'] == 200*scale['grid']**2
    for folder in ['inputs_256', 'reconstructions_256']:
        assert len(list((OUT/folder).glob('*.png'))) == 200
    assert len(list((OUT/'gallery').glob('comparison_*.png'))) == 25
    for p in (OUT/'gallery').glob('*.png'):
        with Image.open(p) as im:
            im.load()
            assert im.size == ((1070, 1235) if p.name == 'details_first4.png' else (1060, 1228))
    for name, digest in json.loads((OUT/'artifact_hashes.json').read_text(encoding='utf-8')).items():
        assert sha(OUT/name) == digest, name
    result = {'status': 'passed', 'images_verified': 200, 'input_pngs': 200, 'reconstruction_pngs': 200,
              'gallery_pages': 25, 'detail_pages': 1, 'metrics_recalculated_from_arrays': True,
              'pngs_match_saved_arrays': True, 'token_histogram_verified': True, 'artifact_hashes_verified': True}
    save(OUT/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
