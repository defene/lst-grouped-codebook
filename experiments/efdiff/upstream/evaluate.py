"""
Evaluate LST super-resolution predictions saved by infer.py.

Computes per-patch and per-tile metrics (RMSE, MAE, PSNR, SSIM, R², Pearson,
LPIPS) and distribution-level FID. Compares against bicubic baseline and
optionally generates diagnostic plots.

Output (written to <pred_dir>/):
    metrics_per_patch.csv   - one row per patch
    metrics_per_tile.csv    - tile-level means
    metrics_summary.json    - aggregated statistics + improvement over bicubic
    figures/                - diagnostic plots (with --plot)

Usage:
    python evaluate.py --pred_dir results/run_name
    python evaluate.py --pred_dir results/run_name --plot
    python evaluate.py --pred_dir results/run_name --qa_mask
"""
import json
import csv
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm



from lst_dataset import LST_RANGE

logger = logging.getLogger('eval')


# ---------------------------------------------------------------------------
# Per-patch metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, gt: np.ndarray, lr_up: np.ndarray,
                    mask: np.ndarray = None) -> dict:
    """Compute all metrics for a single (H, W) patch pair in Celsius.

    Args:
        mask: Optional boolean mask (H, W). True = valid pixel to include.
              If provided, only masked pixels are used for pixel-level metrics.
    """
    if mask is not None:
        pred_m = pred[mask]
        gt_m = gt[mask]
        lr_up_m = lr_up[mask]
        n_valid = int(mask.sum())
    else:
        pred_m = pred.ravel()
        gt_m = gt.ravel()
        lr_up_m = lr_up.ravel()
        n_valid = pred.size

    diff = pred_m - gt_m

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae  = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    psnr = float(10.0 * np.log10(LST_RANGE ** 2 / (np.mean(diff ** 2) + 1e-10)))

    ssim = _ssim(pred, gt, mask)

    # Gradient RMSE (spatial edge accuracy)
    gy_pred, gx_pred = np.gradient(pred)
    gy_gt, gx_gt = np.gradient(gt)
    grad_rmse = float(np.sqrt(np.mean((gx_pred - gx_gt)**2 + (gy_pred - gy_gt)**2)))

    # PSD log-ratio (frequency content match)
    psd_score = _psd_log_ratio(pred, gt)

    result = {
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'psnr': psnr,
        'ssim': ssim,
        'grad_rmse': grad_rmse,
        'psd_score': psd_score,
    }
    if mask is not None:
        result['n_valid'] = n_valid
        result['valid_ratio'] = float(n_valid / pred.size)
    return result


def _psd_log_ratio(pred: np.ndarray, gt: np.ndarray) -> float:
    """Radially averaged PSD log-ratio. Lower = better frequency match.

    Computes 2D FFT of both images, radially averages the power spectra,
    and returns the mean absolute log-ratio across frequency bins.
    A score of 0 means perfect frequency content match.
    """
    def _radial_psd(img):
        f = np.fft.fft2(img - img.mean())
        psd2d = np.abs(f) ** 2
        h, w = img.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[-cy:h-cy, -cx:w-cx]
        R = np.sqrt(X**2 + Y**2).astype(int)
        psd2d_shifted = np.fft.fftshift(psd2d)
        max_r = min(cy, cx)
        radial = np.zeros(max_r)
        for r in range(max_r):
            mask_r = R == r
            if mask_r.any():
                radial[r] = psd2d_shifted[mask_r].mean()
        return radial

    psd_pred = _radial_psd(pred)
    psd_gt = _radial_psd(gt)

    # Avoid log(0)
    eps = 1e-10
    log_ratio = np.abs(np.log10(psd_pred + eps) - np.log10(psd_gt + eps))
    return float(log_ratio.mean())


def _ssim(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """SSIM via skimage if available, else NaN. Uses GT dynamic range.

    When mask is provided, SSIM is still computed on the full image (it's a
    structural metric requiring spatial context), but the data_range is
    determined from the masked region only.
    """
    try:
        from skimage.metrics import structural_similarity
        if mask is not None:
            gt_range = float(gt[mask].max() - gt[mask].min())
        else:
            gt_range = float(gt.max() - gt.min())
        return float(structural_similarity(
            pred, gt, data_range=max(gt_range, 1e-6)))
    except ImportError:
        return float('nan')


def _load_lpips_fn():
    """Load LPIPS model (lazy, returns None if unavailable)."""
    try:
        import torch
        import lpips
        fn = lpips.LPIPS(net='alex', verbose=False).eval()
        if torch.cuda.is_available():
            fn = fn.cuda()
        logger.info('LPIPS loaded (alex)')
        return fn
    except ImportError:
        logger.warning('LPIPS not available (pip install lpips)')
        return None


def compute_lpips(pred: np.ndarray, gt: np.ndarray, lpips_fn) -> float:
    """Compute LPIPS for a single (H, W) patch pair."""
    import torch
    # Normalize to [-1, 1] using GT range for both
    g_min, g_max = gt.min(), gt.max()
    g_range = g_max - g_min
    p_norm = (pred - g_min) / (g_range + 1e-10) * 2 - 1
    g_norm = (gt - g_min) / (g_range + 1e-10) * 2 - 1
    # Expand to (1, 3, H, W) — LPIPS expects 3-channel
    p_t = torch.from_numpy(p_norm).float().unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
    g_t = torch.from_numpy(g_norm).float().unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
    if next(lpips_fn.parameters()).is_cuda:
        p_t, g_t = p_t.cuda(), g_t.cuda()
    with torch.no_grad():
        return float(lpips_fn(p_t, g_t).item())


# ---------------------------------------------------------------------------
# FID (Fréchet Inception Distance) — distribution-level metric
# ---------------------------------------------------------------------------

def _load_inception():
    """Load Inception-v3 for FID feature extraction. Returns (model, device) or (None, None)."""
    try:
        import torch
        from torchvision import models
        model = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        # Replace final FC with identity to get 2048-d features
        model.fc = torch.nn.Identity()
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f'Inception-v3 loaded for FID (device={device})')
        return model, device
    except ImportError:
        logger.warning('FID not available (requires torch, torchvision)')
        return None, None


def _normalize_for_inception(img: np.ndarray) -> np.ndarray:
    """Normalize single-channel Celsius image to [0, 1] range."""
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _extract_inception_features(images: list, model, device, batch_size=32,
                                ref_images: list = None):
    """Extract Inception-v3 pool3 features from a list of (H, W) numpy arrays.

    Each image is normalized to [0,1] using the corresponding ref_image's range
    (if provided), replicated to 3 channels, and resized to 299x299.
    Returns (N, 2048) numpy array.
    """
    import torch
    from torch.nn import functional as F

    features = []
    for i in range(0, len(images), batch_size):
        batch_np = images[i:i + batch_size]
        ref_batch = ref_images[i:i + batch_size] if ref_images else [None] * len(batch_np)
        tensors = []
        for img, ref in zip(batch_np, ref_batch):
            if ref is not None:
                lo, hi = ref.min(), ref.max()
                if hi - lo < 1e-6:
                    img_norm = np.zeros_like(img)
                else:
                    img_norm = (img - lo) / (hi - lo)
            else:
                img_norm = _normalize_for_inception(img)
            # (1, H, W) -> (3, H, W)
            t = torch.from_numpy(img_norm).float().unsqueeze(0).expand(3, -1, -1)
            tensors.append(t)
        batch = torch.stack(tensors).to(device)  # (B, 3, H, W)
        # Resize to 299x299 for Inception
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        feat = model(batch)  # (B, 2048)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(pred_features: np.ndarray, gt_features: np.ndarray) -> float:
    """Compute FID between two sets of Inception features, each (N, 2048)."""
    from scipy.linalg import sqrtm

    mu_p = pred_features.mean(axis=0)
    mu_g = gt_features.mean(axis=0)
    sigma_p = np.cov(pred_features, rowvar=False)
    sigma_g = np.cov(gt_features, rowvar=False)

    diff = mu_p - mu_g
    covmean, _ = sqrtm(sigma_p @ sigma_g, disp=False)

    # Numerical stability: discard imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_p + sigma_g - 2 * covmean))
    return fid


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _agg(values: list) -> dict:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return {'mean': float('nan'), 'std': float('nan'),
                'median': float('nan'), 'p25': float('nan'), 'p75': float('nan')}
    return {
        'mean':   float(arr.mean()),
        'std':    float(arr.std()),
        'median': float(np.median(arr)),
        'p25':    float(np.percentile(arr, 25)),
        'p75':    float(np.percentile(arr, 75)),
    }



def _tile_id(patch_path: str) -> str:
    for part in Path(patch_path).parts:
        if len(part) == 6 and part[0] == 'T' and part[1:3].isdigit() and part[3:].isalpha():
            return part
    return 'unknown'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

HR_SIZE = 224  # Must match inference crop size


def _load_qa_mask(patch_path: str, hr_size: int = HR_SIZE) -> np.ndarray | None:
    """Load qa_pixel.npy and return a boolean clear-sky mask (hr_size, hr_size).

    Landsat QA_PIXEL bit 6 = 1 means "clear". Center-crops to match the
    inference patch size.

    Returns None if qa_pixel.npy is not found.
    """
    qa_path = Path(patch_path) / 'qa_pixel.npy'
    if not qa_path.exists():
        return None
    qa = np.load(qa_path)  # (256, 256) uint16
    if qa.ndim == 3:
        qa = qa[0]
    h, w = qa.shape
    i = (h - hr_size) // 2
    j = (w - hr_size) // 2
    qa_crop = qa[i:i + hr_size, j:j + hr_size]
    # Bit 6 = clear
    clear = (qa_crop.astype(np.uint16) >> 6 & 1).astype(bool)
    return clear


METRIC_COLS = ['rmse', 'mae', 'bias', 'psnr', 'ssim', 'grad_rmse', 'psd_score']

LPIPS_COL = 'lpips'
FID_KEY = 'fid'


def main():
    parser = argparse.ArgumentParser(description='Evaluate LST SR predictions')
    parser.add_argument('--pred_dir', required=True,
                        help='Output directory from infer.py')
    parser.add_argument('--plot', action='store_true',
                        help='Generate diagnostic figures (requires matplotlib, skimage)')
    parser.add_argument('--qa_mask', action='store_true',
                        help='Mask non-clear pixels using qa_pixel.npy (Landsat QA_PIXEL bit 6)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    pred_dir  = Path(args.pred_dir)
    pred_subdir = pred_dir / 'predictions'
    index_path  = pred_dir / 'index.json'

    if not index_path.exists():
        raise FileNotFoundError(
            f'index.json not found in {pred_dir}. Run infer.py first.')

    with open(index_path) as f:
        index = json.load(f)

    use_qa = args.qa_mask
    logger.info(f'Evaluating {len(index)} predictions from {pred_dir}'
                + (' (QA clear-sky mask)' if use_qa else ''))

    # ---- Load perceptual models (graceful fallback if unavailable) ----
    lpips_fn = _load_lpips_fn()
    use_lpips = lpips_fn is not None

    inception_model, inception_device = _load_inception()
    use_fid = inception_model is not None

    metric_cols = METRIC_COLS + ([LPIPS_COL] if use_lpips else [])

    # ---- Per-patch metrics ----
    rows = []
    n_qa_found, n_qa_skipped = 0, 0
    all_preds, all_gts = [], []  # collect for FID
    for rec in tqdm(index, desc='Computing metrics'):
        idx   = rec['idx']
        patch = rec.get('patch_path', '')
        pfx   = f'{idx:06d}'

        pred  = np.load(pred_subdir / f'{pfx}_sr.npy')
        gt    = np.load(pred_subdir / f'{pfx}_gt.npy')
        lr_up = np.load(pred_subdir / f'{pfx}_lr_up.npy')

        mask = None
        if use_qa:
            mask = _load_qa_mask(patch)
            if mask is not None:
                n_qa_found += 1
                if mask.sum() < 0.10 * mask.size:
                    n_qa_skipped += 1
                    continue

        m = compute_metrics(pred, gt, lr_up, mask=mask)

        if use_lpips:
            m[LPIPS_COL] = compute_lpips(pred, gt, lpips_fn)

        if use_fid:
            all_preds.append(pred)
            all_gts.append(gt)

        m['idx']        = idx
        m['tile_id']    = _tile_id(patch)
        m['patch_path'] = patch
        rows.append(m)

    if use_qa:
        logger.info(f'QA mask: {n_qa_found}/{len(index)} had qa_pixel.npy, '
                    f'{n_qa_skipped} skipped (<10% clear), '
                    f'{len(rows)} evaluated')

    # ---- FID (distribution-level) ----
    fid_value = float('nan')
    if use_fid and len(all_preds) > 1:
        logger.info('Extracting Inception features for FID...')
        import torch
        with torch.no_grad():
            pred_feats = _extract_inception_features(
                all_preds, inception_model, inception_device,
                ref_images=all_gts)
            gt_feats = _extract_inception_features(
                all_gts, inception_model, inception_device)
        fid_value = compute_fid(pred_feats, gt_feats)
        logger.info(f'FID = {fid_value:.2f}')

    # ---- Per-patch CSV ----
    csv_path = pred_dir / 'metrics_per_patch.csv'
    extra_cols = []
    if use_qa and rows and 'n_valid' in rows[0]:
        extra_cols = ['n_valid', 'valid_ratio']
    fieldnames = ['idx', 'tile_id'] + metric_cols + extra_cols + ['patch_path']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f'Per-patch metrics -> {csv_path}')

    # ---- Summary JSON ----
    summary = {col: _agg([r[col] for r in rows]) for col in metric_cols}
    summary['n_patches'] = len(rows)
    summary['n_tiles']   = len({r['tile_id'] for r in rows})
    summary[FID_KEY] = fid_value

    json_path = pred_dir / 'metrics_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f'Summary          -> {json_path}')

    # ---- Per-tile CSV ----
    tile_vals: dict = {}
    for r in rows:
        t = r['tile_id']
        tile_vals.setdefault(t, {c: [] for c in metric_cols})
        for c in metric_cols:
            tile_vals[t][c].append(r.get(c, float('nan')))

    tile_csv = pred_dir / 'metrics_per_tile.csv'
    with open(tile_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['tile_id', 'n_patches'] + metric_cols)
        writer.writeheader()
        for tile_id, vals in sorted(tile_vals.items()):
            row = {'tile_id': tile_id, 'n_patches': len(vals['rmse'])}
            row.update({c: float(np.nanmean(vals[c])) for c in metric_cols})
            writer.writerow(row)
    logger.info(f'Per-tile metrics -> {tile_csv}')

    # ---- Console summary ----
    _print_summary(rows, summary)

    # ---- Optional plots ----
    if args.plot:
        _make_plots(rows, tile_vals, pred_dir)


def _print_summary(rows, summary):
    n_tiles = summary['n_tiles']
    n = len(rows)

    logger.info(f'\n{"="*62}')
    logger.info(f'  Results: {n} patches, {n_tiles} tiles')
    logger.info(f'{"="*62}')
    logger.info(f'  {"Metric":<18} {"Value":>9}')
    logger.info(f'  {"-"*30}')

    for label, col in [
        ('RMSE  (degC)', 'rmse'),
        ('MAE   (degC)', 'mae'),
        ('PSNR   (dB)',  'psnr'),
        ('SSIM',         'ssim'),
        ('Grad RMSE',    'grad_rmse'),
        ('PSD Score',    'psd_score'),
        ('Bias   (degC)', 'bias'),
    ]:
        val = summary[col]['mean']
        if not np.isnan(val):
            logger.info(f'  {label:<18} {val:>9.4f}')

    if LPIPS_COL in summary:
        lpips_val = summary[LPIPS_COL]['mean']
        if not np.isnan(lpips_val):
            logger.info(f'  {"LPIPS":<18} {lpips_val:>9.4f}')

    if FID_KEY in summary:
        fid_val = summary[FID_KEY]
        if not np.isnan(fid_val):
            logger.info(f'  {"FID":<18} {fid_val:>9.2f}')

    logger.info('')


def _make_plots(rows, tile_vals, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # 1. Metric distributions
    cols   = ['rmse', 'mae', 'bias', 'ssim']
    labels = ['RMSE (degC)', 'MAE (degC)', 'Bias (degC)', 'SSIM']
    # Add LPIPS if computed
    if rows and LPIPS_COL in rows[0]:
        cols.append('lpips')
        labels.append('LPIPS')

    n_plots = len(cols)
    n_cols_fig = min(n_plots, 4)
    n_rows_fig = (n_plots + n_cols_fig - 1) // n_cols_fig
    fig, axes = plt.subplots(n_rows_fig, n_cols_fig, figsize=(5 * n_cols_fig, 4 * n_rows_fig))
    axes = np.array(axes).flatten()

    for i, (col, label) in enumerate(zip(cols, labels)):
        ax = axes[i]
        vals = [r[col] for r in rows if col in r and not np.isnan(r[col])]
        if not vals:
            continue
        ax.hist(vals, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(np.mean(vals), color='tomato', linestyle='--',
                   label=f'mean={np.mean(vals):.3f}')
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Per-patch metric distributions', fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'metric_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Per-tile RMSE boxplot (top 20 tiles by count)
    top_tiles = sorted(tile_vals, key=lambda t: -len(tile_vals[t]['rmse']))[:20]
    tile_data = [tile_vals[t]['rmse'] for t in top_tiles]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(tile_data, labels=top_tiles)
    ax.set_xticklabels(top_tiles, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('RMSE (degC)')
    ax.set_title('Per-Tile RMSE (top 20 tiles by patch count)')
    plt.tight_layout()
    plt.savefig(fig_dir / 'per_tile_rmse.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f'Figures -> {fig_dir}')


if __name__ == '__main__':
    main()
