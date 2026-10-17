"""
Standalone inference for LST super-resolution.

Runs the trained diffusion model on the test set and saves per-patch predictions
as numpy arrays to disk. Use evaluate.py to compute metrics from saved predictions.

Output layout:
    <output_dir>/
    +-- predictions/
    |   +-- 000000_sr.npy      # SR prediction in degC  (224, 224)
    |   +-- 000000_gt.npy      # Ground truth in degC   (224, 224)
    |   +-- 000000_lr_up.npy   # Bicubic LR in degC     (224, 224)
    |   +-- ...
    +-- index.json             # idx -> patch_path mapping
    +-- run_meta.json          # config / checkpoint used
    +-- infer.log

Usage:
    python infer.py -c configs/config_sr3_prithvi.yaml \\
                    --checkpoint /path/to/I100000_E200 \\
                    -o results/run_name

    # Then evaluate:
    python evaluate.py --pred_dir results/run_name [--plot]
"""
import os
import json
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import model as Model
import util.logger as Logger
from util.setup import load_norm_stats
from util.temperature import get_model_info, condition_to_celsius, reconstruct_temperature
from lst_dataset import LSTDataset


def build_args():
    parser = argparse.ArgumentParser(description='LST SR Inference')
    parser.add_argument('-c', '--config', required=True,
                        help='YAML config file (same one used for training)')
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint prefix without _gen.pth suffix, '
                             'e.g. /path/to/experiments/.../checkpoint/I100000_E200')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Directory to write predictions and logs')
    parser.add_argument('--data_dir', default=None,
                        help='Override test dataroot from config')
    parser.add_argument('--data_len', type=int, default=-1,
                        help='Max test samples (-1 = all)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Inference batch size (default: 8)')
    parser.add_argument('--ddim_steps', type=int, default=None,
                        help='DDIM sampling steps for SR3 (default: None = full DDPM)')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-p', '--phase', default='test')
    return parser.parse_args()


def main():
    args = build_args()

    # ---- Config ----
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    opt['path']['resume_state'] = args.checkpoint
    opt['phase'] = 'test'

    # Validate checkpoint files exist
    for suffix in ('_gen.pth',):
        ckpt_file = f'{args.checkpoint}{suffix}'
        if not os.path.isfile(ckpt_file):
            raise FileNotFoundError(
                f'Checkpoint file not found: {ckpt_file}\n'
                f'Expected format: /path/to/I{{iter}}_E{{epoch}} (without _gen.pth suffix)')

    # ---- Output directory ----
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ---- Logging ----
    Logger.setup_logger(None, str(output_dir), 'infer',
                        level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(f'Config:     {args.config}')
    logger.info(f'Checkpoint: {args.checkpoint}')
    logger.info(f'Output:     {output_dir}')

    # ---- Model variant ----
    model_type, needs_hls, model_label = get_model_info(opt)
    logger.info(f'Model type: {model_type}')

    # ---- Normalization stats ----
    norm_stats = load_norm_stats(opt)

    # ---- Dataset ----
    eval_cfg = opt['datasets'].get('test', opt['datasets'].get('val', {}))
    data_dir = args.data_dir or eval_cfg['dataroot']
    data_len = args.data_len if args.data_len != -1 else eval_cfg.get('data_len', -1)

    test_set = LSTDataset(
        data_dir=data_dir,
        split='val',
        norm_stats=norm_stats,
        load_hls=needs_hls,
        data_len=data_len,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    logger.info(f'Test set: {len(test_set)} patches')

    # ---- Model ----
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['test'], schedule_phase='test')
    logger.info('Model loaded.')

    # ---- Inference loop ----
    index_records = []
    sample_idx = 0
    for batch_data in tqdm(test_loader, total=len(test_loader), desc='Inference'):
        diffusion.feed_data(batch_data)
        diffusion.test(continuous=False, verbose=False,
                       ddim_steps=args.ddim_steps)
        visuals = diffusion.get_current_visuals(need_LR=False)

        batch_size = visuals['SR'].shape[0]
        for b in range(batch_size):
            pred_celsius = reconstruct_temperature(
                visuals['SR'][b:b+1], visuals['INF'][b:b+1], norm_stats)
            gt_celsius = reconstruct_temperature(
                visuals['HR'][b:b+1], visuals['INF'][b:b+1], norm_stats)
            lr_up_celsius = condition_to_celsius(
                visuals['INF'][b:b+1], norm_stats)

            prefix = f'{sample_idx:06d}'
            np.save(pred_dir / f'{prefix}_sr.npy',
                    pred_celsius.astype(np.float32))
            np.save(pred_dir / f'{prefix}_gt.npy',
                    gt_celsius.astype(np.float32))
            np.save(pred_dir / f'{prefix}_lr_up.npy',
                    lr_up_celsius.astype(np.float32))

            patch_path = (test_set.patches[sample_idx]
                          if sample_idx < len(test_set.patches) else '')
            index_records.append({'idx': sample_idx, 'patch_path': patch_path})
            sample_idx += 1

    # ---- Index and metadata ----
    with open(output_dir / 'index.json', 'w') as f:
        json.dump(index_records, f, indent=2)

    run_meta = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'data_dir': str(data_dir),
        'model_type': model_type,
        'n_samples': len(test_set),
    }
    with open(output_dir / 'run_meta.json', 'w') as f:
        json.dump(run_meta, f, indent=2)

    logger.info(f'Saved {len(index_records)} predictions to {pred_dir}')
    logger.info(f'Next:  python evaluate.py --pred_dir {output_dir}')


if __name__ == '__main__':
    main()
