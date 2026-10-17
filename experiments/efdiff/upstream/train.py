"""
LST Super-Resolution Training with Diffusion Models.

Supports four model variants via config (which_model_G):
  - sr3:              Standard UNet diffusion (residual learning)
  - sr3_prithvi:      UNet + Prithvi-EO-2.0 cross-attention
  - resshift:         Few-step residual diffusion
  - resshift_prithvi: ResShift + Prithvi-EO-2.0 cross-attention

Single-GPU:
    python train.py -c configs/config_sr3.yaml

Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=4 train.py -c configs/config_sr3.yaml

Inference and evaluation:
    python infer.py    -c configs/config_sr3_prithvi.yaml --checkpoint <path> -o <out_dir>
    python evaluate.py --pred_dir <out_dir> [--plot]
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import copy
import json
import math
import subprocess
import datetime
import torch
import argparse
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import random
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import model as Model
import util.logger as Logger
import util.metrics as Metrics
from util.logger import WandbLogger
from util.setup import load_norm_stats
from util.temperature import get_model_info, reconstruct_temperature, condition_to_celsius
from lst_dataset import LSTDataset, LST_RANGE
from tensorboardX import SummaryWriter


# ============================================================================
# Helpers
# ============================================================================

def get_lr_lambda(warmup_steps, total_steps, eta_min=1e-6, base_lr=1.0):
    """Return lr_lambda for LambdaLR: warmup + cosine decay to eta_min."""
    min_ratio = eta_min / base_lr
    def lr_lambda(step):
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.5 * (1 + math.cos(math.pi * progress)), min_ratio)
    return lr_lambda


def set_seed(seed, deterministic=False):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_experiment_metadata(opt, args):
    """Save resolved config and run metadata to experiment directory."""
    exp_root = opt['path']['experiments_root']
    os.makedirs(exp_root, exist_ok=True)

    # Save resolved config
    import yaml

    config_path = os.path.join(exp_root, 'config_used.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(dict(opt), f, default_flow_style=False)

    # Save run metadata
    git_hash = 'unknown'
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        pass

    meta = {
        'timestamp': datetime.datetime.now().isoformat(),
        'git_hash': git_hash,
        'command': ' '.join(['python'] + [str(a) for a in vars(args).values()]),
        'seed': opt.get('seed', 42),
        'python_version': '.'.join(str(x) for x in __import__('sys').version_info[:3]),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda or 'N/A',
    }
    meta_path = os.path.join(exp_root, 'run_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def is_main_process():
    """True if this is the rank-0 process (or non-distributed)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_distributed():
    """Initialize DDP if launched with torchrun. Returns local_rank or 0."""
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank >= 0:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Dataset construction
# ============================================================================

def build_datasets(opt, norm_stats, needs_hls, phase):
    """Build train/eval loaders and sampler. Returns (train_loader, eval_loader, train_sampler)."""
    logger = logging.getLogger('base')

    if phase == 'train':
        train_set = LSTDataset(
            data_dir=opt['datasets']['train']['dataroot'],
            split='train',
            norm_stats=norm_stats,
            load_hls=needs_hls,
            data_len=opt['datasets']['train']['data_len'],
        )

        # Split validation from training: pick n_val tiles, 1 patch each
        val_cfg = opt['datasets'].get('val', {})
        n_val = val_cfg.get('data_len', 10)
        val_seed = val_cfg.get('seed', 42)

        tile_to_indices = defaultdict(list)
        for i, p in enumerate(train_set.patches):
            tile_to_indices[LSTDataset._extract_tile_id(p)].append(i)

        tiles = sorted(tile_to_indices.keys())
        rng = random.Random(val_seed)
        val_tiles = rng.sample(tiles, min(n_val, len(tiles)))

        val_indices = set()
        for tile_id in val_tiles:
            val_indices.add(rng.choice(tile_to_indices[tile_id]))

        val_patches = [train_set.patches[i] for i in sorted(val_indices)]
        keep_patches = [p for i, p in enumerate(train_set.patches)
                        if i not in val_indices]

        train_set.patches = keep_patches
        train_set.data_len = len(keep_patches)

        eval_set = copy.copy(train_set)
        eval_set.patches = val_patches
        eval_set.data_len = len(val_patches)
        eval_set.split = 'test'  # center crop

        logger.info('Held out {} patches (1 per tile) for validation'.format(
            len(val_patches)))
        logger.info('Train dataset: {:d} patches'.format(len(train_set)))

        # DDP sampler if distributed
        train_sampler = (DistributedSampler(train_set)
                         if dist.is_initialized() else None)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=opt['datasets']['train']['batch_size'],
            shuffle=(train_sampler is None and opt['datasets']['train']['use_shuffle']),
            sampler=train_sampler,
            num_workers=opt['datasets']['train']['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        eval_batch_size = opt['datasets'].get('val', {}).get('batch_size', 4)
        eval_loader = torch.utils.data.DataLoader(
            eval_set, batch_size=eval_batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True)
        return train_loader, eval_loader, train_sampler
    else:
        eval_cfg = opt['datasets'].get('test', opt['datasets'].get('val', {}))
        eval_set = LSTDataset(
            data_dir=eval_cfg['dataroot'],
            split='test',
            norm_stats=norm_stats,
            load_hls=needs_hls,
            data_len=eval_cfg['data_len'],
        )
        eval_batch_size = eval_cfg.get('batch_size', 4)
        eval_loader = torch.utils.data.DataLoader(
            eval_set, batch_size=eval_batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True)
        logger.info('Test dataset: {:d} patches'.format(len(eval_set)))
        return None, eval_loader, None


# ============================================================================
# Validation
# ============================================================================

def validate(diffusion, eval_loader, norm_stats, result_path, model_label):
    """Run validation and return metrics dict + sample triples for visualization."""
    os.makedirs(result_path, exist_ok=True)

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_rmse = 0.0
    avg_mae = 0.0
    avg_bias = 0.0
    all_samples = []

    for val_data in tqdm(eval_loader, desc='validation'):
        diffusion.feed_data(val_data)
        diffusion.test(continuous=False, verbose=False)
        visuals = diffusion.get_current_visuals(need_LR=False)

        # Handle batched visuals: iterate per sample
        batch_size = visuals['SR'].shape[0]
        for b in range(batch_size):
            pred_celsius = reconstruct_temperature(
                visuals['SR'][b:b+1], visuals['INF'][b:b+1], norm_stats)
            gt_celsius = reconstruct_temperature(
                visuals['HR'][b:b+1], visuals['INF'][b:b+1], norm_stats)
            lr_up_celsius = condition_to_celsius(
                visuals['INF'][b:b+1], norm_stats)

            avg_rmse += np.sqrt(np.mean((pred_celsius - gt_celsius) ** 2))
            avg_mae += np.mean(np.abs(pred_celsius - gt_celsius))
            avg_bias += Metrics.calculate_bias(pred_celsius, gt_celsius)
            avg_psnr += Metrics.calculate_psnr(pred_celsius, gt_celsius)
            gt_range = gt_celsius.max() - gt_celsius.min()
            avg_ssim += Metrics.calculate_ssim(
                pred_celsius, gt_celsius,
                data_range=max(float(gt_range), 1e-6))

            all_samples.append((lr_up_celsius, pred_celsius, gt_celsius))

    n = len(all_samples)
    return {
        'psnr': avg_psnr / n,
        'ssim': avg_ssim / n,
        'rmse': avg_rmse / n,
        'mae': avg_mae / n,
        'bias': avg_bias / n,
    }, all_samples


# ============================================================================
# Visualization
# ============================================================================

def save_lst_grid(samples, save_path, n_cols=5, model_label='SR3'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    vmin = min(v.min() for s in samples for v in s)
    vmax = max(v.max() for s in samples for v in s)

    loss_maps = [pred - gt for (_, pred, gt) in samples]
    loss_abs_max = max(np.abs(lm).max() for lm in loss_maps)

    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 16))
    row_labels = ['LR (Bicubic Up)', '{} (Generated)'.format(model_label),
                  'HR (Ground Truth)', 'Error SR\u2212HR (\u00b0C)']

    for col, (lr_up, pred, gt) in enumerate(samples):
        for row, data in enumerate([lr_up, pred, gt]):
            ax = axes[row, col]
            ax.imshow(data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=12)

        ax = axes[3, col]
        im_loss = ax.imshow(loss_maps[col], cmap='RdBu_r',
                            vmin=-loss_abs_max, vmax=loss_abs_max)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(row_labels[3], fontsize=12)

    fig.subplots_adjust(right=0.88)
    cbar_ax1 = fig.add_axes([0.90, 0.32, 0.015, 0.56])
    fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap='RdYlBu_r'),
        cax=cbar_ax1, label='degC')
    cbar_ax2 = fig.add_axes([0.90, 0.08, 0.015, 0.18])
    fig.colorbar(im_loss, cax=cbar_ax2, label='degC')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Training step helpers
# ============================================================================

def log_training_step(diffusion, epoch, step, lr, tb_logger, wandb_logger):
    """Log training metrics (rank 0 only)."""
    logger = logging.getLogger('base')
    logs = diffusion.get_current_log()
    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.2e}> '.format(epoch, step, lr)
    for k, v in logs.items():
        message += '{:s}: {:.4e} '.format(k, v)
        if tb_logger:
            tb_logger.add_scalar(k, v, step)
    if tb_logger:
        tb_logger.add_scalar('train/lr', lr, step)
    logger.info(message)

    if wandb_logger:
        wandb_logger.log_metrics({
            **{k: v for k, v in logs.items()},
            'train/lr': lr,
            'step': step,
        })


def run_validation_step(diffusion, eval_loader, norm_stats, model_label,
                        epoch, step, result_path, val_step,
                        tb_logger, wandb_logger):
    """Run validation, save grids, log metrics. Returns updated val_step."""
    logger = logging.getLogger('base')

    metrics, all_samples = validate(
        diffusion, eval_loader, norm_stats, result_path, model_label)

    # Save grid images (show first 10 samples only)
    vis_samples = all_samples[:10]
    mid = len(vis_samples) // 2
    for grid_idx, chosen in enumerate([vis_samples[:mid], vis_samples[mid:]]):
        grid_path = os.path.join(
            result_path,
            '{}_grid_{}.png'.format(step, grid_idx + 1))
        save_lst_grid(chosen, grid_path,
                      n_cols=len(chosen), model_label=model_label)
        if wandb_logger:
            wandb_logger.log_image(
                'validation/sample_grid_{}'.format(grid_idx + 1), grid_path)

    logger.info(
        '# Validation # PSNR: {:.4e} | SSIM: {:.4f} '
        '| RMSE: {:.4f} C | MAE: {:.4f} C | Bias: {:+.4f} C'.format(
            metrics['psnr'], metrics['ssim'],
            metrics['rmse'], metrics['mae'], metrics['bias']))
    logger_val = logging.getLogger('val')
    logger_val.info(
        '<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, '
        'ssim: {:.4f}, rmse: {:.4f}, mae: {:.4f}, bias: {:+.4f}'.format(
            epoch, step,
            metrics['psnr'], metrics['ssim'],
            metrics['rmse'], metrics['mae'], metrics['bias']))

    if tb_logger:
        for key in ('psnr', 'ssim', 'rmse', 'mae', 'bias'):
            tb_logger.add_scalar('val/{}'.format(key), metrics[key], step)

    if wandb_logger:
        wandb_logger.log_metrics({
            **{'val/{}'.format(k): v for k, v in metrics.items()},
            'val_step': val_step,
        })
        val_step += 1

    return val_step


def save_checkpoint(diffusion, epoch, step, wandb_logger, log_wandb_ckpt):
    """Save model checkpoint (rank 0 only)."""
    logger = logging.getLogger('base')
    logger.info('Saving checkpoint...')
    diffusion.save_network(epoch, step)
    if wandb_logger and log_wandb_ckpt:
        wandb_logger.log_checkpoint(epoch, step)


# ============================================================================
# Training loop
# ============================================================================

def train_loop(diffusion, train_loader, eval_loader, train_sampler,
               opt, norm_stats, model_label,
               tb_logger=None, wandb_logger=None):
    """Main training loop. Can be called from tests or main()."""
    logger = logging.getLogger('base')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    base_lr = opt['train']['optimizer']['lr']
    warmup_steps = opt['train'].get('warmup_steps', 5000)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        diffusion.optG,
        lr_lambda=get_lr_lambda(warmup_steps, n_iter, eta_min=1e-6, base_lr=base_lr),
        last_epoch=current_step - 1 if current_step > 0 else -1)

    if is_main_process():
        n_gpus = dist.get_world_size() if dist.is_initialized() else 1
        logger.info('Training config: n_iter={}, batch_size={}, lr={}, '
                     'warmup={}, GPUs={}'.format(
                         n_iter, opt['datasets']['train']['batch_size'],
                         base_lr, warmup_steps, n_gpus))

    if opt['path']['resume_state']:
        logger.info('Resuming from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']],
        schedule_phase=opt['phase'])

    val_step = 0
    while current_step < n_iter:
        current_epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(current_epoch)

        for train_data in train_loader:
            current_step += 1
            if current_step > n_iter:
                break

            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            scheduler.step()
            diffusion.update_ema(current_step)
            lr = scheduler.get_last_lr()[0]

            # log (rank 0 only)
            if is_main_process() and current_step % opt['train']['print_freq'] == 0:
                log_training_step(
                    diffusion, current_epoch, current_step, lr,
                    tb_logger, wandb_logger)

            # validation (rank 0 only)
            if is_main_process() and current_step % opt['train']['val_freq'] == 0:
                result_path = '{}/{}'.format(
                    opt['path']['results'], current_epoch)
                val_step = run_validation_step(
                    diffusion, eval_loader, norm_stats, model_label,
                    current_epoch, current_step, result_path, val_step,
                    tb_logger, wandb_logger)

            # save checkpoint (rank 0 only)
            if is_main_process() and current_step % opt['train']['save_checkpoint_freq'] == 0:
                save_checkpoint(
                    diffusion, current_epoch, current_step,
                    wandb_logger, opt.get('log_wandb_ckpt', False))

            # upload checkpoint to wandb every 10k steps (overwrites "latest")
            if is_main_process() and wandb_logger and current_step % 10000 == 0:
                wandb_logger.log_checkpoint(
                    current_epoch, current_step, aliases=["latest"])

    # Save final checkpoint and upload to wandb
    if is_main_process():
        if current_step % opt['train']['save_checkpoint_freq'] != 0:
            diffusion.save_network(current_epoch, current_step)
        if wandb_logger:
            wandb_logger.log_checkpoint(
                current_epoch, current_step, aliases=["latest", "final"])

    logger.info('End of training.')


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='LST Super-Resolution with Diffusion Models')
    parser.add_argument('-c', '--config', type=str,
                        default='config_sr3.yaml',
                        help='YAML/JSON config file')
    parser.add_argument('-p', '--phase', type=str, choices=['train'],
                        help='train', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- DDP setup ----
    local_rank = setup_distributed()

    opt = Logger.parse(args)
    if dist.is_initialized():
        opt['distributed'] = True
        opt['local_rank'] = local_rank
    opt = Logger.dict_to_nonedict(opt)

    # Reproducibility
    seed = opt.get('seed', 42)
    deterministic = opt.get('deterministic', False)
    set_seed(seed, deterministic=deterministic)

    # Detect model variant
    model_type, needs_hls, model_label = get_model_info(opt)

    # logging (only rank 0)
    torch.backends.cudnn.enabled = True
    if not deterministic:
        torch.backends.cudnn.benchmark = True

    if is_main_process():
        Logger.setup_logger(None, opt['path']['log'],
                            'train', level=logging.INFO, screen=True)
        Logger.setup_logger('val', opt['path']['log'], 'val',
                            level=logging.INFO)
    logger = logging.getLogger('base')
    if is_main_process():
        logger.info(Logger.dict2str(opt))
        save_experiment_metadata(opt, args)
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger']) if is_main_process() else None

    # wandb (only rank 0)
    wandb_logger = None
    if opt['enable_wandb'] and is_main_process():
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        wandb.define_metric("train/*", step_metric="step")

    # ---- normalization stats ----
    norm_stats = load_norm_stats(opt)

    # ---- dataset ----
    logger.info('Creating {} datasets...'.format(model_label))
    train_loader, eval_loader, train_sampler = build_datasets(
        opt, norm_stats, needs_hls, args.phase)
    logger.info('Dataset ready.')

    # ---- model ----
    diffusion = Model.create_model(opt)
    logger.info('Model ready ({}).'.format(model_label))

    # ---- train ----
    if opt['phase'] == 'train':
        train_loop(
            diffusion, train_loader, eval_loader, train_sampler,
            opt, norm_stats, model_label,
            tb_logger=tb_logger, wandb_logger=wandb_logger)

    cleanup_distributed()


if __name__ == "__main__":
    main()
