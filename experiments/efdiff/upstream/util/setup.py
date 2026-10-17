"""Shared initialization logic for train.py and infer.py."""
import os
import logging

from util.normalization import NormStats

logger = logging.getLogger('base')


def load_norm_stats(opt):
    """Load normalization stats from config path, or return defaults.

    Raises FileNotFoundError during training if stats file is missing,
    since training with default stats (mean=0, std=1) produces bad results.
    """
    stats_path = opt['datasets'].get(
        'normalization_stats',
        opt['datasets'].get('residual_stats', 'data/normalization_stats.json'))
    if os.path.exists(stats_path):
        return NormStats.from_json(stats_path)

    msg = (f'Normalization stats not found at {stats_path}. '
           f'Run: python scripts/compute_residual_stats.py')
    if opt.get('phase', 'train') == 'train':
        raise FileNotFoundError(msg)
    logger.warning(f'{msg} Using defaults (mean=0, std=1).')
    return NormStats()
