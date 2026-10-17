"""Normalization statistics for LST super-resolution.

Encapsulates z-score normalization/denormalization for both LST condition
and residual, replacing the previous global-variable approach.

Usage:
    stats = NormStats.from_json('normalization_stats.json')
    dataset = LSTDataset(data_dir, norm_stats=stats)
"""
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger('base')

# Number of standard deviations for z-score clipping (maps ~99.7% of data to [-1, 1])
SIGMA_CLIP = 3.0


@dataclass
class NormStats:
    """Z-score normalization statistics for LST condition and residual."""
    lst_mean: float = 0.0
    lst_std: float = 1.0
    residual_mean: float = 0.0
    residual_std: float = 1.0

    # --- LST condition ---

    def normalize_lst(self, x):
        """Normalize LST condition to ~[-1, 1] via z-score (3-sigma clipping)."""
        return (x - self.lst_mean) / (SIGMA_CLIP * self.lst_std)

    def denormalize_lst(self, x):
        """Denormalize LST condition from z-score back to Celsius."""
        return x * (SIGMA_CLIP * self.lst_std) + self.lst_mean

    # --- Residual ---

    def normalize_residual(self, x):
        """Normalize residual to ~[-1, 1] via z-score (3-sigma clipping)."""
        return (x - self.residual_mean) / (SIGMA_CLIP * self.residual_std)

    def denormalize_residual(self, x):
        """Denormalize residual from z-score back to Celsius."""
        return x * (SIGMA_CLIP * self.residual_std) + self.residual_mean

    # --- I/O ---

    @classmethod
    def from_json(cls, stats_path):
        """Load from normalization_stats.json.

        Expected format (from scripts/compute_residual_stats.py):
            {"residual": {"mean": ..., "std": ...},
             "lst":      {"mean": ..., "std": ...}}
        """
        with open(stats_path, 'r') as f:
            raw = json.load(f)

        stats = cls(
            lst_mean=raw['lst']['mean'],
            lst_std=raw['lst']['std'],
            residual_mean=raw['residual']['mean'],
            residual_std=raw['residual']['std'],
        )

        logger.info(
            f"Normalization stats - "
            f"LST: mean={stats.lst_mean:.4f}, std={stats.lst_std:.4f} | "
            f"Residual: mean={stats.residual_mean:.4f}, "
            f"std={stats.residual_std:.4f}"
        )
        return stats
