"""Backward-compatible re-exports from split modules.

The original monolithic model.py has been split into:
  - model/base.py:    BaseModel (device, EMA, checkpoint helpers)
  - model/ddpm.py:    DDPM (training/inference orchestration)
  - model/factory.py: define_G, init_weights (network construction)
"""
# Re-export public API so existing imports still work
from .base import BaseModel, unwrap  # noqa: F401
from .ddpm import DDPM  # noqa: F401
from .factory import define_G, init_weights  # noqa: F401
