"""BaseModel: device management, EMA, checkpoint helpers.

Shared base class for all model orchestrators (currently only DDPM).
"""
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger('base')


def unwrap(net):
    """Unwrap DDP or DataParallel to get the raw module."""
    if isinstance(net, (DDP, nn.DataParallel)):
        return net.module
    return net


class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0
        self.ema_state = None
        self._backup_state = None

    @abstractmethod
    def feed_data(self, data):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    @abstractmethod
    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for i, item in enumerate(x):
                if item is not None:
                    x[i] = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    def get_network_description(self, network):
        """Get the string and total parameters of the network."""
        network = unwrap(network)
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    # ---- EMA (Exponential Moving Average) ----

    def _get_bare_network(self):
        return unwrap(self.netG)

    def init_ema(self, decay=0.9999, start_step=5000, update_every=1):
        self.ema_decay = decay
        self.ema_start = start_step
        self.ema_every = update_every
        self.ema_state = {
            k: v.clone()
            for k, v in self._get_bare_network().state_dict().items()
        }
        logger.info('EMA initialized (decay={}, start_step={}, keys={})'.format(
            decay, start_step, len(self.ema_state)))

    def update_ema(self, step):
        if self.ema_state is None or step < self.ema_start or step % self.ema_every != 0:
            return
        with torch.no_grad():
            for k, v in self._get_bare_network().state_dict().items():
                if k not in self.ema_state:
                    continue
                if not v.is_floating_point():
                    self.ema_state[k].copy_(v)
                    continue
                self.ema_state[k].mul_(self.ema_decay).add_(
                    v, alpha=1 - self.ema_decay)

    def apply_ema(self):
        if self.ema_state is None:
            return
        net = self._get_bare_network()
        full_state = net.state_dict()
        self._backup_state = {k: v.clone() for k, v in full_state.items()
                              if k in self.ema_state}
        # Merge EMA keys into full state dict
        merged = dict(full_state)
        merged.update(self.ema_state)
        net.load_state_dict(merged)

    def restore_from_ema(self):
        if self._backup_state is None:
            return
        net = self._get_bare_network()
        full_state = net.state_dict()
        full_state.update(self._backup_state)
        net.load_state_dict(full_state)
        self._backup_state = None
