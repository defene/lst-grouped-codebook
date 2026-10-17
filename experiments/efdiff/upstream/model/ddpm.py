"""DDPM: training/inference orchestration, save/load.

Wraps a diffusion model (GaussianDiffusion or ResShiftDiffusion) with
training loop helpers, EMA management, checkpoint I/O, and DDP support.
"""
import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .base import BaseModel, unwrap
from .factory import define_G

logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(define_G(opt))
        self.schedule_phase = None

        # Detect frozen encoder (Prithvi variants)
        bare = self._get_bare_network()
        self.has_frozen_params = (
            hasattr(bare, 'prithvi_encoder') and bare.prithvi_encoder is not None)

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()

            if self.has_frozen_params:
                optim_params = [p for p in self.netG.parameters() if p.requires_grad]
                n_trainable = sum(p.numel() for p in optim_params)
                n_total = sum(p.numel() for p in self.netG.parameters())
                logger.info('Trainable params: {:,d} / {:,d} total'.format(
                    n_trainable, n_total))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()

            # Cache trainable params for grad clipping
            self._trainable_params = optim_params

        self.load_network()
        self.print_network()

        # Wrap with DDP if distributed
        if opt.get('distributed') and opt['phase'] == 'train':
            local_rank = opt.get('local_rank', 0)
            self.netG = DDP(
                self.netG,
                device_ids=[local_rank],
                find_unused_parameters=self.has_frozen_params,
            )
            logger.info(f'Wrapped model with DDP (local_rank={local_rank})')

        # Initialize EMA after network is fully loaded (and wrapped)
        if self.opt['phase'] == 'train':
            ema_cfg = opt['train'].get('ema_scheduler', {})
            self.init_ema(
                decay=ema_cfg.get('ema_decay', 0.9999),
                start_step=ema_cfg.get('step_start_ema', 5000),
                update_every=ema_cfg.get('update_ema_every', 1),
            )
            # Override with saved EMA state if resuming
            load_path = self.opt['path']['resume_state']
            if load_path is not None:
                ema_path = '{}_ema.pth'.format(load_path)
                if os.path.exists(ema_path):
                    self.ema_state = torch.load(
                        ema_path, map_location=self.device)
                    logger.info('Loaded EMA state from checkpoint')

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            l_pix = self.netG(self.data)
        l_pix.backward()
        grad_norm = nn.utils.clip_grad_norm_(self._trainable_params, max_norm=1.0)
        self.optG.step()

        self.log_dict['train/loss'] = l_pix.item()
        self.log_dict['train/grad_norm'] = grad_norm.item()

    def test(self, continuous=False, verbose=True, ddim_steps=None):
        self.apply_ema()
        self.netG.eval()
        bare = self._get_bare_network()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            self.SR = bare.super_resolution(
                self.data, continuous, verbose=verbose,
                ddim_steps=ddim_steps)
        self.netG.train()
        self.restore_from_ema()

    def sample(self, batch_size=1, continuous=False):
        self.apply_ema()
        self.netG.eval()
        bare = self._get_bare_network()
        with torch.no_grad():
            self.SR = bare.sample(batch_size, continuous)
        self.netG.train()
        self.restore_from_ema()

    def set_loss(self):
        self._get_bare_network().set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            self._get_bare_network().set_new_noise_schedule(
                schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        net_struc_str = '{}'.format(self.netG.__class__.__name__)
        bare = unwrap(self.netG)
        if bare is not self.netG:
            net_struc_str = '{} - {}'.format(
                self.netG.__class__.__name__, bare.__class__.__name__)
        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(
                net_struc_str, n))

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'],
            'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'],
            'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self._get_bare_network()
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)
        # ema
        if self.ema_state is not None:
            ema_path = os.path.join(
                self.opt['path']['checkpoint'],
                'I{}_E{}_ema.pth'.format(iter_step, epoch))
            torch.save({k: v.cpu() for k, v in self.ema_state.items()},
                       ema_path)

        logger.info('Saved checkpoint [{:s}]'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self._get_bare_network()
            strict = not self.has_frozen_params
            network.load_state_dict(torch.load(gen_path), strict=strict)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
