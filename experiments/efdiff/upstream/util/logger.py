import os
import os.path as osp
import logging
from collections import OrderedDict
import json
import yaml
import string
from datetime import datetime

from util.config import validate_config


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def _expand_path_str(value):
    if not isinstance(value, str):
        return value
    env = dict(os.environ)
    home = env.get('HOME', os.path.expanduser('~'))
    env.setdefault('HOME', home)
    env.setdefault('SCRATCH', os.path.join(home, 'scratch'))
    env.setdefault('ARCHIVE', os.path.join(home, 'archive'))
    return os.path.expanduser(string.Template(value).safe_substitute(env))


def _expand_env_vars(obj):
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return _expand_path_str(obj)
    return obj


def parse(args):
    phase = args.phase
    opt_path = args.config
    gpu_ids = getattr(args, 'gpu_ids', None)
    # Load config (YAML or JSON)
    if opt_path.endswith(('.yaml', '.yml')):
        with open(opt_path, 'r') as f:
            opt = yaml.safe_load(f)
    else:
        json_str = ''
        with open(opt_path, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
        opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt = _expand_env_vars(opt)

    # set log directory
    debug = getattr(args, 'debug', False)
    if debug:
        opt['name'] = 'debug_{}'.format(opt['name'])

    resume_state = opt['path'].get('resume_state')
    if resume_state:
        # Reuse existing experiment directory when resuming
        # resume_state is like .../experiments/<name>/checkpoint/I85000_E48
        experiments_root = os.path.dirname(os.path.dirname(resume_state))
    else:
        scratch_root = os.getenv('SCRATCH')
        if scratch_root:
            experiments_base = os.path.join(
                scratch_root, 'lst-super-resolution', 'experiments')
        else:
            experiments_base = 'experiments'
        experiments_root = os.path.join(
            experiments_base, '{}_{}'.format(opt['name'], get_timestamp()))
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    opt['phase'] = phase

    # export CUDA_VISIBLE_DEVICES
    if gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
        gpu_list = gpu_ids
    else:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    # Only set CUDA_VISIBLE_DEVICES for non-DDP (torchrun manages it)
    if not os.environ.get('LOCAL_RANK'):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    opt['distributed'] = False  # overridden by train.py if launched with torchrun

    # debug overrides
    if 'debug' in opt['name']:
        opt['train']['val_freq'] = 2
        opt['train']['print_freq'] = 2
        opt['train']['save_checkpoint_freq'] = 3
        opt['datasets']['train']['batch_size'] = 2
        opt['model']['beta_schedule']['train']['n_timestep'] = 10
        opt['datasets']['train']['data_len'] = 6
        opt['datasets']['val']['data_len'] = 3

    # Optional flags (safe for callers that don't define them)
    opt['enable_wandb'] = getattr(args, 'enable_wandb', False)
    opt['log_wandb_ckpt'] = getattr(args, 'log_wandb_ckpt', False)
    opt['log_eval'] = getattr(args, 'log_eval', False)

    # Validate required keys early (replaces NoneDict silent failures)
    validate_config(opt, phase=phase)

    return opt


def dict_to_nonedict(opt):
    """Identity function (NoneDict removed). Kept for call-site compatibility."""
    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, '{}.log'.format(phase))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)


class WandbLogger:
    """Log using Weights and Biases."""
    def __init__(self, opt):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        self._wandb = wandb
        self._opt = opt

        if self._wandb.run is None:
            run_name, run_tags = self._make_run_name(opt)
            init_kwargs = dict(
                project=opt['wandb']['project'],
                name=run_name,
                tags=run_tags,
                config=opt,
                dir=opt['path']['experiments_root'],
            )
            # Resume existing wandb run when resuming training
            if opt['path'].get('resume_state'):
                run_id = self._find_wandb_run_id(
                    opt['path']['experiments_root'])
                if run_id:
                    init_kwargs['id'] = run_id
                    init_kwargs['resume'] = 'must'
            self._wandb.init(**init_kwargs)

    @staticmethod
    def _find_wandb_run_id(experiments_root):
        """Find wandb run ID from an existing experiment directory."""
        import glob
        wandb_dir = os.path.join(experiments_root, 'wandb')
        # wandb stores runs as run-YYYYMMDD_HHMMSS-<run_id>
        run_dirs = sorted(glob.glob(os.path.join(wandb_dir, 'run-*')))
        if run_dirs:
            # Use the latest run directory
            last_run = os.path.basename(run_dirs[-1])
            # Format: run-YYYYMMDD_HHMMSS-<run_id>
            parts = last_run.split('-')
            if len(parts) >= 3:
                return parts[-1]
        return None

    @staticmethod
    def _make_run_name(opt):
        """Generate standardized run name and tags from config.

        Pattern: {model}_{conditioning}_seed{seed}_bs{bs}
        Examples: sr3_seed42_bs64, sr3_prithvi_seed42_bs64
        """
        model_type = opt.get('model', {}).get('which_model_G', 'unknown')
        seed = opt.get('seed', 42)
        per_gpu_bs = opt.get('datasets', {}).get('train', {}).get('batch_size', '?')
        n_iter = opt.get('train', {}).get('n_iter', '?')

        # Effective batch size = per-GPU bs × number of GPUs (DDP)
        n_gpus = len(opt.get('gpu_ids', [0]))
        if opt.get('distributed', False):
            import torch.distributed as dist
            if dist.is_initialized():
                n_gpus = dist.get_world_size()
        bs = per_gpu_bs * n_gpus if isinstance(per_gpu_bs, int) else per_gpu_bs

        # Build name parts
        parts = [model_type, f'seed{seed}', f'bs{bs}']
        run_name = '_'.join(parts)

        # Build tags
        tags = [model_type, f'{n_iter // 1000}k' if isinstance(n_iter, int) else str(n_iter)]

        return run_name, tags

    def log_metrics(self, metrics, commit=True):
        self._wandb.log(metrics, commit=commit)

    def log_image(self, key_name, image_array):
        self._wandb.log({key_name: self._wandb.Image(image_array)})

    def log_images(self, key_name, list_images):
        self._wandb.log({key_name: [self._wandb.Image(img) for img in list_images]})

    def log_checkpoint(self, current_epoch, current_step, aliases=None):
        if aliases is None:
            aliases = ["latest"]
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        ckpt_dir = self._opt['path']['checkpoint']
        prefix = 'I{}_E{}'.format(current_step, current_epoch)
        for suffix in ['_gen.pth', '_opt.pth', '_ema.pth']:
            path = os.path.join(ckpt_dir, prefix + suffix)
            if os.path.exists(path):
                model_artifact.add_file(path)

        self._wandb.log_artifact(model_artifact, aliases=aliases)

    def log_eval_data(self, fake_img, sr_img, hr_img, psnr=None, ssim=None):
        if psnr is not None and ssim is not None:
            self.eval_table.add_data(
                self._wandb.Image(fake_img),
                self._wandb.Image(sr_img),
                self._wandb.Image(hr_img),
                psnr,
                ssim
            )
        else:
            self.infer_table.add_data(
                self._wandb.Image(fake_img),
                self._wandb.Image(sr_img),
                self._wandb.Image(hr_img)
            )

    def log_eval_table(self, commit=False):
        if self.eval_table:
            self._wandb.log({'eval_data': self.eval_table}, commit=commit)
        elif self.infer_table:
            self._wandb.log({'infer_data': self.infer_table}, commit=commit)
