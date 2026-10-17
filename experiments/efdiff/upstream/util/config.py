"""Config validation for LST super-resolution.

Validates that required keys exist and have correct types after YAML loading,
replacing the NoneDict approach that silently swallowed missing-key errors.
"""


def _get_nested(d, dotpath):
    """Retrieve a value from a nested dict using dot-separated path."""
    keys = dotpath.split('.')
    val = d
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return None, False
        val = val[k]
    return val, True


# (dotpath, expected_type) — only the keys that MUST exist
_COMMON_REQUIRED = [
    ('model.which_model_G', str),
    ('model.diffusion.image_size', int),
    ('model.diffusion.channels', int),
    ('model.unet.in_channel', int),
    ('model.unet.out_channel', int),
    ('model.unet.inner_channel', int),
    ('model.unet.channel_multiplier', list),
    ('model.unet.attn_res', list),
    ('model.unet.res_blocks', int),
    ('model.diffusion.conditional', bool),
]

_TRAIN_ONLY_REQUIRED = [
    ('model.beta_schedule.train.n_timestep', int),
    ('model.beta_schedule.train.schedule', str),
    ('datasets.train.dataroot', str),
    ('datasets.train.batch_size', int),
    ('train.n_iter', int),
    ('train.optimizer.lr', float),
    ('train.val_freq', int),
    ('train.save_checkpoint_freq', int),
    ('train.print_freq', int),
    ('path.log', str),
    ('path.checkpoint', str),
]

_TEST_ONLY_REQUIRED = [
    ('model.beta_schedule.test.n_timestep', int),
    ('model.beta_schedule.test.schedule', str),
]


def validate_config(opt, phase='train'):
    """Validate config dict has all required keys with correct types.

    Raises ValueError with a summary of all missing/wrong-type keys.
    """
    rules = list(_COMMON_REQUIRED)
    if phase == 'train':
        rules += _TRAIN_ONLY_REQUIRED
    else:
        rules += _TEST_ONLY_REQUIRED

    errors = []
    for dotpath, expected_type in rules:
        val, found = _get_nested(opt, dotpath)
        if not found:
            errors.append(f'  Missing: {dotpath}')
        elif not isinstance(val, expected_type):
            errors.append(
                f'  {dotpath}: expected {expected_type.__name__}, '
                f'got {type(val).__name__} ({val!r})')
    if errors:
        raise ValueError(
            'Config validation failed:\n' + '\n'.join(errors))
