"""Opt-in W&B scalar monitoring. Data, source and checkpoints stay on scratch."""
from contextlib import contextmanager
import math
from pathlib import Path
import time

_run = None
_output = None
_last_progress = 0.


def public_config(config):
    """Allowlist only scientific hyperparameters, never paths/IDs/credentials."""
    return {'model': config['model']['kind'], 'representation': config['representation']['kind'],
        'seed': config['seed'], 'micro_batch': config['train']['micro_batch'],
        'effective_batch': config['train']['effective_batch'], 'epochs': config['train']['epochs'],
        'learning_rate': config['train']['lr'],
        'inverse_gradient_policy': config['representation'].get('inverse_gradient_policy','native'),
        'representation_loss_weight': config['loss']['representation']}


def log(values):
    if _run is None:
        return
    try:
        _run.log(values)
    except Exception as exc:
        from .runtime import append_event
        append_event(_output / 'tracking_errors.jsonl', {'error_type': type(exc).__name__})


def phase(name):
    if _run is not None:
        _run.summary['phase'] = name
        log({'progress/phase': name})


def evaluation_progress(path, rows):
    global _last_progress
    if time.monotonic() - _last_progress >= 30:
        log({'evaluation/condition': Path(path).name, 'evaluation/processed_rows': rows})
        _last_progress = time.monotonic()


def evaluation_metrics(phase_name, result):
    prefix = f"{phase_name}/{result['split']}/{result['condition']}"
    values = {}
    def visit(node, key):
        for name, value in node.items():
            if isinstance(value, dict):
                visit(value, key + '/' + name)
            elif isinstance(value, (int, float)) and math.isfinite(value):
                values[key + '/' + name] = value
    for name in ('raw', 'projected', 'observation_mean_fill_baseline'):
        visit(result[name], prefix + '/' + name)
    log(values)


@contextmanager
def session(config, output):
    global _run, _output
    options = config.get('tracking', {})
    if not options.get('enabled', False):
        yield
        return
    import wandb
    from .config import write_json
    _output = Path(output)
    _output.mkdir(parents=True, exist_ok=True)
    run = wandb.init(entity=options['entity'], project=options['project'],
        id=options['run_id'], name=options['name'], group=options['group'],
        job_type='stage1', tags=['P1', 'micro16', 'effective64', 'val-only'],
        resume='allow', mode='online', dir=str(_output), save_code=False,
        config=public_config(config), settings=wandb.Settings(disable_git=True, disable_code=True,
            x_disable_meta=True, console='off', init_timeout=90))
    _run = run
    write_json(_output / 'wandb_run.json', {'url':run.url, 'id':run.id,
        'entity':run.entity, 'project':run.project, 'mode':'online',
        'uploads':'scalar metrics and allowlisted hyperparameters only'})
    run.define_metric('train/step')
    run.define_metric('train/*', step_metric='train/step')
    run.define_metric('memory/*', step_metric='train/step')
    try:
        yield
        phase('complete')
    except BaseException:
        phase('failed_or_interrupted')
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)
    finally:
        _run = None
        _output = None
