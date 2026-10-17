"""Extract observed training minima; never invent missing checkpoint weights."""
from collections import deque
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'outputs/torch_experiments_20261013'
BUNDLE = ROOT / 'outputs/torch_models_20261013'
OUT = ROOT / 'outputs/training_lowpoints_20261014'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def analyze(model, run):
    config = json.loads((BUNDLE / model['config']).read_text(encoding='utf-8'))
    weights = {k: v for k, v in config['loss'].items() if isinstance(v, (int, float))}
    folder = RESULTS / 'snapshot/output' / model['stage'] / 'runs' / model['name'] / 'training'
    rows, duplicates = {}, 0
    logs = sorted(folder.glob('train_*.jsonl'))
    if not logs:
        raise ValueError(f'No training log: {model["name"]}')
    for path in logs:
        with path.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                step = event['step']
                if step > config['train']['updates']:
                    raise ValueError('Log exceeds completed budget')
                event['source'] = str(path.relative_to(ROOT)).replace('\\', '/') + f':{line_no}'
                event['total_loss_reconstructed'] = sum(weights[k] * event['loss'][k] for k in weights)
                if not math.isfinite(event['total_loss_reconstructed']):
                    raise ValueError('Nonfinite logged objective')
                if step in rows:
                    duplicates += 1
                    if event['utc'] < rows[step]['utc']:
                        continue
                rows[step] = event
    if sorted(rows) != list(range(1, config['train']['updates'] + 1)):
        raise ValueError(f'Missing training steps: {model["name"]}')
    ordered = [rows[s] for s in sorted(rows)]
    n = config['data']['counts']['train']
    batch = config['train']['effective_batch']
    steps_per_epoch = math.ceil(n / batch)
    for event in ordered:
        epoch_index, offset = divmod(event['step'] - 1, steps_per_epoch)
        expected_records = epoch_index * n + min((offset + 1) * batch, n)
        actual = min(batch, n - offset * batch)
        if event['records'] != expected_records or event['actual_batch'] != actual:
            raise ValueError(f'Bad budget cursor: {model["name"]}/{event["step"]}')

    def point(event):
        return {'logged_update': event['step'], 'weights_update_before_this_loss': event['step'] - 1,
                'epoch_of_batch': (event['step'] - 1) // steps_per_epoch + 1,
                'epoch_progress_before_batch': (event['records'] - event['actual_batch']) / n,
                'actual_batch': event['actual_batch'], 'utc': event['utc'],
                'total_loss_reconstructed': event['total_loss_reconstructed'],
                'loss_pixel': event['loss']['pixel'], 'loss_terms': event['loss'],
                'source': event['source']}

    minimum = min(ordered, key=lambda x: (x['total_loss_reconstructed'], x['step']))
    pixel = min(ordered, key=lambda x: (x['loss']['pixel'], x['step']))
    window, total, samples, best_window = deque(), 0., 0, None
    for event in ordered:
        window.append(event)
        total += event['total_loss_reconstructed'] * event['actual_batch']
        samples += event['actual_batch']
        if len(window) > 100:
            previous = window.popleft()
            total -= previous['total_loss_reconstructed'] * previous['actual_batch']
            samples -= previous['actual_batch']
        if len(window) == 100 and (best_window is None or total / samples < best_window['sample_weighted_total_loss']):
            best_window = {'start_update': window[0]['step'], 'end_update': event['step'],
                           'record_presentations': samples, 'sample_weighted_total_loss': total / samples,
                           'checkpoint_claim': 'Descriptive 100-update average, not one model state'}
    cells = {cell['condition']: cell for cell in run['cells']}
    return {'model': model['name'], 'kind': model['kind'], 'stage': model['stage'],
        'logged_updates': len(ordered), 'duplicate_updates_latest_utc_used': duplicates,
        'loss_weights': weights, 'minimum_total_loss': point(minimum), 'minimum_pixel_loss': point(pixel),
        'minimum_100_update_average': best_window,
        'per_epoch_minimum_total_loss': [point(min(ordered[(epoch-1)*steps_per_epoch:epoch*steps_per_epoch],
            key=lambda x: (x['total_loss_reconstructed'], x['step']))) for epoch in range(1, 6)],
        'final_training_batch': point(ordered[-1]),
        'minima_checkpoint_availability': 'No corresponding checkpoint in downloaded inventory; server check pending',
        'available_checkpoint': {'epoch': 5, 'update': config['train']['updates'],
            'original_checkpoint': model['original_checkpoint'],
            'original_checkpoint_sha256': model['original_checkpoint_sha256'],
            'local_inference_weights': str((BUNDLE / model['weights']).relative_to(ROOT)).replace('\\', '/')},
        'existing_final_validation': cells}


def main():
    manifest = json.loads((BUNDLE / 'MODEL_MANIFEST.json').read_text(encoding='utf-8'))
    index = json.loads((RESULTS / 'results_index.json').read_text(encoding='utf-8'))
    runs = {row['model']: row for row in index['runs']}
    models = [analyze(row, runs[row['name']]) for row in manifest['models']]
    inventory_path = OUT / 'server_checkpoint_inventory.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8')) if inventory_path.exists() else None
    if inventory:
        remote = {r['model']: r for r in inventory['models']}
        assert set(remote) == {m['model'] for m in models}
        for model in models:
            row = remote[model['model']]
            expected = model['available_checkpoint']['original_checkpoint'].removeprefix(inventory['remote_root'] + '/')
            assert row['checkpoints'] == [expected], row
            assert row['completion']['epochs'] == 5
            assert row['completion']['checkpoint_sha256'] == model['available_checkpoint']['original_checkpoint_sha256']
            model['minima_checkpoint_availability'] = 'Unavailable: server inventory contains only final-epoch last.pt'
        assert not inventory['additional_checkpoint_paths']
    condition = next(c for c in models[0]['existing_final_validation'] if c.startswith('pixel10_'))
    rank = sorted(models, key=lambda m: m['existing_final_validation'][condition]['raw_rmse_C'])
    best_by_kind = {}
    for row in rank:
        best_by_kind.setdefault(row['kind'], {'model': row['model'],
            'final_pixel10_raw_rmse_C': row['existing_final_validation'][condition]['raw_rmse_C'],
            'checkpoint': row['available_checkpoint'], 'status': 'Existing final-epoch result; not lowpoint validation'})
    report = {'scope': '39 models; training-log minima plus existing final-epoch validation',
        'new_validation_run': False,
        'server_inventory_status': 'verified_only_final_epoch_checkpoints' if inventory else 'pending_authentication',
        'server_checked_utc': inventory['checked_utc'] if inventory else None,
        'total_unique_logged_updates': sum(m['logged_updates'] for m in models),
        'method': ['Total loss reconstructed using each original config; quantizer diagnostics are not added twice.',
                   'Each logged loss is computed before that optimizer update: exact weight update = logged step - 1.',
                   'Training minima compare changing minibatches and are not validation minima.',
                   'Loss functions differ by model family; total training loss is not a cross-family ranking metric.',
                   'Final validation uses all 11684 val records and two fixed views; no test selection.',
                   'No checkpoint can be recovered from scalar loss values alone.'],
        'models': models, 'existing_final_best_by_kind_pixel10': best_by_kind,
        'existing_final_best_by_condition': {c: min(models, key=lambda m: m['existing_final_validation'][c]['raw_rmse_C'])['model']
            for c in models[0]['existing_final_validation']}}
    write(OUT / 'training_lowpoints.json', report)
    lines = ['39 个模型的训练低点与可用权重核对', '',
        ('状态：服务器已确认仅保留第5轮 last.pt；39个模型的低点权重均不可用，未执行低点验证。' if inventory else
         '状态：已提取完整训练日志；尚未对低点权重执行新验证；服务器库存核对等待认证。'),
        'total 为原配置加权目标；step 为日志中的更新编号，对应前向权重处于 step-1。',
        '最后一列是已完成的第5轮权重验证结果，不能当成低点权重结果。', '',
        'model\tminimum_total_loss\tlogged_step\tepoch\tminimum_pixel_loss\tpixel_step\tfinal_val_pixel10_RMSE_C']
    for row in models:
        low, pixel = row['minimum_total_loss'], row['minimum_pixel_loss']
        lines.append(f"{row['model']}\t{low['total_loss_reconstructed']:.9g}\t{low['logged_update']}\t{low['epoch_of_batch']}\t"
                     f"{pixel['loss_pixel']:.9g}\t{pixel['logged_update']}\t{row['existing_final_validation'][condition]['raw_rmse_C']:.9g}")
    (OUT / 'ALL_39_LOWPOINTS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'models': len(models), 'updates': report['total_unique_logged_updates'],
                      'duplicate_updates': {m['model']: m['duplicate_updates_latest_utc_used'] for m in models if m['duplicate_updates_latest_utc_used']},
                      'best_by_kind_existing_final_only': {k: (v['model'], v['final_pixel10_raw_rmse_C']) for k,v in best_by_kind.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
