import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from eo_data.core import json_write, sha256_file
from .config import load_config, validate
from .data import PairedDataset
from .engine import (train, evaluate_checkpoint, read_checkpoint, seed_everything,
                     device_for, environment)
from .mask_bank import build_bank
from .models import CrossViewModel
from .tokens import pack, unpack


def main():
    parser = argparse.ArgumentParser(description='Cross-View Discrete Tokenization for Noise-Robust Earth Observation')
    sub = parser.add_subparsers(dest='command', required=True)
    bank = sub.add_parser('bank', help='Build a bounded, split-local QA geometry bank')
    bank.add_argument('--root', default='.')
    bank.add_argument('--output', default='experiments/paper1/mask_banks')
    bank.add_argument('--split', choices=['train', 'val'], required=True)
    bank.add_argument('--scan', type=int, default=4096)
    bank.add_argument('--shapes', type=int, default=256)
    bank.add_argument('--seed', type=int, default=20260910)
    inspect = sub.add_parser('inspect', help='Inspect paired membership and an actual model input')
    inspect.add_argument('--config', required=True)
    inspect.add_argument('--split', choices=['train', 'val'], default='train')
    for name in ('train', 'smoke'):
        command = sub.add_parser(name)
        command.add_argument('--config', required=True)
        command.add_argument('--output', required=True)
        command.add_argument('--resume')
        command.add_argument('--stop-after', type=int)
        if name == 'smoke':
            command.add_argument('--steps', type=int, default=4)
    evaluate = sub.add_parser('evaluate', help='Evaluate a frozen checkpoint on existing val')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--coverage', type=float)
    evaluate.add_argument('--limit', type=int)
    evaluate.add_argument('--workers', type=int)
    evaluate.add_argument('--noise-kind', choices=['none', 'bank_missing', 'procedural_missing', 'gaussian', 'stripes'])
    export = sub.add_parser('export-tokens', help='Write reversible discrete codes and verify decode equality')
    export.add_argument('--checkpoint', required=True)
    export.add_argument('--output', required=True)
    export.add_argument('--index', type=int, default=0)
    export.add_argument('--modality', choices=['hls', 'lst'], required=True)
    args = parser.parse_args()
    if args.command == 'bank':
        result = build_bank(args.root, args.output, args.split, args.scan, args.shapes, args.seed)
    elif args.command == 'evaluate':
        result = evaluate_checkpoint(args.checkpoint, args.output, args.coverage, args.limit, args.workers, args.noise_kind)
    elif args.command == 'export-tokens':
        result = export_tokens(args.checkpoint, args.output, args.modality, args.index)
    else:
        config = load_config(args.config)
        if args.command == 'inspect':
            ds = PairedDataset(config, args.split, coverage=config['eval']['coverage'] if args.split == 'val' else None)
            sample = ds[0]
            result = {'dataset': 'prithvi', 'split': args.split, 'paired_samples': len(ds),
                      'fingerprint': ds.fingerprint, 'sample_id': sample['sample_id'],
                      'modalities': {m: {k: list(sample[m][k].shape) for k in ('image', 'target', 'input_mask', 'reference_mask', 'corruption_mask')}
                                     for m in ds.modalities}, 'environment': environment(device_for(config))}
            ds.close()
        else:
            if args.command == 'smoke':
                config['name'] = 'smoke_'+config['name']
                config['data'].update(train_limit=16, val_limit=8)
                config['train'].update(steps=args.steps, batch_size=2, accumulation=1, warmup=1,
                                       eval_every=args.steps, save_every=args.steps, log_every=1)
                config['eval'].update(batch_size=2, bootstrap_replicates=50)
                validate(config)
            result = train(config, args.output, args.resume, args.stop_after)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


@torch.no_grad()
def export_tokens(checkpoint_path, output, modality, index):
    checkpoint = read_checkpoint(checkpoint_path)
    config = copy.deepcopy(checkpoint['config'])
    seed_everything(config['seed'])
    ds = PairedDataset(config, 'val', coverage=config['eval']['coverage'])
    sample = ds[index]
    model = CrossViewModel(config).eval()
    model.load_state_dict(checkpoint['model'])
    if modality not in model.codecs:
        raise ValueError('Checkpoint does not contain requested modality')
    codec = model.codecs[modality]
    image = torch.from_numpy(sample[modality]['image'][0:1])
    mask = torch.from_numpy(sample[modality]['input_mask'][0:1])
    prediction = codec(image, mask)
    if prediction['indices'] is None:
        raise ValueError('AE/VAE have continuous latents; select a VQ or FSQ checkpoint')
    blob = pack(prediction['indices'][0].numpy(), codec.quantizer.size, mask[0].numpy(),
                {'modality': modality, 'sample_id': sample['sample_id'],
                 'checkpoint_sha256': sha256_file(checkpoint_path),
                 'multiscale_grids':list(getattr(codec,'scales',[])),
                 'normalization_sha256': ds.sources[modality].fingerprint['normalization']})
    ids, recovered_mask, header = unpack(blob)
    decoded = codec.decode_indices(torch.from_numpy(ids[None]))
    error = float((decoded-prediction['reconstruction']).abs().max())
    if not np.array_equal(recovered_mask, mask[0].numpy()) or error > 2e-5:
        raise ValueError(f'Token roundtrip failed: {error}')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(blob)
    result = {'path': str(output.resolve()), 'bytes_including_mask_header_checksum': len(blob),
              'token_bytes': header['token_bytes'], 'input_mask_bytes': header['mask_bytes'],
              'total_bpp_including_side_information': 8*len(blob)/(256*256),
              'decoder_roundtrip_max_error': error, 'model_weights_included': False,
              'entropy_coding': False, 'sha256': sha256_file(output)}
    json_write(output.with_suffix(output.suffix+'.json'), result)
    ds.close()
    return result


if __name__ == '__main__':
    main()
