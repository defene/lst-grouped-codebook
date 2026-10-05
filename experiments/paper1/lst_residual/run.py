"""Train/evaluate one isolated, scratch-initialized temperature ablation."""
from pathlib import Path
import argparse
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eo_denoise import engine
from eo_denoise.config import validate
from experiments.paper1.lst_residual.runtime import install, HERE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'evaluate'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--coverage', type=float, default=.1)
    args = parser.parse_args()
    install()
    if args.stage == 'train':
        config = json.loads(args.config.read_text(encoding='utf-8'))
        validate(config)
        experiment = config['residual_experiment']
        assert set(experiment) == {'predict_residual', 'spatial_extra_weight', 'id'}
        assert isinstance(experiment['predict_residual'], bool) and experiment['spatial_extra_weight'] in (0, 3)
        checkpoint = args.output/'last.pt'
        resume = checkpoint if checkpoint.exists() else None
        engine.train(config, args.output, resume=resume)
        with zipfile.ZipFile(args.output/'experiment_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for p in HERE.glob('*.py'):
                archive.write(p, 'experiments/paper1/lst_residual/'+p.name)
    else:
        engine.evaluate_checkpoint(args.checkpoint, args.output, coverage=args.coverage)


if __name__ == '__main__':
    main()
