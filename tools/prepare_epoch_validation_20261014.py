"""Create a new source version; never edit the frozen experiment exports."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/torch_models_20261013/sources'
DEST = ROOT / 'experiments/paper1/lst_restoration_epochval_20261014'


def main():
    if DEST.exists():
        raise SystemExit('Refusing to replace existing version')
    src = BASE / 'stage2_vq_20261005_attempt02/experiments/paper1/lst_restoration'
    grouped = BASE / 'stage3_grouped_20261005_attempt01/experiments/paper1/lst_restoration'
    DEST.mkdir(parents=True)
    origins = {}
    files = list(src.glob('*.py')) + list((src / 'geo_sources').glob('*'))
    files += [grouped / n for n in ('grouped_adapter.py', 'vendor_grouped_quantizer.py', 'grouped_audit.py')]
    for path in files:
        if not path.is_file():
            continue
        relative = path.relative_to(grouped if path.is_relative_to(grouped) else src)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        origins[str(relative).replace('\\', '/')] = {
            'source': str(path.relative_to(ROOT)).replace('\\', '/'),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (DEST / 'SOURCE_ORIGINS.json').write_text(json.dumps(origins, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
