from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.paper1.lst_grouped_vq.train import train, final_evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'evaluate'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-after', type=int)
    a = parser.parse_args()
    c = json.loads(a.config.read_text(encoding='utf-8'))
    if a.stage == 'train':
        train(c, a.output, a.stop_after)
    else:
        final_evaluate(c, a.output)
