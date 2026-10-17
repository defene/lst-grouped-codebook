"""Explicit, locally guarded submission of the authorized smoke-then-train job."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ['MODAL_PROFILE'] = 'runlongyu-target'
os.environ['MODAL_ENVIRONMENT'] = 'main'
import modal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true', required=True)
    parser.parse_args()
    base = Path(__file__).resolve().parents[2]
    out = base/'output/efdiff_hlc_raw_mmdit_v_20261014'
    out.mkdir(parents=True, exist_ok=True)
    record = out/'launch_call.json'
    if record.exists():
        print(record.read_text())
        return
    fn = modal.Function.from_name('efdiff-hlc-raw-mmdit-v-a100-20261014', 'launch', environment_name='main')
    call = fn.spawn()
    result = dict(call_id=call.object_id, phase='cpu_preflight_submitted', unix=time.time())
    record.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
