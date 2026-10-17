"""Read the existing DiT jobs and download evidence; never launch GPU jobs."""
import argparse
import json
import os
from pathlib import Path

os.environ['MODAL_PROFILE'] = 'runlongyu-target'
os.environ['MODAL_ENVIRONMENT'] = 'main'
import modal

OUT = Path(__file__).resolve().parents[2] / 'output/efdiff_hlc_dit_timegeo_v_20261004'
APP = 'efdiff-hlc-dit-timegeo-v-a100-20261004'
RUN = 'hlc-dit-timegeo-v-b128-fullval-epoch2-20261004-v1'


def fetch(download=False):
    OUT.mkdir(parents=True, exist_ok=True)
    fn = modal.Function.from_name(APP, 'job', environment_name='main')
    stats = fn.get_current_stats()
    result = modal.Function.from_name(APP, 'status', environment_name='main').remote()
    result['job_stats'] = dict(backlog=stats.backlog, runners=stats.num_total_runners,
                               running=stats.num_running_inputs)
    result['calls'] = {}
    for scope in ('smoke', 'train'):
        initial = result.get(scope+'_launch.json', {}).get('calls', {})
        for variant in ('v',):
            current = result.get(scope, {}).get(variant, {}).get('pipeline.json', {}).get('call_id')
            call_id = current or initial.get(variant)
            if not call_id:
                continue
            seen = set()
            while call_id not in seen:
                seen.add(call_id)
                try:
                    returned = modal.FunctionCall.from_id(call_id).get(timeout=0)
                    state = dict(call_id=call_id, returned=returned)
                    if returned.get('phase') == 'queued_continuation':
                        call_id = returned['call_id']
                        continue
                except TimeoutError:
                    state = dict(call_id=call_id, phase='pending')
                except Exception as exc:
                    state = dict(call_id=call_id, error=repr(exc))
                break
            result['calls'][scope+'/'+variant] = state
    (OUT/'fetch_status.json').write_text(json.dumps(result, indent=2)+'\n')
    if download:
        volume = modal.Volume.from_name('efdiff-hlc-synthetic-runs', environment_name='main')
        try:
            entries = list(volume.iterdir(RUN, recursive=True))
        except modal.exception.NotFoundError:
            entries = []
            result['download_state'] = 'run_directory_not_committed_yet'
        for entry in entries:
            rel = Path(entry.path).relative_to(RUN)
            if rel.suffix not in ('.json', '.jsonl', '.npz', '.png', '.md', '.log') or 'source' in rel.parts:
                continue
            dest = OUT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b''.join(volume.read_file(entry.path)))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    fetch(parser.parse_args().download)
