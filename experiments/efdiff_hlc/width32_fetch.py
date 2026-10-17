"""Read status and download completed comparison artifacts; never starts training."""
import argparse
import json
import os
from pathlib import Path


def main(download=False):
    os.environ['MODAL_PROFILE']='runlongyu-target'
    os.environ['MODAL_ENVIRONMENT']='main'
    import modal
    status=modal.Function.from_name('efdiff-hlc-width32-concurrent-12h-20260923','status',environment_name='main').remote()
    out=Path(__file__).resolve().parents[2]/'output/efdiff_hlc_width32_12h_20260923'
    out.mkdir(parents=True,exist_ok=True)
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    if download and 'latest_comparison.json' in status:
        volume=modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
        records=status['latest_comparison.json']['training_records_per_model']
        destination=out/'comparisons'/str(records)
        destination.mkdir(parents=True,exist_ok=True)
        for name in ['comparison.json','REPORT.md']+[f'comparison_{i}.png' for i in range(1,5)]:
            remote=f'hlc-width32-single-12h-20260923-v1/comparisons/{records}/{name}'
            with (destination/name).open('wb') as f:
                for chunk in volume.read_file(remote):
                    f.write(chunk)
        print(json.dumps({'downloaded':str(destination)}))
    compact={k:v for k,v in status.items() if k not in ('plan.json','concurrent_migration.json')}
    if 'concurrent_migration.json' in status:
        compact['migration']={k:status['concurrent_migration.json'][k]
            for k in ('resume_step','resume_records','new_execution')}
    print(json.dumps(compact,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--download',action='store_true')
    main(parser.parse_args().download)
