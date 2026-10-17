"""Durable local full-test process; publish final metadata only after full coverage."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'output/efdiff_hlc_epoch1_deadline_io16_20260916'
DEST=OUT/'full_test_20260921'


def save(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def main():
    DEST.mkdir(exist_ok=True)
    smoke=json.loads((OUT/'local_epoch1_smoke/metrics.json').read_text())
    assert smoke['status']=='complete' and smoke['split']=='val' and smoke['records']==32
    state={'phase':'running','pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Full HLC test set, both fixed epoch-one EMA variants','records_per_model':141524}
    save(DEST/'pipeline.json',state)
    try:
        subprocess.run([sys.executable,'-u',str(HERE/'full_test_local.py')],cwd=ROOT,check=True)
        result=json.loads((DEST/'metrics.json').read_text())
        assert result['status']=='complete' and result['split']=='test' and result['records']==141524
        assert all(v['records']==141524 for v in result['metrics'].values())
        state.update(phase='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),result=result)
        save(DEST/'pipeline.json',state)
        manifest_path=HERE/'deadline_run_manifest.json'
        manifest=json.loads(manifest_path.read_text())
        manifest['local_full_test']={'phase':'complete','completed_at_utc':state['completed_at_utc'],
            'records_per_model':141524,'checkpoint':'fixed_epoch1_ema','test_used_for_selection':False,
            'report':(DEST/'REPORT.md').relative_to(ROOT).as_posix(),'metrics':result['metrics']}
        save(manifest_path,manifest)
        print('FULL TEST COMPLETE: '+str(DEST/'REPORT.md'),flush=True)
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc),stopped_at_utc=datetime.now(timezone.utc).isoformat())
        save(DEST/'pipeline.json',state)
        raise


if __name__=='__main__':
    main()
