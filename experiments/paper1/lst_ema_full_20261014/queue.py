"""Durable, sequential local GPU queue. Rerun the same command to resume."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys
from pathlib import Path
from eo_data.locking import build_lock
from .config import PROJECT,write_json
from .runtime import train,check_stop,utc,source_hashes,execution_guard

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--job');args=ap.parse_args()
    root=Path(args.output)
    if args.job:
        c=json.loads((root/'configs'/f'{args.job}.json').read_text());train(c,root/'runs'/args.job);return
    with build_lock(root):
        queue=json.loads((root/'queue.json').read_text())
        if queue['status']!='ready' or queue['source_hashes']!=source_hashes():raise RuntimeError('Queue not ready or source changed')
        completed=[]
        for job in queue['jobs']:
            check_stop(root/'runs')
            config=json.loads(Path(job['config']).read_text());execution_guard(config)
            run=root/'runs'/job['name']
            if (run/'training_complete.json').exists():
                # Validate saved budget and epoch receipts rather than blindly skip markers.
                from .epoch_validation import reconcile_selection
                state=json.loads((run/'training_complete.json').read_text())
                if state['records']!=1094875 or state['updates']!=17110:raise ValueError('Invalid completion budget')
                reconcile_selection(config,run,5);completed.append(job['name']);continue
            status=dict(status='running',active=job['name'],completed=completed,pid=os.getpid(),utc=utc())
            write_json(root/'queue_status.json',status)
            logdir=root/'logs';logdir.mkdir(exist_ok=True)
            with (logdir/(job['name']+'.log')).open('a',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','experiments.paper1.lst_ema_full_20261014.queue',
                    '--output',str(root),'--job',job['name']],cwd=str(PROJECT),stdout=log,stderr=subprocess.STDOUT)
                write_json(root/'queue_status.json',{**status,'child_pid':process.pid})
                code=process.wait()
            if code:
                write_json(root/'queue_status.json',{**status,'status':'stopped' if (root/'STOP').exists() else 'failed','exit_code':code,'utc':utc()})
                raise SystemExit(code)
            if not (run/'training_complete.json').exists():raise RuntimeError('Child exited without completed training')
            completed.append(job['name'])
        write_json(root/'queue_status.json',dict(status='completed',completed=completed,utc=utc()))

if __name__=='__main__':main()
