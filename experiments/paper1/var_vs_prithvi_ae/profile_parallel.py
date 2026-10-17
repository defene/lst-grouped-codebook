"""Bounded concurrent benchmark; child workers start together after warmup."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def run(kind):
    folder=HERE/'bench'/kind;folder.mkdir(parents=True,exist_ok=False)
    barrier=folder/'GO'
    specs=[('conv',8,20),('prithvi',8,6)] if kind=='pair' else [('conv',4,12),('conv',4,12),('prithvi',8,6)]
    workers=[];logs=[];monitor=[]
    try:
        for index,(backbone,batch,cap) in enumerate(specs):
            output=folder/f'{index}_{backbone}.json'
            stream=(folder/f'{index}_{backbone}.log').open('w');logs.append(stream)
            p=subprocess.Popen([sys.executable,str(HERE/'profile_gpu.py'),'--backbone',backbone,
                '--batch',str(batch),'--cap',str(cap),'--seconds','20','--barrier',str(barrier),
                '--output',str(output)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            workers.append((p,output))
        deadline=time.time()+90
        while not all(o.with_suffix('.ready').exists() for _,o in workers):
            if any(p.poll() is not None for p,_ in workers):raise RuntimeError('Worker failed before barrier')
            if time.time()>deadline:raise TimeoutError('Warmup timeout')
            time.sleep(.2)
        barrier.write_text('start')
        while any(p.poll() is None for p,_ in workers):
            data=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            used,util=[float(v.strip()) for v in data.strip().split(',')]
            monitor.append(dict(time=time.time(),used_mib=used,utilization=util))
            time.sleep(1)
        if any(p.returncode for p,_ in workers):raise RuntimeError('Worker failed; inspect logs')
        result=dict(kind=kind,workers=[json.loads(o.read_text()) for _,o in workers],gpu=monitor,
                    peak_device_used_gib=max(m['used_mib'] for m in monitor)/1024)
        (folder/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(kind=kind,peak_device_used_gib=result['peak_device_used_gib'],
            workers=[{k:w[k] for k in ('backbone','batch','samples_per_second','memory')} for w in result['workers']])),flush=True)
    finally:
        for p,_ in workers:
            if p.poll() is None:subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
        for log in logs:log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['pair','triple']);run(p.parse_args().kind)
