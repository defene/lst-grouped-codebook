from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, os, shutil, time, zipfile, threading

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'lstsr_tb_final_upload_20260721'
DEST = ROOT / 'data_inspection' / 'extracted_metadata'
STATUS = ROOT / 'data_inspection' / 'extraction_status.json'
DEST.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
started = time.time()
state = {'status':'running','destination':str(DEST),'completed':[], 'active':{}, 'errors':[]}

def save():
    state['elapsed_seconds'] = round(time.time()-started,1)
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2),encoding='utf-8')
    os.replace(tmp,STATUS)

def extract(path):
    with zipfile.ZipFile(path) as z:
        entries = [i for i in z.infolist() if not i.is_dir()]
        size = sum(i.file_size for i in entries)
        done = 0
        for idx,i in enumerate(entries,1):
            relative = Path(i.filename)
            if relative.is_absolute() or '..' in relative.parts or relative.drive:
                raise ValueError('Unsafe member: '+i.filename)
            target = DEST / relative
            target.parent.mkdir(parents=True,exist_ok=True)
            partial = target.with_name(target.name+'.extracting')
            with z.open(i) as src, partial.open('wb') as dst:
                shutil.copyfileobj(src,dst,1024*1024)
            if partial.stat().st_size != i.file_size:
                raise IOError('Size mismatch: '+i.filename)
            os.replace(partial,target)
            done += i.file_size
            if idx % 1000 == 0 or idx == len(entries):
                with lock:
                    state['active'][path.name]={'files':idx,'total_files':len(entries),'bytes':done,'total_bytes':size}
                    save()
        with lock:
            state['active'].pop(path.name,None)
            state['completed'].append({'archive':path.name,'files':len(entries),'bytes':done})
            save()
            print(f'Completed {len(state["completed"])}/53: {path.name}',flush=True)

if __name__ == '__main__':
    paths = sorted(SOURCE.glob('*.zip'))
    with lock: save()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(extract,p):p for p in paths}
        for job in as_completed(jobs):
            try: job.result()
            except Exception as e:
                with lock:
                    state['errors'].append({'archive':jobs[job].name,'error':repr(e)})
                    save()
                print('ERROR',jobs[job].name,repr(e),flush=True)
    state['status']='failed' if state['errors'] else 'complete'
    state['total_files']=sum(x['files'] for x in state['completed'])
    state['total_bytes']=sum(x['bytes'] for x in state['completed'])
    save()
    print(json.dumps({k:v for k,v in state.items() if k not in ['completed','active']},indent=2),flush=True)
