"""Download verified epoch-one inference weights; reuse the verified frozen encoder."""
import hashlib
import json
import shutil
from pathlib import Path
import modal
import download_local_export as transfer

transfer.PREFIX = 'hlc-synthetic-full-epoch1-ddp2-io16-20260916-v3/local_export_epoch1_v1'
transfer.OUT = Path(__file__).resolve().parents[2]/'output/efdiff_hlc_epoch1_deadline_io16_20260916/models'


if __name__ == '__main__':
    transfer.OUT.mkdir(exist_ok=True,parents=True)
    volume = modal.Volume.from_name('efdiff-hlc-synthetic-runs',environment_name='main')
    manifest = json.loads(b''.join(volume.read_file(transfer.PREFIX+'/manifest.json')))
    previous = transfer.OUT.parents[1]/'efdiff_hlc_pilot_20260915/models/prithvi_encoder_batch1.pt'
    target = transfer.OUT/'prithvi_encoder.pt'
    if previous.exists() and not target.exists():
        with previous.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        if digest == manifest['files']['prithvi_encoder.pt']['sha256']:
            shutil.copyfile(previous,target)
            print('Reused exact verified frozen encoder already on this machine',flush=True)
    transfer.main()
