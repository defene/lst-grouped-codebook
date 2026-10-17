"""Export the fixed full-training epoch-one EMA weights for local test evaluation."""
import json
from pathlib import Path
import modal
from modal_app import image, mounts, runs_volume

app = modal.App('efdiff-hlc-epoch1-local-export-20260921')
RUN = 'hlc-synthetic-full-epoch1-ddp2-io16-20260916-v3'


@app.function(image=image, volumes=mounts, cpu=8, memory=32768, timeout=3600)
def export():
    import hashlib
    import shutil
    import torch
    from unittest.mock import patch
    import data
    from model.prithvi_encoder import PrithviEncoder
    from terratorch.registry import BACKBONE_REGISTRY
    torch.set_num_threads(8)
    root = Path('/runs')/RUN
    dest = root/'local_export_epoch1_v1'
    dest.mkdir(exist_ok=True)
    old_encoder = Path('/runs/hlc-synthetic-pilot-16384-512-v1/local_export_v1/prithvi_encoder_batch1.pt')
    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream,'sha256').hexdigest()
    files, checkpoints = {}, {}
    encoder_reference = None
    for variant in ('x0','epsilon'):
        directory = root/'train'/variant
        status = json.loads((directory/'status.json').read_text())
        assert status['phase']=='training_complete'
        checkpoint = directory/'latest.pt'
        state = torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        assert state['epoch']==1 and state['offset']==0 and state['step']==40763
        assert state['presented']==2608772 and state['variant']==variant and not state['smoke']
        encoder_state = {k.removeprefix('prithvi_encoder.'):v for k,v in state['model'].items() if k.startswith('prithvi_encoder.')}
        if encoder_reference is None:
            with patch.object(PrithviEncoder,'_load_backbone',lambda self,name:BACKBONE_REGISTRY.build('prithvi_eo_v2_300',pretrained=False)):
                encoder = PrithviEncoder().eval()
            encoder.load_state_dict(encoder_state,strict=True)
            encoder_reference = {k:v.clone() for k,v in encoder_state.items()}
            traced = torch.jit.load(str(old_encoder),map_location='cpu').eval()
            # Reuse the local portable file only after exact state and dynamic-batch checks.
            assert set(traced.state_dict())==set(encoder_state)|{'pixel_mean','pixel_std'}
            assert all(torch.equal(traced.state_dict()[k],v) for k,v in encoder_state.items())
            assert torch.equal(traced.pixel_mean,encoder.pixel_mean)
            assert torch.equal(traced.pixel_std,encoder.pixel_std)
            with torch.inference_mode():
                for batch in (1,4,16):
                    torch.manual_seed(20260921+batch)
                    x = torch.rand(batch,6,224,224)
                    actual = traced(x)
                    assert actual.shape==(batch,196,1024)
                    torch.testing.assert_close(actual,encoder(x),rtol=1e-5,atol=1e-5)
            shutil.copyfile(old_encoder,dest/'prithvi_encoder.pt')
            del traced,encoder
        else:
            assert encoder_reference.keys()==encoder_state.keys()
            assert all(torch.equal(encoder_reference[k],v) for k,v in encoder_state.items())
        provenance = json.loads((directory/'provenance.json').read_text())
        torch.save({'variant':variant,'checkpoint':'fixed_epoch1_ema','epoch':1,'step':40763,
            'presented':2608772,'denoiser':state['ema'],'config':provenance['model_config'],
            'normalization':provenance['normalization'],'data_fingerprint':state['data_fingerprint'],
            'training_code_sha256':state['code_sha256']},dest/f'{variant}_epoch1_ema.pt')
        checkpoints[variant]={'source':str(checkpoint.relative_to(root)),'sha256':digest(checkpoint)}
        del state,encoder_state
        print('Exported epoch-one',variant,flush=True)
    for p in sorted(dest.glob('*.pt')):
        files[p.name]={'bytes':p.stat().st_size,'sha256':digest(p)}
    manifest={'source_run':RUN,'checkpoint':'fixed_epoch1_ema','files':files,'source_checkpoints':checkpoints,
        'same_encoder_in_both_models':True,'encoder_matches_prior_portable_file':True,
        'encoder_validated_batch_sizes':[1,4,16],'optimizer_states_excluded':True}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    runs_volume.commit()
    return manifest


@app.local_entrypoint()
def main():
    print(json.dumps(export.remote(),indent=2))
