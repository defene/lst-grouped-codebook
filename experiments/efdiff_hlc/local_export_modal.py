"""Read completed pilot checkpoints, export portable inference weights, no training."""
import json
from pathlib import Path
import modal
from modal_app import image, mounts, runs_volume

app = modal.App('efdiff-hlc-pilot-local-export')
RUN = 'hlc-synthetic-pilot-16384-512-v1'


@app.function(image=image,volumes=mounts,cpu=8,memory=32768,timeout=3600)
def export():
    import hashlib
    import torch
    from unittest.mock import patch
    import pilot_data  # establishes the pinned upstream import path
    from model.prithvi_encoder import PrithviEncoder
    from terratorch.registry import BACKBONE_REGISTRY
    from pilot_report import build
    torch.set_num_threads(8)
    root = Path('/runs')/RUN
    dest = root/'local_export_v1'
    dest.mkdir(exist_ok=True)
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):
                h.update(block)
        return h.hexdigest()
    files, checkpoints = {},{}
    encoder_reference = None
    for variant in ('x0','epsilon'):
        status = json.loads((root/variant/'status.json').read_text())
        assert status['phase']=='training_complete'
        checkpoint = root/variant/'latest.pt'
        state = torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        assert state['epoch']==5 and state['offset']==0 and state['step']==1280 and state['presented']==81920
        assert state['variant']==variant and not state['smoke']
        encoder_state = {k.removeprefix('prithvi_encoder.'):v for k,v in state['model'].items() if k.startswith('prithvi_encoder.')}
        if encoder_reference is None:
            with patch.object(PrithviEncoder,'_load_backbone',lambda self,name:BACKBONE_REGISTRY.build('prithvi_eo_v2_300',pretrained=False)):
                encoder = PrithviEncoder().eval()
            encoder.load_state_dict(encoder_state,strict=True)
            encoder_reference = {k:v.clone() for k,v in encoder_state.items()}
            example = torch.rand(1,6,224,224)
            with torch.inference_mode():
                traced = torch.jit.trace(encoder,example,check_trace=False)
                for seed in (123,321):
                    torch.manual_seed(seed)
                    x = torch.rand(1,6,224,224)
                    torch.testing.assert_close(traced(x),encoder(x),rtol=1e-5,atol=1e-5)
            traced.save(str(dest/'prithvi_encoder_batch1.pt'))
            del traced,encoder
        else:
            assert encoder_reference.keys()==encoder_state.keys()
            assert all(torch.equal(encoder_reference[k],v) for k,v in encoder_state.items())
        provenance = json.loads((root/variant/'provenance.json').read_text())
        torch.save({'variant':variant,'checkpoint':'fixed_epoch5_ema','epoch':5,'step':1280,
                    'presented':81920,'denoiser':state['ema'],'config':provenance['model_config'],
                    'normalization':provenance['normalization'],'data_fingerprint':state['data_fingerprint'],
                    'training_code_sha256':state['code_sha256']},dest/f'{variant}_epoch5_ema.pt')
        checkpoints[variant] = {'source':str(checkpoint.relative_to(root)), 'sha256':digest(checkpoint)}
        del state,encoder_state
        print('Exported',variant,flush=True)
    for p in sorted(dest.glob('*.pt')):
        files[p.name] = {'bytes':p.stat().st_size,'sha256':digest(p)}
    manifest = {'source_run':RUN,'checkpoint':'fixed_epoch5_ema','files':files,'source_checkpoints':checkpoints,
                'encoder':'exact trained frozen encoder, TorchScript export validated on two random inputs; batch 1, 6x224x224',
                'same_encoder_in_both_models':True,'optimizer_states_excluded':True}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report = build(root)
    (root/'recovery_20260916.json').write_text(json.dumps({'training_verified_complete':True,'report_regenerated':True,
      'original_pipeline_error_preserved':True,'note':'Original assertion occurred when a completed smoke run returned smoke_passed instead of checkpointed; both training artifacts are complete.'},indent=2)+'\n')
    runs_volume.commit()
    return manifest


@app.function(image=image,volumes=mounts,cpu=2,memory=4096,timeout=600)
def repair_report():
    from pilot_report import build
    directory = Path('/runs')/RUN
    assert (directory/'local_export_v1/manifest.json').exists()
    for variant in ('x0','epsilon'):
        assert json.loads((directory/variant/'status.json').read_text())['phase']=='training_complete'
    result = build(directory)
    original = directory/'pipeline.json'
    history = directory/'pipeline_before_recovery_20260916.json'
    if not history.exists():
        history.write_bytes(original.read_bytes())
    original.write_text(json.dumps({'phase':'complete','scope':'pilot','comparison':result,
        'recovery_note':'Both training artifacts verified; report rebuilt after completed-smoke replay assertion. Original error preserved in pipeline_before_recovery_20260916.json.'},indent=2)+'\n')
    runs_volume.commit()
    return {'phase':'complete','report_rebuilt':True}


@app.local_entrypoint()
def main(report_only:bool=False):
    print(json.dumps(repair_report.remote() if report_only else export.remote(),indent=2))
