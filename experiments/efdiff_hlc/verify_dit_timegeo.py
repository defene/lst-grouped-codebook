"""CPU integration audit on real HLC records; not an accuracy or GPU capacity test."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.utils.data import default_collate
from center_data import HLCPairs as CenterPairs
from timegeo_data import HLCPairs, check_stats
from timegeo_metadata import sha256
from dit_timegeo_model import TimeGeoDiT
from dit_timegeo_objectives import TimeGeoObjective


def verify():
    torch.set_num_threads(2)
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    output = root/'output/efdiff_hlc_dit_timegeo_20261004'
    plan = json.loads((here/'dit_timegeo_plan.json').read_text())
    stats = json.loads((root/plan['normalization_path']).read_text())
    report = dict(scope='CPU functional integration; tiny DiT and deterministic stand-in HLS encoder',
                  training_launched=False, gpu_capacity_tested=False, accuracy_evaluated=False,
                  test_used=False, real_samples=[], checks={})
    training_batch = None
    for split in ('train','val'):
        ds = HLCPairs(root,split=split,stats=stats,augment=False,metadata_dir=root/plan['metadata_path'])
        base = CenterPairs(root,split=split,stats=stats,augment=False)
        if split=='train': check_stats(stats,ds)
        rows=[]
        for i in (0,1,len(ds)-1):
            sample, reference = ds[(0,i)],base[(0,i)]
            for key in reference:
                np.testing.assert_array_equal(sample[key],reference[key])
            repeat=ds[(3,i)]
            for key in sample: np.testing.assert_array_equal(sample[key],repeat[key])
            np.testing.assert_equal(sample['month'],int(ds.lst.indices[i,1])+1)
            rows.append(sample)
            report['real_samples'].append(dict(split=split,index=i,record_id=sample['record_id'],
                    month=sample['month'],geo_latlon=sample['geo_latlon'].tolist()))
        if rows[0]['record_id']==rows[1]['record_id']:
            np.testing.assert_array_equal(rows[0]['geo_latlon'],rows[1]['geo_latlon'])
        if split=='train': training_batch=default_collate(rows[:2])
        ds.lst.close(); ds.hls.close(); base.lst.close(); base.hls.close()
    report['checks']['pixels_masks_normalization_match_frozen_center_loader']=True
    report['checks']['months_are_target_slots_and_crops_stable_across_epochs']=True
    class StandInEncoder(torch.nn.Module):
        def forward(self,hls):
            return torch.nn.functional.adaptive_avg_pool2d(hls, (2,2)).flatten(2).transpose(1,2)
    # Full pixel grid and patch16 retained; shrink only Transformer width/depth on CPU.
    for variant in ('flow','epsilon'):
        torch.manual_seed(20261004)
        net=TimeGeoObjective(TimeGeoDiT(image_size=224,patch_size=16,hidden_dim=32,depth=2,
                    heads=4,context_dim=6,condition_dim=8,gradient_checkpointing=False),StandInEncoder(),variant)
        opt=torch.optim.Adam(net.denoise_fn.parameters(),lr=.001)
        losses=[]
        for _ in range(4):
            opt.zero_grad(); loss=net(training_batch); loss.backward(); opt.step()
            assert torch.isfinite(loss); losses.append(loss.item())
        for encoder in (net.denoise_fn.calendar_geo.month_encoder,net.denoise_fn.calendar_geo.geo_encoder):
            assert sum(p.grad.abs().sum() for p in encoder.parameters()) > 0
        noise=torch.randn_like(training_batch['HR'])
        net.eval(); sample=net.sample(training_batch,5,noise)
        assert sample.shape==training_batch['HR'].shape and torch.isfinite(sample).all()
        restored=copy.deepcopy(net)
        restored.load_state_dict(net.state_dict())
        torch.testing.assert_close(sample,restored.sample(training_batch,5,noise))
        report['checks'][variant]=dict(finite_losses=losses,sampling_nfe=5,shape=list(sample.shape),
                                      metadata_gradients=True,state_dict_roundtrip=True)
    # Count the requested full architecture without allocating its weights on a GPU.
    with torch.device('meta'):
        full=TimeGeoDiT(**plan['architecture'])
    report['full_dit_parameters']=sum(p.numel() for p in full.parameters())
    report['added_condition_parameters']=sum(p.numel() for p in full.calendar_geo.parameters())
    historical=root/'output/efdiff_hlc_dit_a100_b128_fullval_epoch2_20261001/train/flow/provenance.json'
    frozen=json.loads(historical.read_text())['hashes']
    names=('dit_model.py','dit_objectives.py','center_data.py','data.py','center_crop_protocol.json',
           'dit_a100_epoch2_train.py','dit_a100_epoch2_plan.json','dit_full_validation.py')
    report['frozen_source_hashes']={name:sha256(here/name) for name in names}
    for name,digest in report['frozen_source_hashes'].items():
        assert digest==frozen['efdiff_hlc/'+name], f'Historical source modified: {name}'
    report['new_source_hashes']={p.name:sha256(p) for p in [
        *here.glob('dit_timegeo*'), here/'timegeo_data.py',here/'timegeo_metadata.py',
        here/'verify_dit_timegeo.py',here/'test_dit_timegeo.py'] if p.is_file()}
    report['metadata_audit']=json.loads((root/plan['metadata_path']/'audit.json').read_text())
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': verify()
