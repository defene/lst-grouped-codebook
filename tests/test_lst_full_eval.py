import numpy as np
import torch
from eo_data import open_dataset
from experiments.paper1.lst_full_eval.run import ROOT, UnitAdapter
from experiments.paper1.lst_full_eval.data import PREPARED


class RecordingCodec(torch.nn.Module):
    def forward(self, image, mask):
        self.last_image = image.clone()
        return {'reconstruction': image + 0.25}


def test_adapter_preserves_physical_error_and_missing_zero():
    inner = RecordingCodec()
    adapter = UnitAdapter(inner, {'mean': [2000.], 'std': [1500.]},
                          {'mean': [2500.], 'std': [1600.]})
    image = torch.tensor([[[[0.2, float('nan')], [-0.3, 0.7]]]])
    mask = torch.tensor([[[[True, False], [True, True]]]])
    output = adapter(image, mask)['reconstruction']
    assert inner.last_image[~mask].item() == 0
    assert torch.isfinite(output).all()
    # An old-normalization error of .25 represents 3.75 Celsius in both systems.
    assert torch.allclose(((output-image)*16)[mask], torch.full((3,), 3.75), atol=1e-5)


def test_v2_test_record_and_physical_temperature_match_frozen_v1():
    old = open_dataset('prithvi', 'lst', 'val', root=ROOT, prepared=ROOT/'processed_data/v1')
    new = open_dataset('prithvi', 'lst', 'test', root=ROOT, prepared=ROOT/PREPARED)
    try:
        assert np.array_equal(old.indices[:, 0], new.indices[:, 0])
        for i in (0, 447, len(new)-1):
            a, b = old[i], new[i]
            assert np.array_equal(a['reference_mask'], b['reference_mask'])
            valid = a['reference_mask'].astype(bool)
            assert np.allclose((old.denormalize(a['image'])/100)[valid],
                               (new.denormalize(b['image'])/100)[valid], atol=2e-5, rtol=0)
    finally:
        old.close()
        new.close()
