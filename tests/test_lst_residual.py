import torch
from torch import nn
import pytest
from experiments.paper1.lst_residual.runtime import visible_mean, spatial_mse, ResidualCodec


def test_visible_baseline_ignores_missing_pixels_and_rejects_empty():
    image = torch.tensor([[[[1., 3.], [9999., float('nan')]]]])
    mask = torch.tensor([[[[True, True], [False, False]]]])
    torch.testing.assert_close(visible_mean(image, mask), torch.tensor([[[[2.]]]]))
    with pytest.raises(ValueError):
        visible_mean(image, torch.zeros_like(mask))


def test_spatial_loss_removes_only_mean_and_has_correct_gradient():
    target = torch.tensor([[[[24., 25., 26.]]]])
    pred = torch.tensor([[[[25., 25., 25.]]]], requires_grad=True)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = spatial_mse(pred, target, mask)
    torch.testing.assert_close(loss, torch.tensor(2/3))
    torch.testing.assert_close(spatial_mse(pred+7, target, mask), loss)
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[[[2/3, 0., -2/3]]]]))


class IdentityCodec(nn.Module):
    config = {'quantizer': 'var_vq'}
    def forward(self, image, mask):
        return {'reconstruction': image}
    def decode_indices(self, indices):
        return indices
    def rate(self):
        return {'bits_per_image': 6800, 'bpp_spatial': 6800/65536, 'bpp_band_pixel': 6800/65536}


def test_residual_roundtrip_shift_equivariance_and_rate():
    codec = ResidualCodec(IdentityCodec())
    image = torch.tensor([[[[1., 3.], [0., 0.]]]])
    mask = torch.tensor([[[[True, True], [False, False]]]])
    out = codec(image, mask)
    expected = torch.tensor([[[[1., 3.], [2., 2.]]]])
    torch.testing.assert_close(out['reconstruction'], expected)
    shifted = codec(torch.where(mask, image+10, 0), mask)
    torch.testing.assert_close(shifted['predicted_residual'], out['predicted_residual'])
    torch.testing.assert_close(shifted['reconstruction'], expected+10)
    torch.testing.assert_close(codec.decode_indices(out['predicted_residual'], out['temperature_baseline']), expected)
    assert codec.rate()['bits_per_image'] == 6832
