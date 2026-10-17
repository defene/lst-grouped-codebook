import copy
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from eo_denoise.config import DEFAULT, validate
from eo_denoise.models import Codec
from eo_denoise.losses import objective
from eo_denoise.prithvi_reference import PrithviMAE
from eo_denoise.prithvi_spatial import spatial_position
from timm.models.vision_transformer import Block


def test_official_spatial_blocks_budget_and_patch_order():
    c = dict(DEFAULT['model'], backbone='prithvi_eo2_spatial')
    for channels in (1, 6):
        model = Codec(channels, c)
        conv = Codec(channels, dict(c, backbone='conv'))
        count = sum(p.numel() for p in model.parameters())
        conv_count = sum(p.numel() for p in conv.parameters())
        assert count < 108948355 and abs(count/conv_count-1) < .04
        backbone = model.backbone
        assert len(backbone.encoder.blocks) == 6 and len(backbone.decoder.decoder_blocks) == 3
        assert all(isinstance(b, Block) for b in backbone.encoder.blocks)
        assert not any(isinstance(m, torch.nn.Conv3d) for m in model.modules())
        assert not backbone.encoder.temporal_encoding and not backbone.decoder.temporal_encoding
        assert not backbone.encoder.location_encoding and not backbone.decoder.location_encoding
        assert not any('temporal_embed' in k or 'location_embed' in k for k in model.state_dict())
        torch.testing.assert_close(backbone.encoder.pos_embed, spatial_position(c['eo_dim']))
        # Compare the decoder's pixel layout to the reference MAE unpatchify.
        patches = torch.arange(256*256*channels).reshape(1, 256, 16*16*channels)
        stub = SimpleNamespace(encoder=SimpleNamespace(in_chans=channels, img_size=(256,256),
                  patch_embed=SimpleNamespace(patch_size=(1,16,16))))
        expected = PrithviMAE.unpatchify(stub, patches).squeeze(2)
        ours = patches.reshape(1,16,16,16,16,channels).permute(0,5,1,3,2,4).reshape(1,channels,256,256)
        torch.testing.assert_close(ours, expected)


def test_single_view_loss_and_stability_validation():
    c = copy.deepcopy(DEFAULT)
    c['data'].update(modalities=['lst'], train_views=1)
    c['loss'].update(stability=0, cross_view=0)
    validate(c)
    pred = torch.ones(2,1,8,8,requires_grad=True)
    mask = torch.ones(2,1,8,8,dtype=torch.bool)
    out = {'reconstruction':pred, 'bottleneck_loss':pred.new_zeros(()), 'kl':pred.new_zeros(())}
    batch = {'lst':{'target':torch.zeros_like(pred), 'reference_mask':mask, 'input_mask':mask[:,None]}}
    total, terms = objective({'lst':[out]}, batch, c)
    assert total.item() == 1 and terms['stability'].item() == 0
    total.backward()
    assert pred.grad.abs().sum() > 0
    c['loss']['stability'] = .1
    with pytest.raises(ValueError, match='two training views'):
        validate(c)
