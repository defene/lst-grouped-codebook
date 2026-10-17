"""Two-scale SatMAE++ LST adaptation. No jobs or model construction on import.

Uses the pinned official MAE and convolutional upsampling modules. Replaces
internal random masking with the shared external physical patch mask. The
encoder consumes only observed data downsampled to 128; full 256 targets are
available to the objective only. All weights are initialized from scratch.
"""
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .models import (_MetadataMixin, ObservationAdapter, MeanAffine, PrithviSpatialMAE,
                     _shape_mask, _checkpoint_call)


class SatMAEPlusPlusLST(_MetadataMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        from .satmae_reference import MaskedAutoencoderViT
        c, task = config["model"], config["task"]
        if c.get("initialization") != "scratch" or c.get("pretrained_path"):
            raise ValueError("SatMAE++ LST architecture comparison is scratch-only")
        self.kind, self.config = "satmae_pp", config
        self.channels = c["channels"]
        self.encoder_dim = c["encoder_dim"]
        self.checkpointing = c.get("checkpointing", False)
        self.backbone = MaskedAutoencoderViT(img_size=128, patch_size=8, in_chans=self.channels,
            embed_dim=c["encoder_dim"], depth=c["encoder_depth"], num_heads=c["encoder_heads"],
            decoder_embed_dim=c["decoder_dim"], decoder_depth=c["decoder_depth"],
            decoder_num_heads=c["decoder_heads"], mlp_ratio=4.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False,
            proj_ratio=c["upsample_projection_ratio"])
        self.input_adapter = ObservationAdapter(self.channels, task["mean_scale_celsius"], task["mean_center_celsius"])
        self.decoder_mean = MeanAffine(c["decoder_dim"], task["mean_scale_celsius"], task["mean_center_celsius"])

    def forward(self, encoded_input, reference_valid, observed_valid, mean_celsius, patch_missing=None):
        if encoded_input.shape[-2:] != (256, 256):
            raise ValueError("SatMAE++ LST requires native 256x256 observations")
        r = _shape_mask(reference_valid, encoded_input, "reference_valid")
        o = _shape_mask(observed_valid, encoded_input, "observed_valid")
        missing = PrithviSpatialMAE._patch_mask(patch_missing, encoded_input, r, o)
        # Non-overlapping 2x2 area pooling after masking cannot draw clean values
        # across a held-out 16x16 physical patch boundary. Invalid values ignored.
        count = F.avg_pool2d(o.float(), 2)
        numerator = F.avg_pool2d(torch.where(o, encoded_input.float(), 0.), 2)
        low = torch.where(count > 0, numerator / count.clamp_min(.25), 0.)
        low_r = F.avg_pool2d(r.float(), 2) > 0
        low_o = count > 0
        low = self.input_adapter(low, low_r, low_o, mean_celsius)
        b = self.backbone
        patches = b.patch_embed(low) + b.pos_embed[:, 1:]
        order = torch.argsort(missing.long() * 256 + torch.arange(256, device=low.device), dim=1)
        keep_count = 256 - int(missing[0].sum())
        keep = order[:, :keep_count]
        visible = torch.gather(patches, 1, keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
        cls = (b.cls_token + b.pos_embed[:, :1]).expand(low.shape[0], -1, -1)
        latent = torch.cat([cls, visible], 1)
        check = self.training and self.checkpointing
        for block in b.blocks:
            latent = _checkpoint_call(block, latent, check)
        latent = b.norm(latent)
        projected = b.decoder_embed(latent)
        restored = b.mask_token.to(projected.dtype).expand(low.shape[0], 256, -1).clone()
        restored.scatter_(1, keep.unsqueeze(-1).expand(-1, -1, projected.shape[-1]), projected[:, 1:])
        decoded = torch.cat([projected[:, :1], restored], 1)
        decoded = self.decoder_mean(decoded, mean_celsius) + b.decoder_pos_embed
        for block in b.decoder_blocks:
            decoded = _checkpoint_call(block, decoded, check)
        pixels = b.decoder_pred(b.decoder_norm(decoded))[:, 1:]
        low_prediction = b.unpatchify(pixels, 8, self.channels)
        high_prediction = b.forward_multiscale(pixels)
        return {"encoded_prediction": high_prediction.float(),
                "encoded_prediction_low": low_prediction.float(),
                "indices": None, "latent": latent, "kl": latent.new_zeros(()),
                "bottleneck_loss": latent.new_zeros(()), "mae_patch_missing": missing,
                "mae_visible_patch_indices": keep}

    def decode_indices(self, indices, mean_celsius):
        raise ValueError("SatMAE++ is a continuous MAE, with no discrete codebook")

    def rate(self):
        return {"payload_type": "continuous_visible_mae_tokens", "patch_size": 8,
                "patch_footprint_native_pixels": 16, "encoder_resolution": 128,
                "reconstruction_resolutions": [128, 256], "full_patch_tokens": 256,
                "cls_tokens": 1, "feature_dim": self.encoder_dim,
                "mean_side_information_bits": 32, "equal_rate_claim": False,
                "visible_token_count_depends_on_mask": True,
                "decoder_also_requires_patch_positions": True,
                "adaptation": "two-scale SatMAE++; LST physical masked multiscale loss; scratch"}
