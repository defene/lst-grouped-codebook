"""Restoration model contracts; import has no execution/download side effects.

Model construction is explicit. Natural-image weights are local-only and hash
checked. Capability status distinguishes source implementation from a passed
runtime gate: none of these new adapters has passed a forward/backward test.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .quantizers import build_quantizer


def capability_registry() -> dict[str, dict[str, Any]]:
    common = {"runtime_verified": False, "requires_runtime_gate": True}
    result = {k: {**common, "implementation": "implemented_source", "discrete": k != "ae",
                  "local_pretrained_required": False, "track": "restoration"}
              for k in ("vq", "fsq", "lfq", "bsq", "ae", "var_multiscale")}
    for kind in ("sd", "sdxl", "flux1"):
        result[kind] = {**common, "implementation": "local_weight_dependency_gated",
                        "discrete": False, "local_pretrained_required": True,
                        "format": "diffusers AutoencoderKL directory with complete SHA256 manifest",
                        "track": "restoration"}
    result["prithvi_mae"] = {**common, "implementation": "implemented_source",
                              "discrete": False, "track": "patch_mae_only",
                              "note": "single-frame 2D adaptation of pinned Prithvi encoder/MAE decoder; scratch only"}
    result["satmae_pp"] = {**common, "implementation": "implemented_source",
                            "discrete": False, "track": "patch_mae_only",
                            "note": "pinned two-scale SatMAE++ with shared physical patch mask; scratch; runtime unverified"}
    for kind in ("eo_vae", "terramind_fsq"):
        result[kind] = {**common, "implementation": "official_source_dependency_gated",
                        "discrete": kind != "eo_vae", "local_pretrained_required": False,
                        "track": "geo_architecture_scratch", "equal_parameter_claim": False}
    for kind in ("ae", "var_multiscale"):
        result[kind]["active_plan"] = False
    return result


def _shape_mask(mask: Tensor, reference: Tensor, label: str) -> Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    expected = (reference.shape[0], 1, reference.shape[-2], reference.shape[-1])
    if tuple(mask.shape) != expected:
        raise ValueError(f"{label} must have shape {expected}, got {tuple(mask.shape)}")
    return mask.bool()


def _normalized_mean(mean: Tensor, batch: int, scale: float, center: float, device) -> Tensor:
    if mean.numel() != batch:
        raise ValueError("Expected one mean temperature per image")
    return (mean.to(device=device, dtype=torch.float32).reshape(batch, 1) - center) / scale


class ObservationAdapter(nn.Module):
    """Only encoded observation, R, O and one scalar mean enter the encoder.

    The caller must fill unobserved physical residuals before representation
    encoding. In particular RGB missing fill is F(0), not RGB (0,0,0).
    Cloud-opacity/affected masks and reference target pixels are not inputs.
    """

    def __init__(self, channels: int, mean_scale: float, mean_center: float):
        super().__init__()
        if channels not in (1, 3) or mean_scale <= 0:
            raise ValueError("Expected 1 or 3 channels and positive mean scale")
        self.channels, self.mean_scale, self.mean_center = channels, mean_scale, mean_center
        self.projection = nn.Conv2d(channels + 3, channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        with torch.no_grad():
            for c in range(channels):
                self.projection.weight[c, c, 0, 0] = 1

    def forward(self, encoded_input: Tensor, reference_valid: Tensor,
                observed_valid: Tensor, mean_celsius: Tensor) -> Tensor:
        if encoded_input.ndim != 4 or encoded_input.shape[1] != self.channels:
            raise ValueError("Encoded input channel/shape mismatch")
        r = _shape_mask(reference_valid, encoded_input, "reference_valid")
        o = _shape_mask(observed_valid, encoded_input, "observed_valid")
        if bool((o & ~r).any()):
            raise ValueError("Observed support must be a subset of reference support")
        b, _, h, w = encoded_input.shape
        mu = _normalized_mean(mean_celsius, b, self.mean_scale, self.mean_center, encoded_input.device)
        plane = mu.reshape(b, 1, 1, 1).expand(-1, -1, h, w).to(encoded_input.dtype)
        return self.projection(torch.cat([encoded_input, r.to(encoded_input.dtype),
                                          o.to(encoded_input.dtype), plane], dim=1))


class MeanAffine(nn.Module):
    """Zero-initialized scalar-mean affine branch after the first decoder layer."""

    def __init__(self, features: int, scale: float, center: float):
        super().__init__()
        self.scale, self.center = scale, center
        self.affine = nn.Linear(1, 2 * features)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features: Tensor, mean: Tensor) -> Tensor:
        mu = _normalized_mean(mean, features.shape[0], self.scale, self.center, features.device)
        gain, bias = self.affine(mu.to(self.affine.weight.dtype)).chunk(2, -1)
        if features.ndim == 4:
            gain, bias = gain[:, :, None, None], bias[:, :, None, None]
        elif features.ndim == 3:
            gain, bias = gain[:, None, :], bias[:, None, :]
        else:
            raise ValueError("MeanAffine supports BCHW or BND features")
        return features * (1 + gain.to(features.dtype)) + bias.to(features.dtype)


def _stable_module_initialization(module: nn.Module, seed: int, prefix: str) -> None:
    """Match equal-shaped scratch weights independently of quantizer/channel RNG consumption."""
    for name, child in module.named_modules():
        if isinstance(child, (nn.Conv2d, nn.Linear, nn.GroupNorm, nn.LayerNorm)):
            digest = hashlib.sha256(f"{seed}:{prefix}:{name}".encode()).digest()
            local_seed = int.from_bytes(digest[:8], "little") % (2 ** 63 - 1)
            # Construction is CPU-only; move to a device only after building.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(local_seed)
                child.reset_parameters()


def _zero_output_head(head: nn.Module, channels: int, representation: str, patch_area: int = 1) -> None:
    values = [0.0] * channels if representation in ("R", "G", "G_power", "G_asinh") else [-1.0, 1.0, 0.0]
    if len(values) != channels:
        raise ValueError("Representation and output channels are inconsistent")
    nn.init.zeros_(head.weight)
    with torch.no_grad():
        head.bias.copy_(head.bias.new_tensor(values * patch_area))


def _checkpoint_call(module: nn.Module, value: Tensor, enabled: bool) -> Tensor:
    return checkpoint(module, value, use_reentrant=False) if enabled and torch.is_grad_enabled() else module(value)


class _MetadataMixin:
    def model_metadata(self) -> dict[str, Any]:
        parts = {}
        for name, module in self.named_children():
            parts[name] = sum(p.numel() for p in module.parameters())
        return {"kind": self.kind, "parameters_total": sum(p.numel() for p in self.parameters()),
                "parameters_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "parameters_by_component": parts, "runtime_verified": False, **self.rate()}


class ConvolutionalRestorationCodec(_MetadataMixin, nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        from eo_denoise.var_basic import Encoder, Decoder
        self.config = config
        c = config["model"]
        self.kind = c["kind"]
        self.channels, self.grid = int(c["channels"]), int(c.get("grid", 16))
        if self.grid != 16:
            raise ValueError("The frozen 5-stage VAR backbone maps 256 to a 16x16 grid")
        base = int(c.get("base_channels", 128))
        if base % 32:
            raise ValueError("VAR GroupNorm requires base_channels divisible by 32")
        self.checkpointing = bool(c.get("checkpointing", False))
        self.latent_dim = int(c.get("latent_dim", 32))
        task = config.get("task", {})
        scale, center = float(task.get("mean_scale_celsius", 16.29419157454727)), float(task.get("mean_center_celsius", 0.0))
        architecture = dict(ch=base, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=int(c.get("num_res_blocks", 3)),
                            dropout=0.0, in_channels=self.channels, z_channels=self.latent_dim,
                            using_sa=True, using_mid_sa=True)
        self.encoder = Encoder(double_z=False, **architecture)
        self.decoder = Decoder(**architecture)
        self.quant_conv = nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1)
        self.post_quant_conv = nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1)
        seed = int(config.get("seed", 17))
        for name in ("encoder", "decoder", "quant_conv", "post_quant_conv"):
            _stable_module_initialization(getattr(self, name), seed, name)
        self.input_adapter = ObservationAdapter(self.channels, scale, center)
        self.decoder_mean = MeanAffine(base * 4, scale, center)
        # Quantizer construction occurs after all shared scratch components.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 733)
            self.quantizer = build_quantizer(c)
        _zero_output_head(self.decoder.conv_out, self.channels, config["representation"]["kind"])

    def _encode(self, x: Tensor) -> Tensor:
        e, check = self.encoder, self.training and self.checkpointing
        h = e.conv_in(x)
        for level in range(e.num_resolutions):
            for i, block in enumerate(e.down[level].block):
                h = _checkpoint_call(block, h, check)
                if len(e.down[level].attn):
                    h = _checkpoint_call(e.down[level].attn[i], h, check)
            if level != e.num_resolutions - 1:
                h = e.down[level].downsample(h)
        for module in (e.mid.block_1, e.mid.attn_1, e.mid.block_2):
            h = _checkpoint_call(module, h, check)
        return self.quant_conv(e.conv_out(F.silu(e.norm_out(h))))

    def _decode(self, latent: Tensor, mean: Tensor) -> Tensor:
        d, check = self.decoder, self.training and self.checkpointing
        h = self.decoder_mean(d.conv_in(self.post_quant_conv(latent)), mean)
        for module in (d.mid.block_1, d.mid.attn_1, d.mid.block_2):
            h = _checkpoint_call(module, h, check)
        for level in reversed(range(d.num_resolutions)):
            for i, block in enumerate(d.up[level].block):
                h = _checkpoint_call(block, h, check)
                if len(d.up[level].attn):
                    h = _checkpoint_call(d.up[level].attn[i], h, check)
            if level != 0:
                h = d.up[level].upsample(h)
        return d.conv_out(F.silu(d.norm_out(h))).float()

    def forward(self, encoded_input: Tensor, reference_valid: Tensor, observed_valid: Tensor,
                mean_celsius: Tensor, patch_missing: Tensor | None = None) -> dict[str, Any]:
        if encoded_input.shape[-2:] != (256, 256):
            raise ValueError("This protocol requires 256x256 inputs")
        latent = self._encode(self.input_adapter(encoded_input, reference_valid, observed_valid, mean_celsius))
        if self.quantizer is None:
            quantized, indices, loss, diagnostics = latent, None, latent.new_zeros(()), {}
        else:
            result = self.quantizer(latent)
            quantized, indices, loss = result["quantized"], result["indices"], result["loss"]
            diagnostics = result["diagnostics"]
        return {"encoded_prediction": self._decode(quantized, mean_celsius), "indices": indices,
                "bottleneck_loss": loss, "kl": latent.new_zeros(()), "latent": quantized,
                "prequantized_latent": latent, "quantizer_diagnostics": diagnostics}

    def decode_indices(self, indices: Tensor, mean_celsius: Tensor) -> Tensor:
        if self.quantizer is None:
            raise ValueError("Continuous AE has no discrete indices")
        return self._decode(self.quantizer.decode_indices(indices), mean_celsius)

    def rate(self) -> dict[str, Any]:
        if self.quantizer is None:
            elements = self.latent_dim * self.grid ** 2
            return {"latent_grid": [self.grid, self.grid], "latent_elements": elements,
                    "payload_type": "continuous_float32_reference", "latent_float32_bits": elements * 32,
                    "mean_side_information_bits": 32, "equal_rate_claim": False}
        tokens = sum(v * v for v in self.quantizer.scales) if self.kind == "var_multiscale" else self.grid ** 2
        size = self.quantizer.codebook_size
        bits = math.ceil(math.log2(size))
        return {"latent_grid": [self.grid, self.grid], "latent_channels": self.latent_dim,
                "indices_per_image": tokens, "codebook_size": size, "fixed_bits_per_index": bits,
                "nominal_index_bits": tokens * bits, "mean_side_information_bits": 32,
                "nominal_index_plus_mean_bits": tokens * bits + 32,
                "excludes": ["mask/support metadata", "container headers", "shared weights"],
                "actual_bitstream_measured": False, "payload_type": "discrete_fixed_width_nominal"}


def verify_local_pretrained(model_config: dict[str, Any]) -> Path:
    location = model_config.get("pretrained_path")
    if not location:
        raise FileNotFoundError("Local pretrained_path is required; this builder never downloads weights")
    root = Path(location).resolve()
    if not root.is_dir():
        raise ValueError("Expected a diffusers-format local directory, not raw ae.safetensors")
    expected = model_config.get("pretrained_sha256", {})
    if not isinstance(expected, dict) or not expected:
        raise ValueError("A nonempty pretrained_sha256 map is required")
    needed = {p.relative_to(root).as_posix() for p in root.rglob("*.safetensors")}
    needed.update(p.relative_to(root).as_posix() for p in root.rglob("*.safetensors.index.json"))
    needed.add("config.json")
    if len(needed) == 1 or not needed.issubset(expected):
        raise ValueError("Hash manifest must cover config.json and every local safetensors weight file")
    for name, digest in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid pretrained manifest path: {name}")
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                sha.update(part)
        if sha.hexdigest() != digest.lower():
            raise ValueError(f"Pretrained SHA256 mismatch: {name}")
    return root


class PretrainedKLRestorationCodec(_MetadataMixin, nn.Module):
    """SD, SDXL, or verified diffusers-converted FLUX.1 VAE, local-only.

    Autoencoder scaling/shift belongs to the diffusion interface and is not
    applied between encode and decode here. Evaluation uses posterior.mode;
    training uses posterior.sample. All trainable parameters remain enabled.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        c = config["model"]
        self.kind, self.config = c["kind"], config
        if int(c["channels"]) != 3:
            raise ValueError("Natural pretrained VAEs require G/C/C_affine 3-channel representation")
        root = verify_local_pretrained(c)
        with (root / "config.json").open(encoding="utf-8") as handle:
            source_config = json.load(handle)
        latent_channels = 16 if self.kind == "flux1" else 4
        if int(source_config.get("latent_channels", -1)) != latent_channels:
            raise ValueError("Pretrained latent channels do not match the declared model family")
        if source_config.get("_class_name") != "AutoencoderKL":
            raise ValueError("This adapter explicitly supports diffusers AutoencoderKL format only")
        if source_config.get("in_channels") != 3 or source_config.get("out_channels") != 3:
            raise ValueError("Pretrained RGB interface mismatch")
        if list(source_config.get("block_out_channels", [])) != [128, 256, 512, 512]:
            raise ValueError("Unexpected pretrained architecture; requires a separately reviewed adapter")
        if int(source_config.get("layers_per_block", 2)) != 2 or source_config.get("act_fn", "silu") != "silu":
            raise ValueError("Unexpected pretrained block configuration")
        if int(source_config.get("norm_num_groups", 32)) != 32:
            raise ValueError("This adapter expects the official group-normalized decoder")
        if self.kind in ("sd", "sdxl"):
            expected_scale = 0.18215 if self.kind == "sd" else 0.13025
            # The pinned official SD diffusers-0.4.2 config omits this field;
            # AutoencoderKL's documented/default value is 0.18215. SDXL must
            # still explicitly identify its different scaling convention.
            default_scale = 0.18215 if self.kind == "sd" else 0.0
            if not math.isclose(float(source_config.get("scaling_factor", default_scale)), expected_scale, rel_tol=1e-6):
                raise ValueError(f"Expected {self.kind} scaling_factor={expected_scale}; check family/source identity")
        if self.kind == "flux1":
            if not math.isclose(float(source_config.get("scaling_factor", 0)), 0.3611, rel_tol=1e-6):
                raise ValueError("Expected FLUX.1 scaling_factor=0.3611")
            if not math.isclose(float(source_config.get("shift_factor", 0)), 0.1159, rel_tol=1e-6):
                raise ValueError("Expected FLUX.1 shift_factor=0.1159")
            if source_config.get("use_quant_conv", True) or source_config.get("use_post_quant_conv", True):
                raise ValueError("FLUX.1 converted VAE must omit quant/post-quant convolutions")
        from diffusers import AutoencoderKL
        self.vae, loading = AutoencoderKL.from_pretrained(
            str(root), local_files_only=True, use_safetensors=True,
            torch_dtype=torch.float32, low_cpu_mem_usage=False, output_loading_info=True)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Official VAE weights did not load strictly: {loading}")
        self.strict_loading_info = loading
        if int(self.vae.config.latent_channels) != latent_channels:
            raise ValueError("Loaded VAE config changed latent channels")
        self.latent_channels = latent_channels
        self.checkpointing = bool(c.get("checkpointing", False))
        if self.checkpointing:
            self.vae.enable_gradient_checkpointing()
        task = config.get("task", {})
        scale, center = float(task.get("mean_scale_celsius", 16.29419157454727)), float(task.get("mean_center_celsius", 0.0))
        self.input_adapter = ObservationAdapter(3, scale, center)
        self.decoder_mean = MeanAffine(self.vae.decoder.conv_in.out_channels, scale, center)
        # Keep official output head and all original weights intact.
        self.source_sha256 = dict(c["pretrained_sha256"])

    def _decode(self, z: Tensor, mean: Tensor) -> Tensor:
        # This follows diffusers' group-normalized Decoder path and inserts
        # only the declared zero-initialized mean affine after conv_in.
        d = self.vae.decoder
        if getattr(self.vae, "post_quant_conv", None) is not None:
            z = self.vae.post_quant_conv(z)
        h = self.decoder_mean(d.conv_in(z), mean)
        check = self.training and self.checkpointing
        h = _checkpoint_call(d.mid_block, h, check)
        h = h.to(next(d.up_blocks.parameters()).dtype)
        for block in d.up_blocks:
            h = _checkpoint_call(block, h, check)
        return d.conv_out(d.conv_act(d.conv_norm_out(h))).float()

    def forward(self, encoded_input: Tensor, reference_valid: Tensor, observed_valid: Tensor,
                mean_celsius: Tensor, patch_missing: Tensor | None = None) -> dict[str, Any]:
        if encoded_input.shape[-2:] != (256, 256):
            raise ValueError("This protocol requires 256x256 inputs")
        x = self.input_adapter(encoded_input, reference_valid, observed_valid, mean_celsius)
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample() if self.training else posterior.mode()
        with torch.autocast(device_type=z.device.type, enabled=False):
            mean, logvar = posterior.mean.float(), posterior.logvar.float()
            kl = 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).mean()
        return {"encoded_prediction": self._decode(z, mean_celsius), "indices": None,
                "bottleneck_loss": z.new_zeros(()), "kl": kl, "latent": z}

    def decode_indices(self, indices: Tensor, mean_celsius: Tensor) -> Tensor:
        raise ValueError("KL-VAE has continuous latent values, not discrete indices")

    def rate(self) -> dict[str, Any]:
        elements = self.latent_channels * 32 * 32
        return {"latent_grid": [32, 32], "latent_channels": self.latent_channels,
                "latent_elements": elements, "payload_type": "continuous_float32_reference",
                "latent_float32_bits": 32 * elements, "mean_side_information_bits": 32,
                "equal_rate_claim": False, "posterior_evaluation": "mode"}


class PrithviSpatialMAE(_MetadataMixin, nn.Module):
    """Actual visible-token encoder and mask-token decoder, no VQ or KL.

    Uses the existing pinned Prithvi-EO-2.0 reference components. The 2D,
    scratch, reduced-depth architecture is an adaptation, not official 300M
    weights. Pixel/cloud/sensor primary comparisons must not silently route
    through this independent patch-completion model.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        from eo_denoise.prithvi_reference import PrithviViT, MAEDecoder
        from eo_denoise.prithvi_spatial import spatial_position
        c, task = config["model"], config.get("task", {})
        if c.get("pretrained_path"):
            raise ValueError("This spatial MAE adaptation is scratch-only; official weights require a separate adapter")
        self.kind, self.channels = "prithvi_mae", int(c["channels"])
        self.config, self.checkpointing = config, bool(c.get("checkpointing", False))
        dim, depth, heads = int(c.get("encoder_dim", 768)), int(c.get("encoder_depth", 9)), int(c.get("encoder_heads", 12))
        ddim, ddepth, dheads = int(c.get("decoder_dim", 512)), int(c.get("decoder_depth", 7)), int(c.get("decoder_heads", 8))
        if dim % 16 or ddim % 16 or dim % heads or ddim % dheads:
            raise ValueError("Prithvi dimensions must support spatial positions and attention heads")
        self.encoder_dim = dim
        self.encoder = PrithviViT(img_size=256, patch_size=(1, 16, 16), num_frames=1,
                                  in_chans=self.channels, embed_dim=dim, depth=depth,
                                  num_heads=heads, mlp_ratio=4, coords_encoding=[], drop_path=0)
        self.encoder.patch_embed.proj = nn.Conv2d(self.channels, dim, 16, stride=16)
        self.encoder.pos_embed = spatial_position(dim)
        self.decoder = MAEDecoder(patch_size=(1, 16, 16), grid_size=(1, 16, 16), in_chans=self.channels,
                                  encoder_embed_dim=dim, decoder_embed_dim=ddim, depth=ddepth,
                                  num_heads=dheads, mlp_ratio=4, coords_encoding=[])
        self.decoder.decoder_pos_embed = spatial_position(ddim)
        seed = int(config.get("seed", 17))
        _stable_module_initialization(self.encoder, seed, "mae_encoder")
        _stable_module_initialization(self.decoder, seed, "mae_decoder")
        scale, center = float(task.get("mean_scale_celsius", 16.29419157454727)), float(task.get("mean_center_celsius", 0.0))
        self.input_adapter = ObservationAdapter(self.channels, scale, center)
        self.decoder_mean = MeanAffine(ddim, scale, center)
        _zero_output_head(self.decoder.decoder_pred, self.channels,
                          config["representation"]["kind"], patch_area=16 * 16)

    @staticmethod
    def _patch_mask(mask: Tensor | None, x: Tensor, r: Tensor, o: Tensor) -> Tensor:
        b = x.shape[0]
        if mask is None:
            missing = torch.zeros((b, 256), dtype=torch.bool, device=x.device)
        elif mask.shape == (b, 256):
            missing = mask.bool()
        elif mask.shape == (b, 16, 16):
            missing = mask.reshape(b, 256).bool()
        elif mask.shape == (b, 1, 256, 256):
            blocks = mask.bool().reshape(b, 1, 16, 16, 16, 16).permute(0, 2, 4, 1, 3, 5).reshape(b, 256, 256)
            if not torch.equal(blocks.any(-1), blocks.all(-1)):
                raise ValueError("MAE requires whole 16x16 patches, not random missing pixels")
            missing = blocks.all(-1)
        else:
            raise ValueError("patch_missing must be Bx256, Bx16x16 or Bx1x256x256")
        expanded = missing.reshape(b, 1, 16, 16).repeat_interleave(16, 2).repeat_interleave(16, 3)
        if not torch.equal(r & ~o, r & expanded):
            raise ValueError("Observed mask and supplied whole-patch mask disagree")
        counts = missing.sum(1)
        if bool((counts != counts[0]).any()) or int(counts[0]) >= 256:
            raise ValueError("Batched MAE requires equal missing patch counts and at least one visible patch")
        return missing

    def forward(self, encoded_input: Tensor, reference_valid: Tensor, observed_valid: Tensor,
                mean_celsius: Tensor, patch_missing: Tensor | None = None) -> dict[str, Any]:
        if encoded_input.shape[-2:] != (256, 256):
            raise ValueError("MAE requires frozen 256x256 images")
        r = _shape_mask(reference_valid, encoded_input, "reference_valid")
        o = _shape_mask(observed_valid, encoded_input, "observed_valid")
        missing = self._patch_mask(patch_missing, encoded_input, r, o)
        x = self.input_adapter(encoded_input, r, o, mean_celsius)
        patches = self.encoder.patch_embed.proj(x).flatten(2).transpose(1, 2)
        patches = patches + self.encoder.pos_embed[:, 1:].to(patches.dtype)
        # Visible tokens stay in raster order; absent tokens never enter any encoder block.
        order = torch.argsort(missing.long() * 256 + torch.arange(256, device=x.device), dim=1)
        keep_count = 256 - int(missing[0].sum())
        keep = order[:, :keep_count]
        visible = torch.gather(patches, 1, keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
        cls = (self.encoder.cls_token + self.encoder.pos_embed[:, :1]).to(visible.dtype).expand(x.shape[0], -1, -1)
        latent = torch.cat([cls, visible], dim=1)
        check = self.training and self.checkpointing
        for block in self.encoder.blocks:
            latent = _checkpoint_call(block, latent, check)
        latent = self.encoder.norm(latent)
        projected = self.decoder.decoder_embed(latent)
        restored = self.decoder.mask_token.to(projected.dtype).expand(x.shape[0], 256, -1).clone()
        restored.scatter_(1, keep.unsqueeze(-1).expand(-1, -1, projected.shape[-1]), projected[:, 1:])
        decoded = torch.cat([projected[:, :1], restored], dim=1)
        decoded = self.decoder_mean(decoded, mean_celsius) + self.decoder.decoder_pos_embed.to(decoded.dtype)
        for block in self.decoder.decoder_blocks:
            decoded = _checkpoint_call(block, decoded, check)
        pixels = self.decoder.decoder_pred(self.decoder.decoder_norm(decoded))[:, 1:]
        prediction = pixels.reshape(x.shape[0], 16, 16, 16, 16, self.channels).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], self.channels, 256, 256)
        return {"encoded_prediction": prediction.float(), "indices": None,
                "bottleneck_loss": latent.new_zeros(()), "kl": latent.new_zeros(()),
                "latent": latent, "mae_patch_missing": missing, "mae_visible_patch_indices": keep}

    def decode_indices(self, indices: Tensor, mean_celsius: Tensor) -> Tensor:
        raise ValueError("Native MAE uses continuous visible-token states, not discrete indices")

    def rate(self) -> dict[str, Any]:
        return {"payload_type": "continuous_visible_mae_tokens", "patch_size": 16,
                "full_patch_tokens": 256, "cls_tokens": 1, "feature_dim": self.encoder_dim,
                "visible_token_count_depends_on_mask": True, "mean_side_information_bits": 32,
                "decoder_also_requires_patch_positions": True, "equal_rate_claim": False}


def build_model(config: dict[str, Any]) -> nn.Module:
    """Construct the declared model only; never substitute a simpler baseline."""
    kind = config["model"]["kind"]
    if kind in ("vq", "fsq", "lfq", "bsq", "ae", "var_multiscale"):
        return ConvolutionalRestorationCodec(config)
    if kind in ("sd", "sdxl", "flux1"):
        return PretrainedKLRestorationCodec(config)
    if kind == "eo_vae":
        from .geo_models import EOVAEAdapter
        return EOVAEAdapter(config)
    if kind == "terramind_fsq":
        from .geo_models import TerraMindAdapter
        return TerraMindAdapter(config)
    if kind == "prithvi_mae":
        return PrithviSpatialMAE(config)
    if kind == "satmae_pp":
        from .satmae import SatMAEPlusPlusLST
        return SatMAEPlusPlusLST(config)
    raise ValueError(f"Unsupported model kind: {kind}")
