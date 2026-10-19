"""Official EO-VAE / TerraMind tokenizer adapters; explicit local dependencies.

No weights or Python packages are downloaded on import or construction. These
are LST architecture adaptations trained from scratch, with no pretrained weights.
TerraMind retains its diffusion decoder and is NOT a one-pass convolutional AE.
"""
import importlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn

from .config import file_digest
from .models import ObservationAdapter, MeanAffine, _MetadataMixin


def _official_module(c, module_name, expected_entry_hash):
    location = c.get("source_root")
    manifest = c.get("source_sha256", {})
    if not location or not manifest:
        raise FileNotFoundError("Bind a local pinned official source checkout and full Python SHA256 manifest first")
    root = Path(location).resolve()
    package = module_name.split(".")[0]
    actual_files = {str(p.relative_to(root)).replace("\\", "/") for p in (root / package).rglob("*.py")}
    if set(manifest) != actual_files:
        raise ValueError("Source manifest must cover the entire official Python package")
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_digest(path) != expected:
            raise ValueError("Official source checksum mismatch: " + relative)
    relative = module_name.replace(".", "/") + ".py"
    if manifest.get(relative) != expected_entry_hash:
        raise ValueError("Adapter was reviewed against a different official entry point")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != (root / relative).resolve():
        raise ValueError("Another installed package shadows the pinned source")
    return module


def _helpers(owner, config, latent_channels):
    c, task = config["model"], config["task"]
    owner.input_adapter = ObservationAdapter(c["channels"], task["mean_scale_celsius"], task["mean_center_celsius"])
    owner.decoder_mean = MeanAffine(latent_channels, task["mean_scale_celsius"], task["mean_center_celsius"])


class EOVAEAdapter(_MetadataMixin, nn.Module):
    """EO-VAE body/posterior adapted to fixed LST input/output channels.

    Uses the official fixed-channel option instead of assigning LST a fictitious
    optical wavelength. This removes multi-sensor dynamic heads deliberately;
    it is an EO-VAE-LST architecture adaptation, not a full-paper replication.
    """
    def __init__(self, config):
        super().__init__()
        self.kind, self.config = "eo_vae", config
        c = config["model"]
        if c.get("initialization") != "scratch" or c.get("use_dynamic_ops") is not False:
            raise ValueError("EO-VAE LST adapter requires scratch initialization and fixed-channel heads")
        evidence = json.loads((Path(__file__).parent / "geo_sources/manifest.json").read_text(encoding="utf-8"))
        checksum = next(x["sha256"] for x in evidence if x["file"] == "eo_new_autoencoder.py.txt")
        source = _official_module(c, "eo_vae.models.new_autoencoder", checksum)
        common = dict(resolution=256, ch=c["base_channels"], ch_mult=c["ch_mult"],
                      num_res_blocks=c["num_res_blocks"], z_channels=c["latent_dim"], use_dynamic_ops=False)
        self.vae = source.EOFluxVAE(encoder=source.Encoder(in_channels=c["channels"], **common),
            decoder=source.Decoder(out_ch=c["channels"], **common), loss_fn=nn.Identity(),
            ckpt_path=None, freeze_body=False)
        self.z_channels = int(self.vae.encoder.z_channels)
        self.latent_grid = 256 // (2 ** (self.vae.encoder.num_resolutions - 1))
        _helpers(self, config, self.z_channels)

    def forward(self, encoded_input, reference_valid, observed_valid, mean_celsius, patch_missing=None):
        x = self.input_adapter(encoded_input, reference_valid, observed_valid, mean_celsius)
        posterior = self.vae.encode(x, None)
        z = posterior.sample() if self.training else posterior.mode()
        # Raw posterior/decoder path: avoids silently double-applying FLUX BN or
        # invoking @no_grad inference helpers during fine-tuning.
        prediction = self.vae.decode_raw(self.decoder_mean(z, mean_celsius), None)
        with torch.autocast(device_type=z.device.type, enabled=False):
            mean, logvar = posterior.mean.float(), posterior.logvar.float()
            kl = .5 * (mean.square() + logvar.exp() - 1. - logvar).mean()
        return {"encoded_prediction": prediction.float(), "indices": None, "latent": z,
                "bottleneck_loss": z.new_zeros(()), "kl": kl}

    def decode_indices(self, indices, mean_celsius):
        raise ValueError("EO-VAE has a continuous posterior, no discrete codebook")

    def rate(self):
        return {"payload_type": "continuous_posterior", "latent_channels": self.z_channels,
                "latent_grid": [self.latent_grid, self.latent_grid], "latent_elements": self.z_channels * self.latent_grid ** 2,
                "mean_side_information_bits": 32, "equal_rate_claim": False,
                "adaptation": "scratch fixed-channel EO-VAE body; raw posterior decode; shared physical loss, no dynamic wavelength heads or GAN/EQ recipe"}


class TerraMindAdapter(_MetadataMixin, nn.Module):
    """TerraMind ViT + FSQ + diffusion decoder, adapted and trained on LST from scratch.

    Training conditions the encoder exclusively on corrupted observations. Only
    the diffusion decoder sees a noised training target at random t, as required
    for diffusion training. Evaluation starts from fixed public Gaussian noise,
    never from the target, and uses the registered full sampling step count.
    """
    def __init__(self, config):
        super().__init__()
        self.kind, self.config = config["model"]["kind"], config
        c = config["model"]
        if c.get("initialization") != "scratch":
            raise ValueError("LST TerraMind architecture comparison must train from scratch")
        evidence = json.loads((Path(__file__).parent / "geo_sources/manifest.json").read_text(encoding="utf-8"))
        checksum = next(x["sha256"] for x in evidence if x["file"] == "tm_vqvae.py.txt")
        source = _official_module(c, "terratorch.models.backbones.terramind.tokenizer.vqvae", checksum)
        levels = c["levels"]
        self.tokenizer = source.DiVAE(image_size=256, n_channels=c["channels"], enc_type="vit_b_enc",
            dec_type="unet_patched", post_mlp=True, patch_size=16, quant_type="fsq",
            codebook_size="-".join(map(str, levels)), latent_dim=len(levels),
            prediction_type="sample", num_train_timesteps=1000, scheduler="ddim",
            clip_sample=True, thresholding=True, zero_terminal_snr=True,
            freeze_enc=False, ckpt_path=None, sync_codebook=False)
        self.tokenizer.requires_grad_(True)
        self.codebook_size = math.prod(levels)
        _helpers(self, config, len(levels))

    def _encode(self, inputs):
        x = self.input_adapter(inputs["encoded_input"], inputs["reference_valid"],
                               inputs["observed_valid"], inputs["mean_celsius"])
        return self.tokenizer.encode(x)

    def training_prediction(self, inputs, encoded_target):
        quant, native, indices = self._encode(inputs)
        t = torch.randint(0, 1000, (quant.shape[0],), device=quant.device)
        noise = torch.randn_like(encoded_target)
        noisy_target = self.tokenizer.noise_scheduler.add_noise(encoded_target, noise, t)
        prediction = self.tokenizer.decoder(noisy_target, t, self.decoder_mean(quant, inputs["mean_celsius"]),
                                            cond_mask=None, orig_res=None)
        return {"encoded_prediction": prediction.float(), "indices": indices, "latent": quant,
                "bottleneck_loss": native.mean(), "kl": quant.new_zeros(()),
                "training_prediction_type": "diffusion_x0_at_random_t_not_full_restoration"}

    def _decode(self, quant, mean):
        predictions = []
        # Per-image fixed seed makes output independent of batch composition and
        # exactly replayable from indices + mean + public decoder settings.
        for i in range(quant.shape[0]):
            generator = torch.Generator(device=quant.device).manual_seed(self.config["model"]["decoder_seed"])
            conditioned = self.decoder_mean(quant[i:i+1], mean[i:i+1])
            predictions.append(self.tokenizer.decode_quant(conditioned, timesteps=self.config["model"]["decoder_steps"],
                                generator=generator, image_size=256, verbose=False))
        return torch.cat(predictions).float()

    def forward(self, encoded_input, reference_valid, observed_valid, mean_celsius, patch_missing=None):
        if self.training:
            raise RuntimeError("TerraMind training must use training_prediction, not backprop through the inference sampler")
        inputs = dict(encoded_input=encoded_input, reference_valid=reference_valid,
                      observed_valid=observed_valid, mean_celsius=mean_celsius)
        quant, native, indices = self._encode(inputs)
        return {"encoded_prediction": self._decode(quant, mean_celsius), "indices": indices,
                "latent": quant, "bottleneck_loss": native.mean(), "kl": quant.new_zeros(())}

    def decode_indices(self, indices, mean_celsius):
        return self._decode(self.tokenizer.tokens_to_embedding(indices), mean_celsius)

    def rate(self):
        bits = math.ceil(math.log2(self.codebook_size))
        return {"payload_type": "discrete_fsq_with_diffusion_decoder", "latent_grid": [16, 16],
                "indices_per_image": 256, "codebook_size": self.codebook_size, "fixed_bits_per_index": bits,
                "nominal_index_plus_mean_bits": 256 * bits + 32, "mean_side_information_bits": 32,
                "decoder_steps": self.config["model"]["decoder_steps"], "equal_parameter_claim": False,
                "decoder_noise_seed_public": self.config["model"]["decoder_seed"],
                "excludes": ["container", "support masks", "shared weights"],
                "adaptation": "scratch LST architecture adaptation; FSQ resized to 1024 states; full 50-step diffusion decoder retained"}
