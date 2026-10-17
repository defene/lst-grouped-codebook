# Source provenance and adaptations

VAR commit: `78b95394fc5896192e3a003e4b295f8ea743c48f`. Unmodified downloaded sources and hashes remain under `../ablation_vq_prithvi/sources/var/`. The MIT notice is preserved as `eo_denoise/VAR_LICENSE.txt`.

| Upstream | Runtime copy | Changes |
|---|---|---|
| models/vqvae.py | eo_denoise/var_reference.py | Relative import names only |
| models/basic_vae.py | eo_denoise/var_basic.py | Nearest 2× upsampling uses equivalent repeat_interleave for deterministic backward |
| models/quant.py | eo_denoise/var_quant.py | Relative imports; uninitialized distributed world size defaults to one; modern autocast; deterministic bicubic resize |
| dist.py | eo_denoise/var_dist.py | None |

`var_ops.py` implements align_corners=False bicubic interpolation with the same a=-0.75 cubic kernel, using separable matrix multiplication for deterministic CUDA gradients. Tests compare all nine coarse-scale outputs and gradients with PyTorch's reference interpolation, and compare the complete small VQVAE forward/loss with the pinned unmodified source. `var_codec.py` adapts channels/mask and exposes train/eval interfaces; it retains VAR's actual convolutional blocks and residual quantizer. Its legacy `var_prithvi` option is not selected by any configuration in A02.

Prithvi revision: `9eb1b1102806593963daa333bcc491b1c6f8562f`, Apache-2.0. Runtime source is `eo_denoise/prithvi_reference.py`; license is `PRITHVI_LICENSE.txt`. `prithvi_continuous.py` constructs the reference ViT and decoder at the documented smaller sizes, substitutes spatial-only patch embedding/positions, and sends the native continuous encoder tokens directly through the decoder. A test compares its reconstruction against the reference decoder with all patch positions restored in identity order. There is no codebook module or VQ/KL objective in this branch.

Historical A01 and cancelled V2 runs retain their own source archives. Current-code checkpoint loading intentionally rejects mismatched framework source hashes; it does not reinterpret old checkpoints with the corrected architecture.
