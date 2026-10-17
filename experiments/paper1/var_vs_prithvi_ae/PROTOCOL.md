# A02 — VAR ResNet VQ-VAE versus continuous spatial Prithvi AE

This is the corrected experiment requested on 2026-09-10. Prithvi uses its ViT encoder and decoder with continuous features. It has no vector quantizer, codebook, VQ loss, Gaussian sampling, or KL loss. The cancelled `var_aligned_v2` experiment incorrectly quantized both branches and must not be used as A02 results.

## Models

| Branch | Encoder → representation → decoder | HLS parameters | LST parameters |
|---|---|---:|---:|
| VAR VQ-VAE | Official ResNet-style encoder → official multiscale residual VQ → official ResNet-style decoder | 108,960,022 | 108,945,617 |
| Prithvi AE | Spatial ViT → 256 continuous patch vectors + CLS, dimension 768 → spatial ViT decoder | 108,900,608 | 107,260,928 |

VAR source: [FoundationVision/VAR](https://github.com/FoundationVision/VAR/tree/78b95394fc5896192e3a003e4b295f8ea743c48f), MIT. Actual `VQVAE`/`Encoder`/`Decoder`/`VectorQuantizer2` classes are vendored in `eo_denoise/var_*.py`, not a separately implemented CNN. Released-model configuration: ch=160, latent=32, vocabulary=4096, scales=(1,2,3,4,5,6,8,10,13,16), four shared residual transforms, beta=.25. There are 680 indices per image. EO adaptations add the input validity mask channel and use six output channels for HLS or one for LST. There is no output clipping. Single-process distributed fallback, modern autocast, and numerically equivalent deterministic resize implementations are documented in `SOURCE_ADAPTATIONS.md`.

Prithvi source: [IBM–NASA Prithvi EO 2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/tree/9eb1b1102806593963daa333bcc491b1c6f8562f), Apache-2.0. Encoder dimension 768, depth 11, heads 12; decoder dimension 512, depth 9, heads 8; MLP ratio 4. Runtime patch embedding is Conv2d with patch size 16, and positional encoding is spatial XY only. No time or geolocation input, temporal attention, MAE token dropping, or auxiliary projection to a small latent dimension is used. The native CLS token accompanies the continuous patch features. Both branches start from random weights; no pretrained weight files are loaded.

Parameter size, input data, effective batch, optimizer schedule, and evaluation conditions are comparable. **Compression rates are not matched.** VAR transmits discrete indices; Prithvi passes continuous vectors. This comparison changes both the backbone and representation type and cannot isolate the causal effect of VQ alone. Prithvi is an AE here, not a Gaussian VAE.

## Frozen data and noise

Only the independently processed Prithvi dataset is used: 230,659 train records / 2,943 tiles, 11,757 validation records / 151 tiles. Each HLS–LST pair retains its original one-to-one record ID. HLS and LST train independent models. LSTSR is not merged. Input arrays are read directly from `artifacts/prithvi_data_v1_20260909.zip`; no reprocessing or decompression to a second dataset.

HLS uses six reflectance bands and the frozen train-only normalization. LST storage is Celsius × 100; physical evaluation divides by 100 and does not apply Landsat DN conversion again. The original masks and normalized values are unchanged. All architectures receive the same image plus input mask, and clean reference pixels supply reconstruction targets.

Eight runs = {HLS,LST} × {VAR VQ-VAE,Prithvi AE} × {clean input,10% synthetic missing-region input}. Noise uses the existing split-specific QA shape banks, with no extra clean mixture. The noisy condition is known-mask missing-region restoration, not additive Gaussian noise. Masks are deterministic by sample, epoch and view and identical across architectures. Training uses one view; stability and cross-view losses are zero. Both models use sample-equal normalized MSE + 0.1 gradient MSE. Only VAR adds its native VQ objective.

## Budgets and parallel execution

Small gate: the same fixed 64 train / 64 validation records; 128 optimizer steps, microbatch 8, accumulation 2, effective batch 16, warmup 16. All eight runs must finish, losses must be finite and reconstruction must improve by at least 10% between the first and last 16 steps. Prithvi must have exactly zero VQ/KL loss; validation must use the same IDs; VAR must use more than one code. This is an implementation and learning check, not a convergence claim.

Full first round: all 230,659 train records; 10,813 optimizer steps, microbatch 8, accumulation 8, effective batch 64 except an epoch boundary's short minibatch. This consumes three complete shuffled passes plus 40 records of the next pass. AdamW lr=1e-4, betas=(.9,.95), weight decay=.01, gradient clip=1, warmup=500, cosine decay to 0.1×lr. Seed=17; no pilot checkpoint is used to initialize full training. Longer training or additional seeds require a separately recorded extension; this first round does not assert convergence.

One RTX 5090, two independent GPU queues: at most one convolutional run and one Prithvi run at a time. Each queue processes clean and noisy HLS, then clean and noisy LST. BF16 autocast; quantizer arithmetic stays float32. Allocator caps: 20 GiB for VAR, 6 GiB for Prithvi. Device-wide memory is sampled with `nvidia-smi` every 10 seconds, since CUDA per-process free-memory reporting under Windows can differ from physical device usage. A free-memory reading below 1 GiB stops queues at checkpoints. No eight-way simultaneous launch.

The real-HLS concurrent benchmark measured 22.49 GiB device peak for two tasks, versus 25.17 GiB for two smaller convolutional tasks plus one ViT. The three-task option had lower aggregate throughput and was rejected. Concurrent execution overlaps progress but does not multiply the compute capacity of a single GPU.

## Evaluation, artifacts, and control

Full training saves `last.pt` every 300 steps and evaluates the same fixed 256 validation records every 1,802 steps for diagnostics. `best.pt` is diagnostic only. The final comparison always uses the fixed final `last.pt` and **all 11,757 validation records**, evaluated on clean input and 10% missing input with two fixed realizations. There is no independent test split. Report physical RMSE/MAE, HLS spectral/NDVI errors, LST °C errors, missing/retained regions, and tile-cluster confidence intervals. Discrete usage and flip metrics apply only to VAR.

Each run stores exact config, data and source fingerprints, a source archive, training sample IDs, loss/memory logs, and optimizer/RNG checkpoints. The queue stores `status.json`, `gpu.jsonl`, and stdout logs. On a worker failure it stops the other queue, preserving the last completed optimizer step when possible. Add a `STOP` file at the suite directory to checkpoint and stop active training. Remove that deliberate stop marker before an explicit resume; use the identical source and config. Full runs only start after `check_pilot.py` writes a successful `gate.json`.

```powershell
.venv-paper1/Scripts/python.exe experiments/paper1/var_vs_prithvi_ae/run_suite.py --phase pilot --output runs/paper1/var_vs_prithvi_ae_pilot
.venv-paper1/Scripts/python.exe experiments/paper1/var_vs_prithvi_ae/check_pilot.py --output runs/paper1/var_vs_prithvi_ae_pilot
.venv-paper1/Scripts/python.exe experiments/paper1/var_vs_prithvi_ae/run_suite.py --phase full --output runs/paper1/var_vs_prithvi_ae_full
```
