Local EMA technical pilot (2026-10-14)

Scope: ordinary single-scale VQ-VAE and the existing grouped_g8k256 model.
Three arms per family: gradient baseline, EMA-only, EMA + K-means + dead-code maintenance.
No extra latent RMS normalization; no pretrained weights; no remote submission or tracking upload.

Run from the workspace root:
.venv-paper1/Scripts/python.exe -m experiments.paper1.lst_ema_pilot_20261014.run --output outputs/NEW_UNIQUE_DIRECTORY --updates 128 --effective-batch 16 --micro 2 --val-records 64

Output must not already exist. Each run writes exact source hashes, data fingerprints,
train/validation sample membership, effective configs, per-update logs, validation
metrics and code histograms, and checkpoints at step 0/midpoint/end. A final state
also retains optimizer and Torch RNG state. Automatic continuation is not implemented.

Analyze a completed default six-arm / 128-update run:
.venv-paper1/Scripts/python.exe -m experiments.paper1.lst_ema_pilot_20261014.analyze --output outputs/ema_local_pilot_20261014_attempt01

Technical budget: 2048 distinct train records per model, drawn from the fixed HLP v2
epoch-0 order. This is 0.00935 of a complete training epoch, NOT a formal 5-epoch run.
All arms share samples, corruption, initial backbone weights within family, effective
batch16, microbatch2, and seed17. Validation uses 64 fixed random v2-val records,
one clean view and two fixed pixel10 views. No test-set use. No independent seed repeats.
Keep original training configs/checkpoints frozen. Pilot settings are separate files.

EMA specifics:
- Codebook embeddings are frozen for autograd and excluded from AdamW.
- Accumulate assignment counts and input sums across all microbatches, then update
  with decay0.99 after each optimizer step. No updates during validation.
- Grouped has one shared dictionary per group; aggregate all its residual scales.
- Phi convolutions remain trainable using the native fitting term; this is a
  project-specific grouped adaptation, not a literal audio SoundStream reproduction.
- Maintenance uses 10-iteration deterministic K-means on the first effective batch
  of training latents; this requires extra encoder-only passes. No validation input.
- Dead threshold is bias-corrected EMA assignment frequency <0.01/K; check every
  16 updates after a 16-update grace period. Reset centroids and EMA state together
  using a bounded deterministic pool of current-batch residual vectors.
- K-means and resets are bundled in this initial pilot and cannot be individually
  attributed. Further isolated arms would be needed to separate their effects.

Comparison:
- Compare raw physical validation RMSE and quantization MSE. Total losses differ
  because ordinary EMA removes the gradient codebook-fitting loss.
- Per-update active-code/perplexity logs are averages over microbatches. Validation
  histograms aggregate the fixed set; Grouped reports independent groups/scales.
- Original grouped model has a larger backbone and 17x ordinary VQ's nominal index
  bits, so between-family comparisons are not equal-rate claims.
- These models retain the original oracle scalar-mean side information. At step0
  the zero output head predicts the image mean; it is a useful reference baseline.
- Best checkpoint selection is only among the three saved steps on the small fixed
  subset, including step0. A step0 win means no observed improvement over that baseline.
