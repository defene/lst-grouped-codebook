FULL LOCAL EMA EXPERIMENT — 2026-10-15

Six G-representation runs: VQ-VAE and grouped_g8k256, each with gradient baseline,
EMA only, EMA + K-means + dead-code maintenance. All start from scratch; pilot
weights are not used. Code and hyperparameters are frozen by SHA256 preflight.

Data: HLP v2, explicit processed_data/hlp_split_v2_20260914, LST only, 256x256.
Train: all 218975 records / 2796 tiles, five complete epochs.
Per model: 1094875 record presentations; effective batch64; 17110 updates.
Microbatch8 + accumulation8; each epoch ends with a correctly weighted batch31.
No dropping or repeated padding. Same seed17 and epoch-dependent order in all arms.
Pixel10 training; original G residual representation and oracle mean side information.
No additional latent RMS normalization. Train-only v2 normalization is retained.

LR: original 1e-4 cosine, VQ warmup0 and Grouped warmup500 updates, shared within
each family. AdamW betas(.9,.95), wd.01, BF16, clip1. Original model sizes and rates.
The two families are not parameter/rate matched; Grouped has 17x nominal index bits.

Validation: EVERY complete epoch, all 11684 validation records /147 tiles;
clean and pixel10 each have TWO fixed views. Eval/inference mode with RNG restoration.
Selection: minimum pixel10 raw valid-pixel weighted Celsius RMSE, earliest epoch on ties.
No test data. No 64-record subset. Validation failure prevents the next epoch.
Reports also include MAE, equal-image normalized reconstruction MSE, and streaming
per-group/per-scale code usage. Total training loss differs for EMA codebook updates.
K-means and dead-code maintenance remain bundled in this initial comparison.

Artifacts under runs/<name>:
- config.json / model.json / source_hashes.json / data_fingerprint.json
- train_<session>.jsonl: per-update losses, gradient probes, actual sample counts
- last.pt: full optimizer, model, EMA and RNG state, saved every100 updates + epoch end
- epochs/epoch_01.pt ... epoch_05.pt: retained immutable epoch states
- epoch_validation/epoch_01.json ... epoch_05.json: complete validation receipts
- best.pt / selection.json: validation selection, independent of final epoch
- training_complete.json: written only after full budget and last epoch validation

Queue progress: queue_status.json. Active run progress: runs/<name>/status.json.
Validation progress: runs/<name>/validation_progress.json.
Logs: logs/<name>.log. The local GPU runs one job at a time.

Start/resume: launch.ps1 (hidden background window; duplicate runs rejected by lock).
The queue resumes existing last.pt and finishes pending validation before training.
To stop safely, create a plain file named STOP in this output directory. It is checked
at update/validation boundaries. Work since the most recent periodic checkpoint may
be recomputed on resume, and append-only session logs retain that evidence.
Remove STOP before restarting. No automatic restart after a machine reboot is installed.
Keep the machine powered on for this multi-day experiment.

Preflight is a separate technical check: 64-record +31-record updates on real data,
8 real validation samples, CPU full-validation/tail/resume contracts. These preflight
steps do not count towards the formal training budget or full validation completion.
