K512 formal local extension, G representation only.
Six jobs: VQ K512 / Grouped 8 x K512, each baseline / EMA / EMA+K-means+dead reset.
5 full HLP v2 epochs, 218975 train records/epoch, effective batch64, micro8.
Per model: 1094875 record presentations, 17110 updates; final batch31 each epoch.
Each epoch: full val11684 records, clean and pixel10, two fixed views each.
Select minimum pixel10 valid-pixel-weighted Celsius RMSE; retain every epoch. No test selection.
Same data, sample-order rules, LR, losses, EMA settings and latent geometry as parent.
Fresh initialization; no pilot or parent weights reused. No latent RMS normalization added.
Grouped internal kind retains legacy grouped_g8k256 routing; actual book size512 in all8 groups.
Nominal index bits/image: VQ2304 (was2560), Grouped48960 (was43520); plus mean32bits.
This is a codebook-size ablation, not an equal-bit-rate comparison.
Scheduler waits for all6 parent jobs to finish successfully, audits their validation receipts,
then runs real CUDA full/tail preflight for all6 K512 models before enabling training.
Parent failure/stopping keeps this queue waiting. STOP in this root stops scheduler/training.
Keep this computer running. Restart launch.ps1 to resume after interruption.
Parent: C:\Users\rog\Desktop\iclr\outputs\ema_local_full_20261015_attempt01
