# Data

## AV-FFIA (used in the article)

The article's experiments use the public **AV-FFIA** dataset of Cui *et al.*
(“Multimodal Fish Feeding Intensity Assessment in Aquaculture”, IEEE
Transactions on Automation Science and Engineering, 2024), available at
https://zenodo.org/records/11059975. It contains 27,000 two-second paired
video/hydrophone clips of *Oplegnathus punctatus* in a recirculating tank,
labelled with four feeding-intensity levels (none, weak, medium, strong).
Please cite the original article when using it.

`scripts/prepare_avffia.py` converts it to the `.npz` layout below; see
[REPRODUCE.md](REPRODUCE.md) for the complete protocol.

| array          | shape / dtype            | content                                              |
|----------------|--------------------------|------------------------------------------------------|
| `frames`       | `(N, H, W, 3)` uint8     | RGB frame (middle frame of the clip)                  |
| `mels`         | `(N, F, T)` float16      | log-Mel spectrogram scaled per clip to [-1, 1]        |
| `labels`       | `(N,)` int64             | 0 none, 1 weak, 2 medium, 3 strong                    |
| `group_id`     | `(N,)` int32             | recording session = split unit                        |
| `video_id`     | `(N,)` int32             | clip index (row of `<name>_clips.csv`)                |

## Synthetic AV-FFIA-like data (shipped with this repository)

`tinyfeeder/data/synthetic.py` generates paired frames and spectrograms with
the same structure. Each video is a feeding session with its own water
colour (three synthetic farms), turbidity, illumination, glare, pump hum and
noise floor; the latent feeding intensity follows a ramp-up / plateau / decay
trajectory, and about one third of sessions are low-appetite sessions that
never reach strong feeding.

| cue        | grows with feeding intensity                                   | label-independent nuisances             |
|------------|----------------------------------------------------------------|-----------------------------------------|
| visual     | fish aggregation, rise to surface, motion blur, surface splashes | turbidity haze/blur, glare, illumination |
| acoustic   | splash transients, chewing/snapping clicks                     | pump hum harmonics, aerator bursts       |

Labels are the rounded latent intensity, so samples close to class
boundaries are genuinely ambiguous (weak <-> medium confusion).

**Synthetic data are for pipeline testing, teaching and methodological
reproducibility only. Metrics computed on them are not comparable with — and
must not be reported as — the results of the article.**

## Leakage control

`video_level_split` assigns every split unit — the recording session
(`group_id`) for AV-FFIA, the synthetic video for the demo — to exactly one
of train/val/test (64/16/20%), stratified by dominant level. NAS fitness, hyperparameter and
temperature selection, quantization sensitivity, INT8 calibration
(`deploy/calibrate.py`) and early stopping use only train/val; teachers are
trained on the training split only; the test split is evaluated once.
`runs/<run>/splits.json` records the exact assignment and
`data_stats.json` reports `unit_overlap_train_test` (must be 0).
