# Data

## AV-FFIA (used in the article)

The audio-visual fish feeding-intensity recordings used in the article
(Nile tilapia, recirculating tanks; 1,950 videos, four expert-labelled
intensity levels) and the 30-day field-trial recordings were collected at
commercial partner farms. They are not redistributable under the
data-sharing agreements with the farm operators and are available from the
corresponding author on reasonable request.

To run the pipeline on these data, convert them to a single `.npz` with the
arrays below and point `data.path` in `configs/paper.yaml` to it.

| array          | shape / dtype            | content                                         |
|----------------|--------------------------|-------------------------------------------------|
| `frames`       | `(N, H, W, 3)` uint8     | RGB frame (sampled at 1 FPS)                     |
| `mels`         | `(N, F, T)` float16      | log-Mel spectrogram scaled per clip to [-1, 1]   |
| `labels`       | `(N,)` int64             | 0 none, 1 weak, 2 medium, 3 strong               |
| `video_id`     | `(N,)` int32             | recording-session id (used for the split)        |
| `turbidity`    | `(N,)` float32, optional | session turbidity (for stratified analysis)      |

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

`video_level_split` assigns every video to exactly one of train/val/test
(64/16/20%), stratified by dominant level. NAS fitness, hyperparameter and
temperature selection, quantization sensitivity, INT8 calibration
(`deploy/calibrate.py`) and early stopping use only train/val; teachers are
trained on the training split only; the test split is evaluated once.
`runs/<run>/splits.json` records the exact assignment and
`data_stats.json` reports `video_overlap_train_test` (must be 0).
