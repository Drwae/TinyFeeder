#!/usr/bin/env python
"""Convert the public AV-FFIA dataset (Cui et al., IEEE T-ASE 2024;
https://zenodo.org/records/11059975) into the TinyFeeder .npz layout.

Step 1 - inspect the download (prints class counts, folder structure and
how clips will be grouped into recording sessions; nothing is written):

  python scripts/prepare_avffia.py --video-root /data/AV-FFIA/video_dataset \
         --audio-root /data/AV-FFIA/audio_dataset --inspect

Step 2 - convert:

  python scripts/prepare_avffia.py --video-root ... --audio-root ... \
         --out data/avffia/avffia_224.npz --img-size 224 --n-mels 128 --workers 16

Expected layout (as in the U-FFIA loaders): <root>/<d1>/<d2>/<class>/<clip>.mp4
and the same tree with .wav files for audio; class folders are
strong / medium (or middle) / weak / none. Audio is paired to video by the
clip id (file stem without its 9-character prefix, as in U-FFIA), falling
back to an identical stem.

Split unit. Each 2-s clip is cut from a one-minute recording, so clips of
the same recording - and recordings of the same session - are strongly
correlated. `group_id` is therefore the recording session, defined as the
directory path above the class folder (`--group-depth` controls how many
levels are used); `--group-regex` refines it to the one-minute recording when
the recording id can be read from the clip file name. scripts/run_pipeline.py splits train/val/test by
`group_id`, so no session contributes to more than one split. A per-clip
index (clips.csv) is written next to the .npz so that the exact split can
be published.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CLASS_MAP = {"none": 0, "weak": 1, "medium": 2, "middle": 2, "strong": 3}


def find_clips(video_root, audio_root, group_depth, group_regex=None):
    rx = re.compile(group_regex) if group_regex else None
    vids = sorted(glob.glob(os.path.join(video_root, "**", "*.mp4"), recursive=True))
    wavs = sorted(glob.glob(os.path.join(audio_root, "**", "*.wav"), recursive=True)) if audio_root else []
    wav_index = {}
    for w in wavs:
        rel = os.path.relpath(os.path.dirname(w), audio_root)
        stem = Path(w).stem
        wav_index[(rel, stem)] = w
        wav_index.setdefault((rel, stem[9:]), w)
    clips, unpaired = [], []
    for v in vids:
        rel_dir = os.path.relpath(os.path.dirname(v), video_root)
        parts = Path(rel_dir).parts
        cls = next((p.lower() for p in reversed(parts) if p.lower() in CLASS_MAP), None)
        if cls is None:
            unpaired.append((v, "no class folder"))
            continue
        stem = Path(v).stem
        w = wav_index.get((rel_dir, stem)) or wav_index.get((rel_dir, stem[9:]))
        if w is None:
            unpaired.append((v, "no matching wav"))
            continue
        session_parts = [p for p in parts if p.lower() not in CLASS_MAP]
        group = "/".join(session_parts[:group_depth]) if group_depth > 0 else "/".join(session_parts)
        if rx is not None:  # finer unit: the one-minute recording a clip was cut from
            m = rx.search(stem)
            if m is None:
                unpaired.append((v, f"--group-regex did not match {stem}"))
                continue
            group = f"{group}/{m.group(1)}"
        clips.append({"video": v, "audio": w, "label": CLASS_MAP[cls], "class": cls, "group": group or "root"})
    return clips, unpaired


def inspect(clips, unpaired):
    print(f"paired clips: {len(clips)}   unpaired videos: {len(unpaired)}")
    for v, why in unpaired[:5]:
        print("   unpaired:", why, v)
    print("per class:", dict(Counter(c["class"] for c in clips)))
    g = defaultdict(Counter)
    for c in clips:
        g[c["group"]][c["class"]] += 1
    print(f"recording sessions (split units): {len(g)}")
    for k in list(g)[:15]:
        print(f"   {k:40s} {dict(g[k])}")
    if len(g) < 10:
        print("WARNING: fewer than 10 split units. Use --group-regex to split by the one-minute recording each "
              "clip was cut from (e.g. --group-regex '^(.*)_\\d+$' if clips are named <recording>_<n>), "
              "otherwise the test split rests on very few sessions.")
    print("example:", clips[0] if clips else None)


def _frame(path, size, which):
    import cv2

    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = {"middle": n // 2, "first": 0}.get(which, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {path}")
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    h, w = f.shape[:2]
    s = size / min(h, w)
    f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h, w = f.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return f[y0:y0 + size, x0:x0 + size]


def _mel(path, sr, n_mels, n_fft):
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    hop = max(1, len(y) // (n_mels - 1))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    S = librosa.power_to_db(S, ref=np.max)[:, :n_mels]
    if S.shape[1] < n_mels:
        S = np.pad(S, ((0, 0), (0, n_mels - S.shape[1])), constant_values=S.min())
    S = 2 * (S - S.min()) / (S.max() - S.min() + 1e-8) - 1
    return S.astype(np.float16)


def _work(args):
    clip, size, sr, n_mels, n_fft, which = args
    return _frame(clip["video"], size, which), _mel(clip["audio"], sr, n_mels, n_fft)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--group-depth", type=int, default=0, help="directory levels above the class folder that "
                    "define a recording session (0 = all levels)")
    ap.add_argument("--group-regex", help="regex with one capture group extracting the recording id from the "
                    "clip file stem; groups become <session>/<recording>")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--n-mels", type=int, default=128)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--frame", default="middle", choices=["middle", "first"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    clips, unpaired = find_clips(a.video_root, a.audio_root, a.group_depth, a.group_regex)
    inspect(clips, unpaired)
    if a.inspect:
        return
    if not a.out:
        sys.exit("--out is required unless --inspect is given")
    groups = sorted({c["group"] for c in clips})
    gid = {g: i for i, g in enumerate(groups)}
    N = len(clips)
    frames = np.zeros((N, a.img_size, a.img_size, 3), np.uint8)
    mels = np.zeros((N, a.n_mels, a.n_mels), np.float16)
    jobs = [(c, a.img_size, a.sr, a.n_mels, a.n_fft, a.frame) for c in clips]
    with Pool(a.workers) as pool:
        for i, (f, m) in enumerate(pool.imap(_work, jobs, chunksize=16)):
            frames[i], mels[i] = f, m
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{N}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, frames=frames, mels=mels,
             labels=np.array([c["label"] for c in clips], np.int64),
             group_id=np.array([gid[c["group"]] for c in clips], np.int32),
             video_id=np.arange(N, dtype=np.int32))
    with open(out.with_name(out.stem + "_clips.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "group_id", "group", "label", "class", "video", "audio"])
        for i, c in enumerate(clips):
            w.writerow([i, gid[c["group"]], c["group"], c["label"], c["class"],
                        os.path.relpath(c["video"], a.video_root), os.path.relpath(c["audio"], a.audio_root)])
    print(f"wrote {out} ({N} clips, {len(groups)} sessions)")


if __name__ == "__main__":
    main()
