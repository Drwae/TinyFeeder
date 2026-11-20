#!/usr/bin/env python
"""Download (and extract) the public AV-FFIA dataset from Zenodo.

  python scripts/download_avffia.py --out /workspace/AV-FFIA            # list + download everything
  python scripts/download_avffia.py --out /workspace/AV-FFIA --list     # only list the files

Files are fetched from the Zenodo API of record 11059975 (Cui et al., IEEE
T-ASE 2024), checked against the published MD5, and archives (.zip, .tar,
.tar.gz) are extracted in place. Re-running skips files that are already
complete.
"""
import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RECORD = "11059975"


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default=RECORD)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-extract", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{a.record}", timeout=60))
    files = meta["files"]
    print(f"{meta['metadata']['title']}  —  license: {meta['metadata'].get('license', {}).get('id')}")
    total = 0
    for f in files:
        total += f["size"]
        print(f"  {f['key']:50s} {f['size'] / 1e9:8.2f} GB")
    print(f"  total {total / 1e9:.1f} GB; free space here: {shutil.disk_usage(out).free / 1e9:.0f} GB")
    if a.list:
        return
    for f in files:
        dst = out / f["key"]
        want = f.get("checksum", "").replace("md5:", "")
        if dst.exists() and dst.stat().st_size == f["size"] and (not want or md5(dst) == want):
            print("already complete:", dst.name)
        else:
            print("downloading", dst.name, flush=True)
            os.system(f"wget -c -q --show-progress -O '{dst}' '{f['links']['self']}'")
            if want and md5(dst) != want:
                raise SystemExit(f"MD5 mismatch for {dst.name}; delete it and run again")
        if a.no_extract:
            continue
        marker = out / f".extracted_{dst.name}"
        if marker.exists():
            continue
        if dst.suffix == ".zip":
            print("extracting", dst.name, flush=True)
            zipfile.ZipFile(dst).extractall(out)
        elif dst.name.endswith((".tar", ".tar.gz", ".tgz")):
            print("extracting", dst.name, flush=True)
            tarfile.open(dst).extractall(out)
        else:
            continue
        marker.touch()
    print("done. Folder contents:")
    for p in sorted(out.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
