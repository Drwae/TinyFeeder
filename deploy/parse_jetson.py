#!/usr/bin/env python
"""Parse trtexec + tegrastats logs written by deploy/benchmark_jetson.sh into
one JSON keyed by model name (used by scripts/paper_tables.py --jetson)."""
import json
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
res = {}
for log in sorted(d.glob("*.trtexec.log")):
    name = log.name[: -len(".trtexec.log")]
    t = log.read_text()
    lat = re.search(r"Latency: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms", t)
    thr = re.search(r"Throughput: ([\d.]+) qps", t)
    r = {}
    if lat:
        r.update(latency_ms_min=float(lat[1]), latency_ms_max=float(lat[2]), latency_ms_mean=float(lat[3]),
                 latency_ms_median=float(lat[4]), fps_from_median_latency=1000 / float(lat[4]))
    for q in ("95", "99"):
        m = re.search(rf"percentile\({q}%\) = ([\d.]+) ms", t)
        if m:
            r[f"latency_ms_p{q}"] = float(m[1])
    if thr:
        r["trtexec_throughput_qps"] = float(thr[1])
    ts = d / f"{name}.tegrastats.log"
    if ts.exists():
        pw = [int(m[1]) for m in re.finditer(r"VDD_IN (\d+)mW", ts.read_text())] or \
             [int(m[1]) for m in re.finditer(r"POM_5V_IN (\d+)/", ts.read_text())]
        if pw:
            r["board_power_w_mean"] = sum(pw) / len(pw) / 1000
    res[name] = r
pm = d / "power_mode.txt"
if pm.exists():
    res["_power_mode"] = pm.read_text().strip()
print(json.dumps(res, indent=2))
