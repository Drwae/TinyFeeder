#!/usr/bin/env python
"""Parse the trtexec + tegrastats logs written by deploy/run_all_jetson.sh into one JSON keyed by
<model>_<precision>_b<batch> (used by scripts/paper_tables.py --jetson).

latency_ms_median   median GPU-compute + transfer time of one inference of the whole batch (trtexec "Latency")
clips_per_s         batch / median latency: clips classified per second
power_w             mean module input power (VDD_IN) during the timed runs, minus the idle power (idle.tegrastats.log)
"""
import json
import re
import sys
from pathlib import Path


def power(log: Path):
    t = log.read_text() if log.exists() else ""
    pw = [int(m[1]) for m in re.finditer(r"VDD_IN (\d+)mW", t)] or [int(m[1]) for m in re.finditer(r"POM_5V_IN (\d+)/", t)]
    return sum(pw) / len(pw) / 1000 if pw else None


d = Path(sys.argv[1])
idle = power(d / "idle.tegrastats.log")
res = {"_idle_power_w": idle}
for log in sorted(d.glob("*.trtexec.log")):
    tag = log.name[: -len(".trtexec.log")]
    b = int(re.search(r"_b(\d+)$", tag)[1]) if re.search(r"_b(\d+)$", tag) else 1
    t = log.read_text()
    lat = re.search(r"Latency: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms", t)
    r = {"batch": b}
    if lat:
        med = float(lat[4])
        r.update(latency_ms_min=float(lat[1]), latency_ms_mean=float(lat[3]), latency_ms_median=med,
                 clips_per_s=b * 1000 / med)
    for q in ("95", "99"):
        m = re.search(rf"percentile\({q}%\) = ([\d.]+) ms", t)
        if m:
            r[f"latency_ms_p{q}"] = float(m[1])
    pw = power(d / f"{tag}.tegrastats.log")
    if pw is not None:
        r["power_w_total"] = pw
        if idle is not None:
            r["power_w_above_idle"] = pw - idle
    res[tag] = r
env = d / "environment.txt"
if env.exists():
    res["_environment"] = env.read_text().strip()
print(json.dumps(res, indent=2))
