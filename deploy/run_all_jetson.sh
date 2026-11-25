#!/usr/bin/env bash
# Every on-device measurement of the paper in one run (Section 3.10), ON the Jetson:
#   bash deploy/run_all_jetson.sh <run_dir>          e.g. runs/avffia_md (copied from the training machine)
# needs: <run_dir>/tinyfeeder.onnx, <run_dir>/onnx/*.onnx, <run_dir>/trt_layer_precisions.txt
# then:  python deploy/parse_jetson.py jetson_results > jetson_results/jetson_benchmark.json
set -uo pipefail
RUN=${1:?run dir}; export OUT=${OUT:-jetson_results}; mkdir -p "$OUT"
export LAYER_PRECISIONS="$RUN/trt_layer_precisions.txt"
{ echo "date: $(date -Is)"; cat /etc/nv_tegra_release 2>/dev/null; dpkg -l 2>/dev/null | grep -E "nvidia-jetpack |tensorrt " ;
  echo "nvpmodel:"; sudo nvpmodel -q 2>/dev/null; echo "jetson_clocks:"; sudo jetson_clocks --show 2>/dev/null | head -20; } > "$OUT/environment.txt"
# idle power: 30 s with nothing running, subtracted from the model power in parse_jetson.py
tegrastats --interval 100 > "$OUT/idle.tegrastats.log" & TP=$!; sleep 30; kill $TP; wait $TP 2>/dev/null
run() { bash "$(dirname "$0")/benchmark_jetson.sh" "$@" || echo "FAILED: $*" | tee -a "$OUT/failed.txt"; }
for B in 1 8; do
  for P in fp32 fp16 int8 mixed; do run "$RUN/tinyfeeder.onnx" tinyfeeder "$P" "$B"; done
  for f in "$RUN"/onnx/*.onnx; do
    n=$(basename "$f" .onnx)
    for P in fp32 fp16 int8; do run "$f" "$n" "$P" "$B"; done
  done
done
python3 "$(dirname "$0")/parse_jetson.py" "$OUT" > "$OUT/jetson_benchmark.json" && echo "results: $OUT/jetson_benchmark.json"
