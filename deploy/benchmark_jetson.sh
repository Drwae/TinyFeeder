#!/usr/bin/env bash
# Measure latency, throughput and power of ONNX models on a Jetson (run ON the device).
#   bash deploy/benchmark_jetson.sh runs/avffia/tinyfeeder.onnx tinyfeeder_final [int8]
#   bash deploy/benchmark_jetson.sh runs/avffia/onnx/baseline_mobilenet_v3_large.onnx baseline_mobilenet_v3_large fp16
# Writes runs/<...>/jetson/<name>.trtexec.log and <name>.tegrastats.log; then run
#   python deploy/parse_jetson.py runs/avffia/jetson > runs/avffia/jetson_benchmark.json
# Batch size 1, single forward pass per clip (no test-time augmentation).
# Record the power mode used (sudo nvpmodel -q) and keep it identical for all models.
set -euo pipefail
ONNX=${1:?onnx}; NAME=${2:?name}; PREC=${3:-int8}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
OUT=$(dirname "$ONNX")/jetson; [[ "$ONNX" == */onnx/* ]] && OUT=$(dirname "$(dirname "$ONNX")")/jetson
mkdir -p "$OUT"
read -r IMG MEL <<< "${SHAPES:-224 128}"
SH="frame:1x3x${IMG}x${IMG},mel:1x1x${MEL}x${MEL}"
FLAGS=(--fp16); [[ "$PREC" == int8 ]] && FLAGS+=(--int8 ${CALIB:+--calib=$CALIB})
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$OUT/$NAME.engine" --shapes="$SH" "${FLAGS[@]}" > "$OUT/$NAME.build.log" 2>&1
tegrastats --interval 100 > "$OUT/$NAME.tegrastats.log" & TP=$!
"$TRTEXEC" --loadEngine="$OUT/$NAME.engine" --shapes="$SH" --warmUp=2000 --duration=60 \
           --percentile=50,95,99 --useSpinWait > "$OUT/$NAME.trtexec.log" 2>&1
kill $TP
nvpmodel -q > "$OUT/power_mode.txt" 2>/dev/null || true
echo "done: $OUT/$NAME.*"
