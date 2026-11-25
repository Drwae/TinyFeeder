#!/usr/bin/env bash
# Latency, throughput and power of one ONNX model on a Jetson (run ON the device; protocol of Section 3.10).
#   bash deploy/benchmark_jetson.sh <model.onnx> <name> <fp32|fp16|int8|mixed> [batch]
#   mixed = INT8 with the layers listed in LAYER_PRECISIONS kept in FP16 (TinyFeeder's QAT precision map,
#           written by `run_pipeline.py --stages export` as <run>/trt_layer_precisions.txt)
# Writes <out>/<name>_<prec>_b<batch>.{build,trtexec,tegrastats}.log; OUT defaults to ./jetson_results.
# INT8 engines are built without calibration data: TensorRT then uses placeholder scales, which is valid for
# timing and power but NOT for accuracy (accuracy is measured on the training machine, Section 4).
set -euo pipefail
ONNX=${1:?onnx}; NAME=${2:?name}; PREC=${3:?fp32|fp16|int8|mixed}; B=${4:-1}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
OUT=${OUT:-jetson_results}; mkdir -p "$OUT"
read -r IMG MEL <<< "${SHAPES:-224 128}"
SH="frame:${B}x3x${IMG}x${IMG},mel:${B}x1x${MEL}x${MEL}"
case "$PREC" in
  fp32)  FLAGS=() ;;
  fp16)  FLAGS=(--fp16) ;;
  int8)  FLAGS=(--int8 --fp16) ;;
  mixed) FLAGS=(--int8 --fp16 --precisionConstraints=obey --layerPrecisions="$(cat "${LAYER_PRECISIONS:?file}")") ;;
  *) echo "unknown precision $PREC"; exit 1 ;;
esac
TAG="${NAME}_${PREC}_b${B}"
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$OUT/$TAG.engine" --minShapes="$SH" --optShapes="$SH" --maxShapes="$SH" \
           "${FLAGS[@]}" > "$OUT/$TAG.build.log" 2>&1
tegrastats --interval 100 > "$OUT/$TAG.tegrastats.log" & TP=$!
"$TRTEXEC" --loadEngine="$OUT/$TAG.engine" --shapes="$SH" --warmUp=2000 --iterations=1000 --duration=0 \
           --percentile=50,95,99 --useSpinWait > "$OUT/$TAG.trtexec.log" 2>&1
kill $TP; wait $TP 2>/dev/null || true
echo "done: $OUT/$TAG"
