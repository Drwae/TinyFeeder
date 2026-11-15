#!/usr/bin/env bash
# Build a mixed-precision TensorRT engine on the Jetson (Section 3.7.1).
#   INT8 everywhere except layers kept in FP16 by the QAT sensitivity analysis.
# Usage: bash deploy/build_tensorrt.sh runs/<run>/tinyfeeder.onnx runs/<run>/calib.cache
set -euo pipefail
ONNX=${1:?path to tinyfeeder.onnx}
CALIB=${2:-}
ENGINE=${ONNX%.onnx}.engine
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

ARGS=(--onnx="$ONNX" --saveEngine="$ENGINE" --fp16 --int8
      --minShapes=frame:1x3x224x224,mel:1x1x128x128
      --optShapes=frame:1x3x224x224,mel:1x1x128x128
      --maxShapes=frame:4x3x224x224,mel:4x1x128x128
      --precisionConstraints=prefer)
# Calibration cache from deploy/calibrate.py (500 training-split samples)
[[ -n "$CALIB" ]] && ARGS+=(--calib="$CALIB")
# Keep sensitive layers in FP16, e.g. --layerPrecisions=/visual/stem/stem.0/Conv:fp16
[[ -n "${FP16_LAYERS:-}" ]] && ARGS+=(--layerPrecisions="$FP16_LAYERS")

"$TRTEXEC" "${ARGS[@]}"
"$TRTEXEC" --loadEngine="$ENGINE" --shapes=frame:1x3x224x224,mel:1x1x128x128 \
           --iterations=1000 --avgRuns=100 --percentile=95,99
echo "engine written to $ENGINE"
