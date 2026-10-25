# TFLite export (Raspberry Pi 4, Google Coral Edge TPU)

```bash
pip install onnx2tf tensorflow
# NHWC TFLite model, full-integer quantised with the training-split calibration set
onnx2tf -i runs/<run>/tinyfeeder.onnx -o runs/<run>/tflite \
        -oiqt -qt per-channel -cind frame runs/<run>/calibration_frame.npy [[[[0.485,0.456,0.406]]]] [[[[0.229,0.224,0.225]]]] \
        -cind mel runs/<run>/calibration_mel.npy [[[[0]]]] [[[[1]]]]
# Coral: compile the INT8 model for the Edge TPU
edgetpu_compiler runs/<run>/tflite/tinyfeeder_full_integer_quant.tflite
```

The Edge TPU executes INT8 only, so the FP16 layers selected by the
sensitivity analysis run in INT8 on Coral (Section 6.3.1); expect a small
accuracy drop relative to the Jetson mixed-precision engine.
