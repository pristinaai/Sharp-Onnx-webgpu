# SHARP ONNX model derivatives

The files below are **Model Derivatives** of Apple’s SHARP research model
(`sharp_2572gikvuh.pt` from [apple/ml-sharp](https://github.com/apple/ml-sharp)), as
defined in [APPLE_SHARP_LICENSE_MODEL](./APPLE_SHARP_LICENSE_MODEL).

| File | Description |
|------|-------------|
| `public/models/sharp_web_predictor.onnx` | ONNX export of SHARP for browser inference (FP32 weights) |
| `public/models/sharp_web_predictor.onnx.data` | External weight blob for the FP32 ONNX graph |
| `public/models/sharp_web_predictor_fp16.onnx` | ONNX export with FP16 weights for WebGPU float16 |
| `public/models/sharp_web_predictor_fp16.onnx.data` | External weight blob for the FP16 ONNX graph |

## Modifications from the original Apple checkpoint

- Exported from PyTorch to ONNX for use with ONNX Runtime Web.
- Packaged large weights in ONNX external-data (`.onnx` + `.onnx.data`) for browser download.
- FP16 variant uses half-precision weights where supported by WebGPU.
- Intended for non-commercial research and demonstration in a web browser only.

## Required attribution (when redistributing)

> Apple Machine Learning Research Model is licensed under the Apple Machine Learning
> Research Model License Agreement.

A verbatim copy of the license is in [APPLE_SHARP_LICENSE_MODEL](./APPLE_SHARP_LICENSE_MODEL).
