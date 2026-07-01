---

## license: other
license_name: apple-ml-research-model
license_link: [https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL](https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL)
tags:
  - onnx
  - sharp
  - gaussian-splatting
  - research

# SHARP ONNX weights (browser)

ONNX exports of [Apple SHARP](https://github.com/apple/ml-sharp) for the
[Sharp-Onnx-webgpu](https://github.com/pristinaai/Sharp-Onnx-webgpu) browser demo.

**Research use only.** Apple Machine Learning Research Model is licensed under the
[Apple Machine Learning Research Model License Agreement](https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL).

## Files


| File                                 | Size (approx.) |
| ------------------------------------ | -------------- |
| `sharp_web_predictor.onnx`           | 7 MB           |
| `sharp_web_predictor.onnx.data`      | 2.5 GB         |
| `sharp_web_predictor_fp16.onnx`      | 7 MB           |
| `sharp_web_predictor_fp16.onnx.data` | 1.2 GB         |


Each `.onnx` file requires its matching `.onnx.data` sidecar in the same folder/URL path.

## Usage in the web app

Set these environment variables (Vercel → Settings → Environment Variables):

```
VITE_MODEL_URL_FP32=https://huggingface.co/YOUR_USER/sharp-onnx-webgpu-weights/resolve/main/sharp_web_predictor.onnx
VITE_MODEL_URL_FP16=https://huggingface.co/YOUR_USER/sharp-onnx-webgpu-weights/resolve/main/sharp_web_predictor_fp16.onnx
```

Replace `YOUR_USER` with your Hugging Face username or org.