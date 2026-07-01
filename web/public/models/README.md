# Model files

| File | In Git (LFS) | Size |
|------|----------------|------|
| `sharp_web_predictor_fp16.onnx` | Yes | ~7 MB |
| `sharp_web_predictor_fp16.onnx.data` | Yes | ~1.2 GB |
| `sharp_web_predictor.onnx` | Yes | ~7 MB |
| `sharp_web_predictor.onnx.data` | Assembled locally | ~2.5 GB |
| `sharp_web_predictor.onnx.data.part*` | Yes (2 parts) | ~1.5 GB + ~1.1 GB |

GitHub LFS rejects individual files over **2 GB**. The FP32 sidecar is split into parts; `npm install` (postinstall) reassembles them:

```bash
git lfs pull
cd web && npm install   # runs join_model_parts.mjs
```

To re-split after export:

```bash
node scripts/split_model_data.mjs web/public/models/sharp_web_predictor.onnx.data
```

The FP16 pair works as-is on WebGPU with `shader-f16`. FP32 is for WebGPU without float16 shaders (common on Linux/NVIDIA).
