# Model files

| File | In Git (LFS) | Size |
|------|----------------|------|
| `sharp_web_predictor_fp16.onnx` | Yes | ~7 MB |
| `sharp_web_predictor_fp16.onnx.data` | Yes | ~1.2 GB |
| `sharp_web_predictor.onnx` | Yes | ~7 MB |
| `sharp_web_predictor.onnx.data` | **No** | ~2.5 GB |

GitHub LFS rejects individual files larger than **2 GB**, so the FP32 weight sidecar is not in this repository.

## FP32 weights (local export)

After cloning, generate the FP32 sidecar locally:

```bash
./scripts/setup_export_env.sh
source .venv-export/bin/activate
python scripts/export_sharp_onnx.py \
  --sharp-repo vendor/ml-sharp \
  --output web/public/models/sharp_web_predictor.onnx \
  --verbose
```

The FP16 pair in Git is enough for WebGPU with `shader-f16`. Use FP32 when your GPU has WebGPU but not float16 shaders (common on Linux/NVIDIA).
