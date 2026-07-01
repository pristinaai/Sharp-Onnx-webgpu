export const SHARP_INTERNAL_RESOLUTION = 1536

/** Hosted ONNX weights (Vercel / production). Local dev uses /public/models/ unless overridden. */
export const HUGGINGFACE_WEIGHTS_REPO = 'sentiantai/sharp-onnx-webgpu-weights'
export const HUGGINGFACE_WEIGHTS_BASE = `https://huggingface.co/${HUGGINGFACE_WEIGHTS_REPO}/resolve/main`

function modelUrlFromEnv(envKey: keyof ImportMetaEnv, localPath: string, hfFile: string): string {
  const value = import.meta.env[envKey]
  if (typeof value === 'string' && value.trim().length > 0) {
    return value.trim()
  }
  if (import.meta.env.DEV) {
    return localPath
  }
  return `${HUGGINGFACE_WEIGHTS_BASE}/${hfFile}`
}

export const DEFAULT_WEB_MODEL_URL_FP16 = modelUrlFromEnv(
  'VITE_MODEL_URL_FP16',
  '/models/sharp_web_predictor_fp16.onnx',
  'sharp_web_predictor_fp16.onnx',
)
export const DEFAULT_WEB_MODEL_URL_FP32 = modelUrlFromEnv(
  'VITE_MODEL_URL_FP32',
  '/models/sharp_web_predictor.onnx',
  'sharp_web_predictor.onnx',
)
/** Probed at runtime — FP32 for WebGPU-without-f16, FP16 when shader-f16 is available. */
export const DEFAULT_WEB_MODEL_URL = DEFAULT_WEB_MODEL_URL_FP32
export const DEFAULT_OPACITY_THRESHOLD = 0.1
export const DEFAULT_MAX_GAUSSIANS = 5_000_000
export const DEFAULT_FOCAL_MM = 30
export const FILM_35MM_DIAGONAL_MM = Math.hypot(36, 24)
