export const SHARP_INTERNAL_RESOLUTION = 1536
export const DEFAULT_WEB_MODEL_URL_FP16 = '/models/sharp_web_predictor_fp16.onnx'
export const DEFAULT_WEB_MODEL_URL_FP32 = '/models/sharp_web_predictor.onnx'
/** Probed at runtime — FP32 for WebGPU-without-f16, FP16 when shader-f16 is available. */
export const DEFAULT_WEB_MODEL_URL = DEFAULT_WEB_MODEL_URL_FP32
export const DEFAULT_OPACITY_THRESHOLD = 0.1
export const DEFAULT_MAX_GAUSSIANS = 5_000_000
export const DEFAULT_FOCAL_MM = 30
export const FILM_35MM_DIAGONAL_MM = Math.hypot(36, 24)
