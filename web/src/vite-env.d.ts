/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Full URL to FP32 ONNX graph (e.g. Hugging Face resolve URL). Sidecar .data is derived automatically. */
  readonly VITE_MODEL_URL_FP32?: string
  /** Full URL to FP16 ONNX graph. */
  readonly VITE_MODEL_URL_FP16?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
