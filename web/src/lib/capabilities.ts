import {
  DEFAULT_WEB_MODEL_URL_FP16,
  DEFAULT_WEB_MODEL_URL_FP32,
} from './sharpConstants'

export interface InferenceCapabilities {
  webGpuAvailable: boolean
  webGpuFloat16: boolean
  recommendedModelUrl: string
  runtimeLabel: string
  summary: string
}

type GpuNavigator = Navigator & {
  gpu?: {
    requestAdapter: () => Promise<GpuAdapter | null>
  }
}

type GpuAdapterInfo = {
  vendor?: string
  architecture?: string
  device?: string
  description?: string
}

type GpuAdapter = {
  features: { has: (name: string) => boolean }
  info?: GpuAdapterInfo
  requestAdapterInfo?: () => Promise<GpuAdapterInfo>
}

export interface GpuStackInfo {
  webGpuAvailable: boolean
  webGpuFloat16: boolean
  vendor: string
  /** Full adapter identity string (vendor / architecture / description) for diagnostics. */
  adapterLabel: string
  /** True when the WebGPU adapter is a CPU software renderer (SwiftShader / llvmpipe). */
  isSoftwareAdapter: boolean
  /** FP32 conv on WebGPU often fails without shader-f16 (typical on Linux + discrete GPU). */
  riskyWebGpuFp32: boolean
}

export async function probeGpuStack(): Promise<GpuStackInfo> {
  const webGpuAvailable = await isWebGpuAvailable()
  const webGpuFloat16 = webGpuAvailable ? await webGpuSupportsFloat16() : false
  let vendor = ''
  let adapterLabel = ''

  if (webGpuAvailable) {
    try {
      const gpu = (navigator as GpuNavigator).gpu
      const adapter = (await gpu?.requestAdapter()) as GpuAdapter | null
      let info: GpuAdapterInfo | undefined = adapter?.info
      if (!info && adapter?.requestAdapterInfo) {
        info = await adapter.requestAdapterInfo()
      }
      if (info) {
        vendor = info.vendor ?? ''
        adapterLabel = [info.vendor, info.architecture, info.device, info.description]
          .filter(Boolean)
          .join(' / ')
      }
    } catch {
      // Adapter info is optional in some browsers.
    }
  }

  const vendorLower = vendor.toLowerCase()
  const labelLower = adapterLabel.toLowerCase()
  const isSoftwareAdapter =
    webGpuAvailable &&
    (labelLower.includes('swiftshader') ||
      labelLower.includes('llvmpipe') ||
      labelLower.includes('software') ||
      labelLower.includes('microsoft basic') ||
      vendorLower.includes('swiftshader'))
  const riskyWebGpuFp32 =
    webGpuAvailable &&
    !webGpuFloat16 &&
    (vendorLower.includes('nvidia') ||
      vendorLower.includes('amd') ||
      vendorLower.includes('intel'))

  return {
    webGpuAvailable,
    webGpuFloat16,
    vendor,
    adapterLabel,
    isSoftwareAdapter,
    riskyWebGpuFp32,
  }
}

export async function probeInferenceCapabilities(): Promise<InferenceCapabilities> {
  const stack = await probeGpuStack()
  const { webGpuAvailable, webGpuFloat16, riskyWebGpuFp32 } = stack

  if (webGpuAvailable && webGpuFloat16) {
    return {
      webGpuAvailable,
      webGpuFloat16,
      recommendedModelUrl: DEFAULT_WEB_MODEL_URL_FP16,
      runtimeLabel: 'WebGPU + FP16',
      summary:
        'WebGPU with float16 shader support detected. Using FP16 ONNX on GPU.',
    }
  }

  if (webGpuAvailable && riskyWebGpuFp32) {
    return {
      webGpuAvailable,
      webGpuFloat16,
      recommendedModelUrl: DEFAULT_WEB_MODEL_URL_FP32,
      runtimeLabel: 'WebGPU + FP32',
      summary:
        'Discrete GPU without WebGPU float16 shaders — using FP32 ONNX on WebGPU (browser WASM cannot hold this model).',
    }
  }

  if (webGpuAvailable) {
    return {
      webGpuAvailable,
      webGpuFloat16,
      recommendedModelUrl: DEFAULT_WEB_MODEL_URL_FP32,
      runtimeLabel: 'WebGPU + FP32',
      summary:
        'WebGPU without float16 shaders. Using FP32 ONNX on GPU so weights stay off the 4 GB WASM heap.',
    }
  }

  return {
    webGpuAvailable,
    webGpuFloat16,
    recommendedModelUrl: DEFAULT_WEB_MODEL_URL_FP32,
    runtimeLabel: 'WebGPU required',
    summary:
      'WebGPU is unavailable. This model is too large for browser WASM (4 GB heap limit). Use Chrome/Edge with a supported GPU.',
  }
}

export async function isWebGpuAvailable(): Promise<boolean> {
  try {
    const gpu = (navigator as GpuNavigator).gpu
    if (!gpu) return false
    const adapter = await gpu.requestAdapter()
    return adapter !== null
  } catch {
    return false
  }
}

export async function webGpuSupportsFloat16(): Promise<boolean> {
  try {
    const gpu = (navigator as GpuNavigator).gpu
    if (!gpu) return false
    const adapter = await gpu.requestAdapter()
    if (!adapter) return false
    return adapter.features.has('shader-f16')
  } catch {
    return false
  }
}

export function modelUrlLooksFp16(modelUrl: string): boolean {
  return modelUrl.toLowerCase().includes('fp16')
}

export function shouldUseWebGpuEp(
  webGpuAvailable: boolean,
  webGpuFloat16: boolean,
  modelUrl: string,
): boolean {
  if (!webGpuAvailable) return false
  if (modelUrlLooksFp16(modelUrl)) return webGpuFloat16
  return true
}
