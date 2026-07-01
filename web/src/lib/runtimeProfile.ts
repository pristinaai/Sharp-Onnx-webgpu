import {
  DEFAULT_WEB_MODEL_URL_FP16,
  DEFAULT_WEB_MODEL_URL_FP32,
} from './sharpConstants'
import { modelUrlLooksFp16, probeGpuStack } from './capabilities'

export type RuntimeFailureKind = 'webgpu_conv' | 'webgpu_f16'

export interface RuntimeProfile {
  modelUrl: string
  runtimeLabel: string
  summary: string
  confidence: 'high' | 'medium' | 'low' | 'unsupported'
  webGpuAvailable: boolean
  webGpuFloat16: boolean
  riskyWebGpuFp32: boolean
  adapterLabel: string
  allowWebGpuFp32: boolean
}

interface StoredRuntimeHistory {
  failures: Partial<Record<RuntimeFailureKind, number>>
  lastSuccess?: {
    modelUrl: string
    runtimeLabel: string
  }
}

const STORAGE_KEY = 'sharp-onnx-runtime-v1'

function readHistory(): StoredRuntimeHistory {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { failures: {} }
    const parsed = JSON.parse(raw) as StoredRuntimeHistory
    return {
      failures: parsed.failures ?? {},
      lastSuccess: parsed.lastSuccess,
    }
  } catch {
    return { failures: {} }
  }
}

function writeHistory(history: StoredRuntimeHistory): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
  } catch {
    // Ignore quota / private-mode errors.
  }
}

export function recordRuntimeFailure(kind: RuntimeFailureKind): void {
  const history = readHistory()
  history.failures[kind] = (history.failures[kind] ?? 0) + 1
  writeHistory(history)
}

export function recordRuntimeSuccess(profile: RuntimeProfile): void {
  const history = readHistory()
  history.lastSuccess = {
    modelUrl: profile.modelUrl,
    runtimeLabel: profile.runtimeLabel,
  }
  if (profile.modelUrl === DEFAULT_WEB_MODEL_URL_FP32) {
    delete history.failures.webgpu_conv
  }
  writeHistory(history)
}

export function clearRuntimeHistory(): void {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {
    // ignore
  }
}

function hasFailure(kind: RuntimeFailureKind): boolean {
  return (readHistory().failures[kind] ?? 0) > 0
}

function profileFrom(
  modelUrl: string,
  runtimeLabel: string,
  summary: string,
  confidence: RuntimeProfile['confidence'],
  stack: {
    webGpuAvailable: boolean
    webGpuFloat16: boolean
    riskyWebGpuFp32: boolean
    adapterLabel?: string
  },
  allowWebGpuFp32: boolean,
): RuntimeProfile {
  return {
    modelUrl,
    runtimeLabel,
    summary,
    confidence,
    webGpuAvailable: stack.webGpuAvailable,
    webGpuFloat16: stack.webGpuFloat16,
    riskyWebGpuFp32: stack.riskyWebGpuFp32,
    adapterLabel: stack.adapterLabel ?? '',
    allowWebGpuFp32,
  }
}

export async function resolveBestRuntimeProfile(): Promise<RuntimeProfile> {
  const stack = await probeGpuStack()
  const { webGpuAvailable, webGpuFloat16, riskyWebGpuFp32, isSoftwareAdapter, adapterLabel } = stack
  const history = readHistory()

  if (isSoftwareAdapter) {
    return profileFrom(
      DEFAULT_WEB_MODEL_URL_FP32,
      'Software WebGPU (unsupported)',
      `Chrome is using a CPU software WebGPU adapter (${adapterLabel || 'SwiftShader'}), not your GPU. SHARP cannot run this way. Enable hardware WebGPU: chrome://flags → "Unsafe WebGPU Support" + "Vulkan" = Enabled, update NVIDIA drivers, and check chrome://gpu shows your GPU (not SwiftShader).`,
      'unsupported',
      stack,
      false,
    )
  }

  if (!webGpuAvailable) {
    return profileFrom(
      DEFAULT_WEB_MODEL_URL_FP32,
      'WebGPU required',
      'WebGPU is unavailable. This ~2.4 GB model cannot run in browser WASM (4 GB heap limit). Use Chrome/Edge with a supported GPU and drivers.',
      'unsupported',
      stack,
      false,
    )
  }

  const webGpuConvFailed = hasFailure('webgpu_conv')
  const webGpuF16Failed = hasFailure('webgpu_f16')
  const blockWebGpuFp32 = riskyWebGpuFp32 || webGpuConvFailed

  if (history.lastSuccess) {
    const last = history.lastSuccess
    const lastWouldFail =
      (last.modelUrl === DEFAULT_WEB_MODEL_URL_FP32 && webGpuConvFailed) ||
      (modelUrlLooksFp16(last.modelUrl) && webGpuF16Failed)

    if (!lastWouldFail) {
      return profileFrom(
        last.modelUrl,
        last.runtimeLabel,
        `Previously worked on this browser — reusing ${last.runtimeLabel}.`,
        'high',
        stack,
        true,
      )
    }
  }

  if (webGpuFloat16 && !webGpuF16Failed) {
    return profileFrom(
      DEFAULT_WEB_MODEL_URL_FP16,
      'WebGPU + FP16',
      'Best option: WebGPU with float16 shaders. Weights and compute stay on the GPU.',
      'high',
      stack,
      true,
    )
  }

  if (riskyWebGpuFp32 && !webGpuConvFailed) {
    return profileFrom(
      DEFAULT_WEB_MODEL_URL_FP32,
      'WebGPU + FP32',
      'Your GPU lacks WebGPU float16 shaders — using FP32 on WebGPU. Browser WASM cannot load this model.',
      'medium',
      stack,
      true,
    )
  }

  if (webGpuConvFailed) {
    return profileFrom(
      DEFAULT_WEB_MODEL_URL_FP32,
      'WebGPU + FP32 (retry)',
      'WebGPU conv failed before — retrying FP32 on WebGPU with compatibility adapters.',
      'low',
      stack,
      true,
    )
  }

  if (!blockWebGpuFp32) {
    return profileFrom(
      DEFAULT_WEB_MODEL_URL_FP32,
      'WebGPU + FP32',
      'Recommended: FP32 ONNX on WebGPU.',
      webGpuFloat16 ? 'medium' : 'high',
      stack,
      true,
    )
  }

  return profileFrom(
    DEFAULT_WEB_MODEL_URL_FP32,
    'Limited in browser',
    'WebGPU failed on this machine. Update Chrome/GPU drivers, close other tabs, then click Apply recommendation and Run.',
    'unsupported',
    stack,
    webGpuAvailable,
  )
}

export function profilesMatch(
  current: { modelUrl: string },
  recommended: RuntimeProfile,
): boolean {
  return current.modelUrl === recommended.modelUrl
}

export function inferFailureFromError(message: string): RuntimeFailureKind | null {
  if (
    message.includes('Conv2dMM') ||
    message.includes('ShaderModule') ||
    message.includes('compute pipeline') ||
    (message.includes('ERROR_CODE: 1') && message.includes('Conv'))
  ) {
    return 'webgpu_conv'
  }
  if (
    message.includes('requires f16') ||
    message.includes('shader-f16') ||
    (message.includes('ERROR_CODE: 1') && message.includes('f16'))
  ) {
    return 'webgpu_f16'
  }
  return null
}
