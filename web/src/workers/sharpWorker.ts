/// <reference lib="WebWorker" />

import * as ort from 'onnxruntime-web/webgpu'
import type * as OrtNamespace from 'onnxruntime-web'

import { buildSharpPlyBinary } from '../lib/ply'
import {
  isWebGpuAvailable,
  modelUrlLooksFp16,
  shouldUseWebGpuEp,
  webGpuSupportsFloat16,
} from '../lib/capabilities'
import {
  DEFAULT_WEB_MODEL_URL_FP32,
  SHARP_INTERNAL_RESOLUTION,
} from '../lib/sharpConstants'
import type {
  LoadModelRequestPayload,
  RunInferenceRequestPayload,
  WorkerInferenceResult,
  WorkerMessage,
  WorkerReply,
  WorkerRequest,
  WorkerRuntimeHintMessage,
  WorkerStatusMessage,
} from './messages'

type InferenceSession = OrtNamespace.InferenceSession
type OrtTensor = OrtNamespace.Tensor
type SessionReturnType = OrtNamespace.InferenceSession.ReturnType
type WorkerGpuPowerPreference = 'low-power' | 'high-performance'
type WorkerGpuDevice = unknown
type WorkerGpuAdapter = {
  requestDevice: () => Promise<WorkerGpuDevice>
}
type WorkerGpuNavigator = Navigator & {
  gpu?: {
    requestAdapter: (options?: {
      powerPreference?: WorkerGpuPowerPreference
      forceFallbackAdapter?: boolean
    }) => Promise<WorkerGpuAdapter | null>
  }
}

const workerScope = self as DedicatedWorkerGlobalScope
const sessionCache = new Map<string, Promise<InferenceSession>>()
const modelBufferCache = new Map<
  string,
  { graph: ArrayBuffer; data: ArrayBuffer | null; sidecarPath: string | null }
>()
const webGpuCompatibilityTierForModel = new Map<string, number>()
const webGpuDeviceCache = new Map<string, WorkerGpuDevice>()

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(0)} KB`
  }
  return `${bytes} B`
}

async function fetchWithProgress(
  url: string,
  onProgress: (loaded: number, total: number | null) => void,
): Promise<ArrayBuffer> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to download ${url} (${response.status})`)
  }

  const totalHeader = response.headers.get('content-length')
  const total = totalHeader ? Number(totalHeader) : null
  const reader = response.body?.getReader()
  if (!reader) {
    const buffer = await response.arrayBuffer()
    onProgress(buffer.byteLength, buffer.byteLength)
    return buffer
  }

  const chunks: Uint8Array[] = []
  let loaded = 0
  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      break
    }
    chunks.push(value)
    loaded += value.byteLength
    onProgress(loaded, total)
  }

  const merged = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) {
    merged.set(chunk, offset)
    offset += chunk.byteLength
  }
  return merged.buffer
}

async function loadModelBuffers(
  modelUrl: string,
  requestId?: string,
): Promise<{ graph: ArrayBuffer; data: ArrayBuffer | null; sidecarPath: string | null }> {
  const cached = modelBufferCache.get(modelUrl)
  if (cached) {
    postStatus('loading-model', 'Model weights cached in memory', requestId, 100)
    return cached
  }

  const resolved = new URL(modelUrl, self.location.href)
  let sidecarUrl: URL | null = null
  let sidecarPath: string | null = null
  if (resolved.pathname.endsWith('.onnx')) {
    sidecarUrl = new URL(resolved.href)
    sidecarUrl.pathname = `${resolved.pathname}.data`
    sidecarPath = `${resolved.pathname.split('/').pop() ?? 'model.onnx'}.data`
  }

  const graphTotalEstimate = 7 * 1024 * 1024
  const graph = await fetchWithProgress(resolved.href, (loaded, total) => {
    const graphTotal = total ?? graphTotalEstimate
    const overallTotal = sidecarUrl ? graphTotal + 2_500_000_000 : graphTotal
    const pct = sidecarUrl
      ? Math.min(5, (loaded / overallTotal) * 100)
      : total
        ? (loaded / total) * 100
        : undefined
    postStatus(
      'loading-model',
      `Downloading model graph… ${formatBytes(loaded)}${total ? ` / ${formatBytes(total)}` : ''}`,
      requestId,
      pct,
    )
  })

  let data: ArrayBuffer | null = null
  if (sidecarUrl && sidecarPath) {
    data = await fetchWithProgress(sidecarUrl.href, (loaded, total) => {
      const graphBytes = graph.byteLength
      const weightTotal = total ?? Math.max(loaded, 1)
      const overallTotal = graphBytes + weightTotal
      const pct = Math.min(99, ((graphBytes + loaded) / overallTotal) * 100)
      postStatus(
        'loading-model',
        `Downloading weights… ${formatBytes(loaded)}${total ? ` / ${formatBytes(total)}` : ''}`,
        requestId,
        pct,
      )
    })
  }

  const buffers = { graph, data, sidecarPath }
  modelBufferCache.set(modelUrl, buffers)
  return buffers
}

function resetRuntimeState(): void {
  webGpuCompatibilityTierForModel.clear()
  sessionCache.clear()
}

function sessionCacheKey(modelUrl: string): string {
  const tier = webGpuCompatibilityTierForModel.get(modelUrl) ?? 0
  const compat = tier > 0 ? `::compat${tier}` : ''
  return `webgpu::${modelUrl}${compat}`
}

function isWebGpuRuntimeError(message: string): boolean {
  return (
    message.includes('WebGPU') ||
    message.includes('ShaderModule') ||
    message.includes('compute pipeline') ||
    message.includes('Conv2dMM') ||
    message.includes('node_conv') ||
    (message.includes('ERROR_CODE: 1') && (message.includes('Conv') || message.includes('conv')))
  )
}

async function getWebGpuDevice(
  powerPreference?: WorkerGpuPowerPreference,
  forceFallbackAdapter = false,
): Promise<WorkerGpuDevice | null> {
  const gpu = (navigator as WorkerGpuNavigator).gpu
  if (!gpu) {
    return null
  }
  const cacheKey = `${powerPreference ?? 'default'}:${forceFallbackAdapter ? 'fallback' : 'std'}`
  const cached = webGpuDeviceCache.get(cacheKey)
  if (cached) {
    return cached
  }
  try {
    const adapter = await gpu.requestAdapter({ powerPreference, forceFallbackAdapter })
    if (!adapter) {
      return null
    }
    const device = await adapter.requestDevice()
    webGpuDeviceCache.set(cacheKey, device)
    return device
  } catch {
    return null
  }
}

function formatInferenceError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error)
  if (isWebGpuRuntimeError(message)) {
    return [
      'WebGPU failed to compile or run convolution kernels on this GPU/driver (common on Linux + NVIDIA).',
      'The app retries with compatibility WebGPU adapters/layouts only — browser WASM cannot load this ~2.4 GB model.',
      'Try: update Chrome/GPU drivers, close other tabs, click Apply recommendation, then Run.',
      `Details: ${message}`,
    ].join(' ')
  }
  if (
    message.includes('requires f16') ||
    message.includes('shader-f16') ||
    (message.includes('ERROR_CODE: 1') && message.includes('f16'))
  ) {
    return [
      'WebGPU on this GPU does not support float16 shaders (shader-f16).',
      'Use the FP32 model on WebGPU: click Apply recommendation, reload, then Run.',
      `Details: ${message}`,
    ].join(' ')
  }
  if (message.includes('WebGPU is required')) {
    return message
  }
  if (message.includes('All inference backends failed')) {
    return [
      'WebGPU could not run SHARP on this machine.',
      'Browser WASM is not supported for this model (4 GB heap limit).',
      'Try: update Chrome/GPU drivers, close other tabs, click Apply recommendation, then Run.',
      `Details: ${message}`,
    ].join(' ')
  }
  return message
}

const ortBaseUrl = new URL(`${import.meta.env.BASE_URL}ort/`, self.location.origin).href
ort.env.wasm.simd = true
ort.env.wasm.wasmPaths = {
  mjs: new URL('ort-wasm-simd-threaded.asyncify.mjs', ortBaseUrl).href,
  wasm: new URL('ort-wasm-simd-threaded.asyncify.wasm', ortBaseUrl).href,
}

function configureWasmThreading(useWebGpuEp: boolean): void {
  ort.env.wasm.numThreads = useWebGpuEp
    ? Math.max(1, Math.min(4, self.navigator.hardwareConcurrency || 2))
    : 1
}

function buildLowMemorySessionOptions(
  externalData?: OrtNamespace.InferenceSession.SessionOptions['externalData'],
  webGpuDevice?: WorkerGpuDevice,
  preferredLayout: 'NHWC' | 'NCHW' = 'NHWC',
): OrtNamespace.InferenceSession.SessionOptions {
  const executionProviders = [
    {
      name: 'webgpu',
      preferredLayout,
      device: webGpuDevice as object | undefined,
    },
  ] as const

  return {
    graphOptimizationLevel: 'all',
    enableMemPattern: false,
    enableCpuMemArena: false,
    executionMode: 'sequential',
    externalData,
    extra: {
      session: {
        disable_prepacking: '1',
        use_device_allocator_for_initializers: '0',
        use_ort_model_bytes_directly: '1',
        use_ort_model_bytes_for_initializers: '1',
      },
    },
    executionProviders: executionProviders as unknown as OrtNamespace.InferenceSession.SessionOptions['executionProviders'],
  }
}

function postMessageSafe(message: WorkerMessage, transfer?: Transferable[]): void {
  if (transfer && transfer.length > 0) {
    workerScope.postMessage(message, transfer)
    return
  }
  workerScope.postMessage(message)
}

function postStatus(
  stage: WorkerStatusMessage['stage'],
  message: string,
  requestId?: string,
  progress?: number,
): void {
  postMessageSafe({ type: 'status', stage, message, requestId, progress })
}

function postRuntimeHint(
  failure: WorkerRuntimeHintMessage['failure'],
  message: string,
  modelUrl?: string,
): void {
  postMessageSafe({ type: 'runtime-hint', failure, message, modelUrl })
}

function postError(requestId: string, error: unknown): void {
  const text = formatInferenceError(error)
  const reply: WorkerReply = {
    type: 'reply',
    requestId,
    ok: false,
    error: text,
  }
  postMessageSafe(reply)
}

function getSession(modelUrl: string, requestId?: string): Promise<InferenceSession> {
  const key = sessionCacheKey(modelUrl)
  const cached = sessionCache.get(key)
  if (cached) {
    return cached
  }

  const sessionPromise = createSession(modelUrl, requestId)
  sessionCache.set(key, sessionPromise)
  sessionPromise.catch(() => {
    if (sessionCache.get(key) === sessionPromise) {
      sessionCache.delete(key)
    }
  })
  return sessionPromise
}

function invalidateSessions(modelUrl: string): void {
  const prefix = `webgpu::${modelUrl}`
  for (const key of sessionCache.keys()) {
    if (key.startsWith(prefix)) {
      sessionCache.delete(key)
    }
  }
}

async function runSessionWithFallback(
  session: InferenceSession,
  modelUrl: string,
  feeds: Record<string, OrtTensor>,
  requestId?: string,
  rebuildFeeds?: (session: InferenceSession) => Record<string, OrtTensor>,
  allowWebGpuFp32 = true,
  attempted: Set<string> = new Set(),
): Promise<{ outputs: SessionReturnType; modelUrl: string }> {
  const tier = webGpuCompatibilityTierForModel.get(modelUrl) ?? 0
  const attemptKey = `webgpu::${modelUrl}::t${tier}`
  if (attempted.has(attemptKey)) {
    throw new Error('All inference backends failed for this model and hardware.')
  }
  attempted.add(attemptKey)

  try {
    const outputs = await session.run(feeds)
    return { outputs, modelUrl }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)

    if (isWebGpuRuntimeError(message)) {
      const currentTier = webGpuCompatibilityTierForModel.get(modelUrl) ?? 0
      if (currentTier < 2) {
        const nextTier = currentTier + 1
        webGpuCompatibilityTierForModel.set(modelUrl, nextTier)
        invalidateSessions(modelUrl)
        postRuntimeHint(
          'webgpu_conv',
          nextTier === 1
            ? 'WebGPU failed on default adapter; retrying with compatibility adapter/layout.'
            : 'WebGPU compatibility retry 2: high-performance adapter + NCHW layout.',
          modelUrl,
        )
        postStatus(
          'running-inference',
          nextTier === 1
            ? 'WebGPU compatibility retry: low-power/fallback adapter + NCHW layout…'
            : 'WebGPU compatibility retry 2: high-performance adapter + NCHW layout…',
          requestId,
        )
        const retrySession = await getSession(modelUrl, requestId)
        const retryFeeds = rebuildFeeds ? rebuildFeeds(retrySession) : feeds
        return runSessionWithFallback(
          retrySession,
          modelUrl,
          retryFeeds,
          requestId,
          rebuildFeeds,
          allowWebGpuFp32,
          attempted,
        )
      }

      if (
        allowWebGpuFp32 &&
        modelUrlLooksFp16(modelUrl) &&
        (await isWebGpuAvailable())
      ) {
        postRuntimeHint(
          'webgpu_f16',
          'FP16 WebGPU failed on this GPU — retrying FP32 model on WebGPU.',
          DEFAULT_WEB_MODEL_URL_FP32,
        )
        invalidateSessions(modelUrl)
        postStatus('running-inference', 'Retrying FP32 model on WebGPU…', requestId)
        const fp32Session = await getSession(DEFAULT_WEB_MODEL_URL_FP32, requestId)
        const newFeeds = rebuildFeeds ? rebuildFeeds(fp32Session) : feeds
        return runSessionWithFallback(
          fp32Session,
          DEFAULT_WEB_MODEL_URL_FP32,
          newFeeds,
          requestId,
          rebuildFeeds,
          allowWebGpuFp32,
          attempted,
        )
      }

      throw new Error(
        `WebGPU convolution kernels failed after compatibility retries. Browser WASM cannot load this model (4 GB heap limit). ${message}`,
      )
    }

    throw error
  }
}

async function createSession(modelUrl: string, requestId?: string): Promise<InferenceSession> {
  const gpuAvailable = await isWebGpuAvailable()
  const gpuFloat16 = gpuAvailable ? await webGpuSupportsFloat16() : false
  const modelIsFp16 = modelUrlLooksFp16(modelUrl)
  const useWebGpuEp = shouldUseWebGpuEp(gpuAvailable, gpuFloat16, modelUrl)
  configureWasmThreading(useWebGpuEp)

  if (!gpuAvailable) {
    throw new Error(
      'WebGPU is required. This ~2.4 GB model cannot run in browser WASM (4 GB heap limit). Use Chrome/Edge with a supported GPU.',
    )
  }
  if (!useWebGpuEp) {
    throw new Error(
      modelIsFp16
        ? 'WebGPU float16 shaders (shader-f16) are required for the FP16 model on this GPU. Use the FP32 model on WebGPU instead.'
        : 'WebGPU is required for this model. Browser WASM cannot load the ~2.4 GB weights.',
    )
  }

  const buffers = await loadModelBuffers(modelUrl, requestId)

  let externalData: OrtNamespace.InferenceSession.SessionOptions['externalData']
  if (buffers.data && buffers.sidecarPath) {
    externalData = [
      {
        path: buffers.sidecarPath,
        data: buffers.data,
      },
    ]
  }

  let webGpuDevice: WorkerGpuDevice | undefined
  let preferredLayout: 'NHWC' | 'NCHW' = 'NHWC'
  const compatTier = webGpuCompatibilityTierForModel.get(modelUrl) ?? 0
  if (compatTier > 0) {
    preferredLayout = 'NCHW'
    if (compatTier === 1) {
      const lowPowerDevice = await getWebGpuDevice('low-power', false)
      webGpuDevice = lowPowerDevice ?? (await getWebGpuDevice(undefined, true)) ?? undefined
    } else {
      webGpuDevice = (await getWebGpuDevice('high-performance', false)) ?? undefined
    }
  }

  const sessionOptions = buildLowMemorySessionOptions(externalData, webGpuDevice, preferredLayout)

  const loadMessage = modelIsFp16
    ? 'Initializing FP16 model on WebGPU…'
    : 'Initializing FP32 model on WebGPU…'
  postStatus('loading-model', loadMessage, requestId, 99)

  const session = await ort.InferenceSession.create(buffers.graph, sessionOptions)
  postStatus('loading-model', 'Model ready', requestId, 100)
  return session
}

function getTensor(outputs: SessionReturnType, key: string): OrtTensor {
  const tensor = outputs[key]
  if (!tensor) {
    const available = Object.keys(outputs)
    throw new Error(`Missing output tensor '${key}'. Available outputs: ${available.join(', ')}`)
  }
  return tensor
}

function getTensorAny(
  outputs: SessionReturnType,
  keys: readonly string[],
): { tensor: OrtTensor; key: string } {
  for (const key of keys) {
    const tensor = outputs[key]
    if (tensor) {
      return { tensor, key }
    }
  }
  const available = Object.keys(outputs)
  throw new Error(`Missing required output tensor. Tried: ${keys.join(', ')}. Available: ${available.join(', ')}`)
}

function getInputElementType(session: InferenceSession, inputIndex: number): 'float32' | 'float16' {
  const meta = session.inputMetadata[inputIndex]
  if (meta && 'type' in meta && meta.type === 'float16') {
    return 'float16'
  }
  return 'float32'
}

function float32ToFloat16Storage(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length)
  const view = new DataView(out.buffer)
  for (let i = 0; i < src.length; i += 1) {
    view.setFloat16(i * 2, src[i], true)
  }
  return out
}

function float16StorageToFloat32(data: ArrayBufferView): Float32Array {
  const count = data.byteLength / 2
  const out = new Float32Array(count)
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
  for (let i = 0; i < count; i += 1) {
    out[i] = view.getFloat16(i * 2, true)
  }
  return out
}

function createModelTensor(
  elementType: 'float32' | 'float16',
  values: Float32Array,
  dims: readonly number[],
): OrtTensor {
  if (elementType === 'float16') {
    return new ort.Tensor('float16', float32ToFloat16Storage(values), dims)
  }
  return new ort.Tensor('float32', values, dims)
}

function isFloat16Array(data: unknown): data is ArrayLike<number> {
  const Float16ArrayCtor = globalThis.Float16Array
  return Float16ArrayCtor !== undefined && data instanceof Float16ArrayCtor
}

function asFloat32(name: string, tensor: OrtTensor): Float32Array {
  const data = tensor.data
  if (data instanceof Float32Array) {
    return data
  }
  if (tensor.type === 'float16') {
    if (isFloat16Array(data)) {
      return Float32Array.from(data)
    }
    if (data instanceof Uint16Array || ArrayBuffer.isView(data)) {
      return float16StorageToFloat32(data)
    }
  }
  throw new Error(
    `Expected '${name}' tensor as float32/float16, got ${tensor.type} / ${Object.prototype.toString.call(data)}`,
  )
}

interface PrunedGaussians {
  count: number
  meanVectors: Float32Array
  singularValues: Float32Array
  quaternions: Float32Array
  colors: Float32Array
  opacities: Float32Array
}

function copyTriplets(source: Float32Array, indices: number[]): Float32Array {
  const out = new Float32Array(indices.length * 3)
  let outOffset = 0
  for (const index of indices) {
    const srcOffset = index * 3
    out[outOffset] = source[srcOffset]
    out[outOffset + 1] = source[srcOffset + 1]
    out[outOffset + 2] = source[srcOffset + 2]
    outOffset += 3
  }
  return out
}

function copyQuads(source: Float32Array, indices: number[]): Float32Array {
  const out = new Float32Array(indices.length * 4)
  let outOffset = 0
  for (const index of indices) {
    const srcOffset = index * 4
    out[outOffset] = source[srcOffset]
    out[outOffset + 1] = source[srcOffset + 1]
    out[outOffset + 2] = source[srcOffset + 2]
    out[outOffset + 3] = source[srcOffset + 3]
    outOffset += 4
  }
  return out
}

function copySingles(source: Float32Array, indices: number[]): Float32Array {
  const out = new Float32Array(indices.length)
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = source[indices[i]]
  }
  return out
}

function flattenBatchTensor(
  tensor: OrtTensor,
  channels: number,
  label: string,
): { data: Float32Array; count: number } {
  const dims = tensor.dims
  const data = asFloat32(label, tensor)

  if (dims.length < 2) {
    throw new Error(`Output '${label}' should have rank >= 2. Got dims=${dims.join('x')}`)
  }

  const count = channels === 1 ? data.length : Math.floor(data.length / channels)
  if (count <= 0) {
    throw new Error(`Output '${label}' has no data.`)
  }
  if (channels > 1 && count * channels !== data.length) {
    throw new Error(`Output '${label}' length (${data.length}) is not divisible by ${channels}.`)
  }

  return { data, count }
}

function pruneGaussians(
  meanVectors: Float32Array,
  singularValues: Float32Array,
  quaternions: Float32Array,
  colors: Float32Array,
  opacities: Float32Array,
  opacityThreshold: number,
  maxGaussians: number,
): { pruned: PrunedGaussians; totalCount: number } {
  const totalCount = opacities.length
  const threshold = Number.isFinite(opacityThreshold) ? opacityThreshold : 0
  const cappedMax = Number.isFinite(maxGaussians) && maxGaussians > 0 ? Math.floor(maxGaussians) : 0

  const selected: number[] = []
  for (let i = 0; i < totalCount; i += 1) {
    if (opacities[i] >= threshold) {
      selected.push(i)
    }
  }

  if (selected.length === 0) {
    for (let i = 0; i < totalCount; i += 1) {
      selected.push(i)
    }
  }

  if (cappedMax > 0 && selected.length > cappedMax) {
    selected.sort((a, b) => opacities[b] - opacities[a])
    selected.length = cappedMax
    selected.sort((a, b) => a - b)
  }

  const pruned: PrunedGaussians = {
    count: selected.length,
    meanVectors: copyTriplets(meanVectors, selected),
    singularValues: copyTriplets(singularValues, selected),
    quaternions: copyQuads(quaternions, selected),
    colors: copyTriplets(colors, selected),
    opacities: copySingles(opacities, selected),
  }

  return { pruned, totalCount }
}

function quaternionToRotationMatrix(
  qw: number,
  qx: number,
  qy: number,
  qz: number,
): [number, number, number, number, number, number, number, number, number] {
  const norm = Math.hypot(qw, qx, qy, qz) || 1
  const w = qw / norm
  const x = qx / norm
  const y = qy / norm
  const z = qz / norm

  const ww = w * w
  const xx = x * x
  const yy = y * y
  const zz = z * z
  const wx = w * x
  const wy = w * y
  const wz = w * z
  const xy = x * y
  const xz = x * z
  const yz = y * z

  return [
    ww + xx - yy - zz,
    2 * (xy - wz),
    2 * (xz + wy),
    2 * (xy + wz),
    ww - xx + yy - zz,
    2 * (yz - wx),
    2 * (xz - wy),
    2 * (yz + wx),
    ww - xx - yy + zz,
  ]
}

function jacobiRotateSymmetric3x3(matrix: Float64Array, vectors: Float64Array, p: number, q: number): void {
  const pp = p * 3 + p
  const qq = q * 3 + q
  const pq = p * 3 + q
  const qp = q * 3 + p

  const app = matrix[pp]
  const aqq = matrix[qq]
  const apq = matrix[pq]
  if (Math.abs(apq) < 1e-18) {
    return
  }

  const tau = (aqq - app) / (2 * apq)
  const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau))
  const c = 1 / Math.sqrt(1 + t * t)
  const s = t * c

  for (let k = 0; k < 3; k += 1) {
    if (k === p || k === q) {
      continue
    }

    const kp = k * 3 + p
    const pk = p * 3 + k
    const kq = k * 3 + q
    const qk = q * 3 + k

    const mkp = matrix[kp]
    const mkq = matrix[kq]

    const newMkp = c * mkp - s * mkq
    const newMkq = s * mkp + c * mkq

    matrix[kp] = newMkp
    matrix[pk] = newMkp
    matrix[kq] = newMkq
    matrix[qk] = newMkq
  }

  matrix[pp] = c * c * app - 2 * s * c * apq + s * s * aqq
  matrix[qq] = s * s * app + 2 * s * c * apq + c * c * aqq
  matrix[pq] = 0
  matrix[qp] = 0

  for (let k = 0; k < 3; k += 1) {
    const kp = k * 3 + p
    const kq = k * 3 + q
    const vkp = vectors[kp]
    const vkq = vectors[kq]
    vectors[kp] = c * vkp - s * vkq
    vectors[kq] = s * vkp + c * vkq
  }
}

function jacobiEigenSymmetric3x3(matrix: Float64Array, vectors: Float64Array): void {
  vectors.fill(0)
  vectors[0] = 1
  vectors[4] = 1
  vectors[8] = 1

  for (let sweep = 0; sweep < 8; sweep += 1) {
    const offDiag = Math.abs(matrix[1]) + Math.abs(matrix[2]) + Math.abs(matrix[5])
    if (offDiag < 1e-14) {
      break
    }
    jacobiRotateSymmetric3x3(matrix, vectors, 0, 1)
    jacobiRotateSymmetric3x3(matrix, vectors, 0, 2)
    jacobiRotateSymmetric3x3(matrix, vectors, 1, 2)
  }
}

function swapEigenColumns(vectors: Float64Array, c0: number, c1: number): void {
  for (let row = 0; row < 3; row += 1) {
    const i0 = row * 3 + c0
    const i1 = row * 3 + c1
    const temp = vectors[i0]
    vectors[i0] = vectors[i1]
    vectors[i1] = temp
  }
}

function sortEigenpairsDescending(eigenvalues: Float64Array, vectors: Float64Array): void {
  if (eigenvalues[0] < eigenvalues[1]) {
    const temp = eigenvalues[0]
    eigenvalues[0] = eigenvalues[1]
    eigenvalues[1] = temp
    swapEigenColumns(vectors, 0, 1)
  }
  if (eigenvalues[1] < eigenvalues[2]) {
    const temp = eigenvalues[1]
    eigenvalues[1] = eigenvalues[2]
    eigenvalues[2] = temp
    swapEigenColumns(vectors, 1, 2)
  }
  if (eigenvalues[0] < eigenvalues[1]) {
    const temp = eigenvalues[0]
    eigenvalues[0] = eigenvalues[1]
    eigenvalues[1] = temp
    swapEigenColumns(vectors, 0, 1)
  }
}

function ensureProperRotation(vectors: Float64Array): void {
  const r00 = vectors[0]
  const r01 = vectors[1]
  const r02 = vectors[2]
  const r10 = vectors[3]
  const r11 = vectors[4]
  const r12 = vectors[5]
  const r20 = vectors[6]
  const r21 = vectors[7]
  const r22 = vectors[8]

  const det =
    r00 * (r11 * r22 - r12 * r21) -
    r01 * (r10 * r22 - r12 * r20) +
    r02 * (r10 * r21 - r11 * r20)

  if (det < 0) {
    vectors[2] *= -1
    vectors[5] *= -1
    vectors[8] *= -1
  }
}

function quaternionFromRotationMatrix(
  r00: number,
  r01: number,
  r02: number,
  r10: number,
  r11: number,
  r12: number,
  r20: number,
  r21: number,
  r22: number,
): [number, number, number, number] {
  const trace = r00 + r11 + r22
  let qw: number
  let qx: number
  let qy: number
  let qz: number

  if (trace > 0) {
    const s = 2 * Math.sqrt(Math.max(1e-12, trace + 1))
    qw = 0.25 * s
    qx = (r21 - r12) / s
    qy = (r02 - r20) / s
    qz = (r10 - r01) / s
  } else if (r00 > r11 && r00 > r22) {
    const s = 2 * Math.sqrt(Math.max(1e-12, 1 + r00 - r11 - r22))
    qw = (r21 - r12) / s
    qx = 0.25 * s
    qy = (r01 + r10) / s
    qz = (r02 + r20) / s
  } else if (r11 > r22) {
    const s = 2 * Math.sqrt(Math.max(1e-12, 1 + r11 - r00 - r22))
    qw = (r02 - r20) / s
    qx = (r01 + r10) / s
    qy = 0.25 * s
    qz = (r12 + r21) / s
  } else {
    const s = 2 * Math.sqrt(Math.max(1e-12, 1 + r22 - r00 - r11))
    qw = (r10 - r01) / s
    qx = (r02 + r20) / s
    qy = (r12 + r21) / s
    qz = 0.25 * s
  }

  const norm = Math.hypot(qw, qx, qy, qz) || 1
  return [qw / norm, qx / norm, qy / norm, qz / norm]
}

function unprojectGaussiansInPlace(
  gaussians: Pick<PrunedGaussians, 'count' | 'meanVectors' | 'singularValues' | 'quaternions'>,
  scaleX: number,
  scaleY: number,
): void {
  const matrix = new Float64Array(9)
  const vectors = new Float64Array(9)
  const eigenvalues = new Float64Array(3)

  for (let i = 0; i < gaussians.count; i += 1) {
    const idx3 = i * 3
    const idx4 = i * 4

    gaussians.meanVectors[idx3] *= scaleX
    gaussians.meanVectors[idx3 + 1] *= scaleY

    const [r00, r01, r02, r10, r11, r12, r20, r21, r22] = quaternionToRotationMatrix(
      gaussians.quaternions[idx4],
      gaussians.quaternions[idx4 + 1],
      gaussians.quaternions[idx4 + 2],
      gaussians.quaternions[idx4 + 3],
    )

    const v0 = gaussians.singularValues[idx3] ** 2
    const v1 = gaussians.singularValues[idx3 + 1] ** 2
    const v2 = gaussians.singularValues[idx3 + 2] ** 2

    const c00 = r00 * r00 * v0 + r01 * r01 * v1 + r02 * r02 * v2
    const c01 = r00 * r10 * v0 + r01 * r11 * v1 + r02 * r12 * v2
    const c02 = r00 * r20 * v0 + r01 * r21 * v1 + r02 * r22 * v2
    const c11 = r10 * r10 * v0 + r11 * r11 * v1 + r12 * r12 * v2
    const c12 = r10 * r20 * v0 + r11 * r21 * v1 + r12 * r22 * v2
    const c22 = r20 * r20 * v0 + r21 * r21 * v1 + r22 * r22 * v2

    // A * C * A^T where A = diag(scaleX, scaleY, 1)
    matrix[0] = c00 * scaleX * scaleX
    matrix[1] = c01 * scaleX * scaleY
    matrix[2] = c02 * scaleX
    matrix[3] = matrix[1]
    matrix[4] = c11 * scaleY * scaleY
    matrix[5] = c12 * scaleY
    matrix[6] = matrix[2]
    matrix[7] = matrix[5]
    matrix[8] = c22

    jacobiEigenSymmetric3x3(matrix, vectors)
    eigenvalues[0] = matrix[0]
    eigenvalues[1] = matrix[4]
    eigenvalues[2] = matrix[8]
    sortEigenpairsDescending(eigenvalues, vectors)
    ensureProperRotation(vectors)

    gaussians.singularValues[idx3] = Math.sqrt(Math.max(eigenvalues[0], 1e-12))
    gaussians.singularValues[idx3 + 1] = Math.sqrt(Math.max(eigenvalues[1], 1e-12))
    gaussians.singularValues[idx3 + 2] = Math.sqrt(Math.max(eigenvalues[2], 1e-12))

    const [qw, qx, qy, qz] = quaternionFromRotationMatrix(
      vectors[0],
      vectors[1],
      vectors[2],
      vectors[3],
      vectors[4],
      vectors[5],
      vectors[6],
      vectors[7],
      vectors[8],
    )
    gaussians.quaternions[idx4] = qw
    gaussians.quaternions[idx4 + 1] = qx
    gaussians.quaternions[idx4 + 2] = qy
    gaussians.quaternions[idx4 + 3] = qz
  }
}

function resolveOutputTensors(outputs: SessionReturnType): {
  meanVectors: OrtTensor
  singularValues: OrtTensor
  quaternions: OrtTensor
  colors: OrtTensor
  opacities: OrtTensor
  isNdcOutput: boolean
} {
  const mean = getTensorAny(outputs, ['mean_vectors_ndc', 'mean_vectors'])
  const scales = getTensorAny(outputs, ['singular_values_ndc', 'singular_values'])
  const quats = getTensorAny(outputs, ['quaternions_ndc', 'quaternions'])
  const colors = getTensor(outputs, 'colors')
  const opacities = getTensor(outputs, 'opacities')

  const isNdcOutput =
    mean.key === 'mean_vectors_ndc' ||
    scales.key === 'singular_values_ndc' ||
    quats.key === 'quaternions_ndc'

  return {
    meanVectors: mean.tensor,
    singularValues: scales.tensor,
    quaternions: quats.tensor,
    colors,
    opacities,
    isNdcOutput,
  }
}

function validateModelInputs(session: InferenceSession): { supportsWrapperScalars: boolean } {
  if (session.inputNames.length < 2) {
    throw new Error(
      `Unexpected model inputs (${session.inputNames.join(', ')}). Expected at least image + disparity_factor inputs.`,
    )
  }

  if (session.inputNames.length !== 2 && session.inputNames.length < 5) {
    throw new Error(
      `Unsupported model input count ${session.inputNames.length}. Expected 2 (raw predictor export) or 5 (legacy wrapper export).`,
    )
  }

  return { supportsWrapperScalars: session.inputNames.length >= 5 }
}

function buildSessionFeeds(
  session: InferenceSession,
  payload: RunInferenceRequestPayload,
  imageTensorData: Float32Array,
  supportsWrapperScalars: boolean,
): Record<string, OrtTensor> {
  const feeds: Record<string, OrtTensor> = {
    [session.inputNames[0]]: createModelTensor(
      getInputElementType(session, 0),
      imageTensorData,
      [1, 3, SHARP_INTERNAL_RESOLUTION, SHARP_INTERNAL_RESOLUTION],
    ),
    [session.inputNames[1]]: createModelTensor(
      getInputElementType(session, 1),
      new Float32Array([payload.disparityFactor]),
      [1],
    ),
  }
  if (supportsWrapperScalars) {
    feeds[session.inputNames[2]] = createModelTensor(
      getInputElementType(session, 2),
      new Float32Array([payload.focalPx]),
      [1],
    )
    feeds[session.inputNames[3]] = createModelTensor(
      getInputElementType(session, 3),
      new Float32Array([payload.imageWidth]),
      [1],
    )
    feeds[session.inputNames[4]] = createModelTensor(
      getInputElementType(session, 4),
      new Float32Array([payload.imageHeight]),
      [1],
    )
  }
  return feeds
}

async function handleLoadModel(requestId: string, payload: LoadModelRequestPayload): Promise<void> {
  if (payload.resetRuntimeState) {
    resetRuntimeState()
  } else {
    const key = sessionCacheKey(payload.modelUrl)
    const existing = sessionCache.get(key)
    if (existing) {
      try {
        await existing
        postStatus('loading-model', 'Model ready (cached)', requestId, 100)
        const reply: WorkerReply = {
          type: 'reply',
          requestId,
          ok: true,
          result: { modelUrl: payload.modelUrl },
        }
        postMessageSafe(reply)
        return
      } catch {
        sessionCache.delete(key)
      }
    }
  }

  postStatus('loading-model', 'Starting model download…', requestId, 0)
  const session = await getSession(payload.modelUrl, requestId)
  validateModelInputs(session)

  const reply: WorkerReply = {
    type: 'reply',
    requestId,
    ok: true,
    result: { modelUrl: payload.modelUrl },
  }
  postMessageSafe(reply)
}

async function handleRunInference(
  requestId: string,
  payload: RunInferenceRequestPayload,
): Promise<void> {
  if (payload.imageWidth <= 0 || payload.imageHeight <= 0) {
    throw new Error('Image width/height must be > 0.')
  }
  if (payload.focalPx <= 0 || !Number.isFinite(payload.focalPx)) {
    throw new Error('Focal length must be a positive finite number.')
  }

  const allowWebGpuFp32 = payload.allowWebGpuFp32 !== false

  const session = await getSession(payload.modelUrl, requestId)
  const { supportsWrapperScalars } = validateModelInputs(session)

  const imageTensorData = new Float32Array(payload.imageTensor)
  const expectedImageValues = 3 * SHARP_INTERNAL_RESOLUTION * SHARP_INTERNAL_RESOLUTION
  if (imageTensorData.length !== expectedImageValues) {
    throw new Error(
      `Unexpected image tensor size ${imageTensorData.length}. Expected ${expectedImageValues}.`,
    )
  }

  postStatus('running-inference', 'Running SHARP inference in the browser…', requestId)

  const rebuildFeeds = (activeSession: InferenceSession) =>
    buildSessionFeeds(activeSession, payload, imageTensorData, supportsWrapperScalars)

  const feeds = rebuildFeeds(session)

  const runResult = await runSessionWithFallback(
    session,
    payload.modelUrl,
    feeds,
    requestId,
    rebuildFeeds,
    allowWebGpuFp32,
  )
  const outputs = runResult.outputs
  const resolved = resolveOutputTensors(outputs)

  const { data: meanVectors, count } = flattenBatchTensor(
    resolved.meanVectors,
    3,
    resolved.isNdcOutput ? 'mean_vectors_ndc' : 'mean_vectors',
  )
  const { data: singularValues, count: singularCount } = flattenBatchTensor(
    resolved.singularValues,
    3,
    resolved.isNdcOutput ? 'singular_values_ndc' : 'singular_values',
  )
  const { data: quaternions, count: quaternionCount } = flattenBatchTensor(
    resolved.quaternions,
    4,
    resolved.isNdcOutput ? 'quaternions_ndc' : 'quaternions',
  )
  const { data: colors, count: colorCount } = flattenBatchTensor(resolved.colors, 3, 'colors')
  const { data: opacities, count: opacityCount } = flattenBatchTensor(resolved.opacities, 1, 'opacities')

  if (
    count !== singularCount ||
    count !== quaternionCount ||
    count !== colorCount ||
    count !== opacityCount
  ) {
    throw new Error(
      `Output count mismatch: means=${count}, scales=${singularCount}, quat=${quaternionCount}, colors=${colorCount}, opacities=${opacityCount}`,
    )
  }

  postStatus('filtering', 'Filtering and capping Gaussians for browser preview/export…', requestId)
  const { pruned, totalCount } = pruneGaussians(
    meanVectors,
    singularValues,
    quaternions,
    colors,
    opacities,
    payload.opacityThreshold,
    payload.maxGaussians,
  )

  if (resolved.isNdcOutput) {
    postStatus('filtering', 'Converting NDC Gaussians to metric space in-browser…', requestId)
    const scaleX = payload.imageWidth / (2 * payload.focalPx)
    const scaleY = payload.imageHeight / (2 * payload.focalPx)
    unprojectGaussiansInPlace(pruned, scaleX, scaleY)
  }

  postStatus('building-ply', 'Building binary .ply for preview and download…', requestId)
  const ply = buildSharpPlyBinary({
    ...pruned,
    imageWidth: payload.imageWidth,
    imageHeight: payload.imageHeight,
    focalPx: payload.focalPx,
  })

  const result: WorkerInferenceResult = {
    plyBuffer: ply.buffer.slice(ply.byteOffset, ply.byteOffset + ply.byteLength),
    selectedGaussians: pruned.count,
    totalGaussians: totalCount,
    effectiveModelUrl: runResult.modelUrl,
  }

  const reply: WorkerReply = {
    type: 'reply',
    requestId,
    ok: true,
    result,
  }

  postMessageSafe(reply, [result.plyBuffer as ArrayBuffer])
}

workerScope.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { data } = event

  try {
    if (data.type === 'load-model') {
      await handleLoadModel(data.requestId, data.payload)
      return
    }

    if (data.type === 'run-inference') {
      await handleRunInference(data.requestId, data.payload)
      return
    }

    throw new Error(
      `Unknown worker request type: ${(data as { type?: string }).type ?? 'undefined'}`,
    )
  } catch (error) {
    postError(data.requestId, error)
  }
}
