import { StrictMode, useCallback, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { EXAMPLE_IMAGES, fetchExampleImageFile, type ExampleImage } from './lib/exampleImages'
import { estimateFocalLengthFromFile } from './lib/focal'
import { imageFileToSharpTensor, readImageInfo } from './lib/image'
import {
  inferFailureFromError,
  profilesMatch,
  recordRuntimeFailure,
  recordRuntimeSuccess,
  resolveBestRuntimeProfile,
  clearRuntimeHistory,
  type RuntimeProfile,
} from './lib/runtimeProfile'
import {
  APPLE_MODEL_ATTRIBUTION_NOTICE,
  SHARP_MODEL_DERIVATIVE_NOTICE_URL,
  SHARP_MODEL_LICENSE_URL,
  WORDPRESS_PHOTOS_URL,
} from './lib/sharpLicense'
import { SharpWorkerClient } from './lib/sharpWorkerClient'
import {
  DEFAULT_MAX_GAUSSIANS,
  DEFAULT_OPACITY_THRESHOLD,
  DEFAULT_WEB_MODEL_URL_FP16,
  DEFAULT_WEB_MODEL_URL_FP32,
} from './lib/sharpConstants'
import type { WorkerRuntimeHintMessage, WorkerStatusMessage } from './workers/messages'
import { SplatViewer } from './components/SplatViewer'
import { SPLATEDIT_VIEWER_URL } from './lib/viewers'
import './App.css'

function simplifyStatus(message: string): string {
  const lower = message.toLowerCase()
  if (lower.includes('loading example')) {
    return 'Loading example…'
  }
  if (lower.includes('loading model') || lower.includes('starting model download')) {
    return 'Loading model…'
  }
  if (lower.includes('running') || lower.includes('inference')) {
    return 'Generating splat…'
  }
  if (
    lower.includes('filtering') ||
    lower.includes('converting') ||
    lower.includes('building')
  ) {
    return 'Processing…'
  }
  if (lower.includes('done') || lower.includes('preview below')) {
    return 'Done'
  }
  if (lower.includes('select an image') || lower.includes('load an image')) {
    return 'Load an image first'
  }
  if (lower.includes('model loaded')) {
    return 'Model ready — load an image and run'
  }
  if (lower.includes('selected ') && lower.includes('click run')) {
    return 'Image ready — click Run'
  }
  if (lower.includes('ready') && lower.includes('click run')) {
    return 'Ready'
  }
  if (lower.includes('retrying') || lower.includes('auto-switching')) {
    return 'Retrying…'
  }
  if (lower.includes('webgpu required') || lower.includes('unsupported')) {
    return 'WebGPU required'
  }
  if (message.length > 100) {
    return `${message.slice(0, 97)}…`
  }
  return message
}

function App() {
  const [recommended, setRecommended] = useState<RuntimeProfile | null>(null)
  const [modelUrl, setModelUrl] = useState(DEFAULT_WEB_MODEL_URL_FP32)
  const [runtimeLabel, setRuntimeLabel] = useState('detecting…')
  const [backendHint, setBackendHint] = useState('')
  const [status, setStatusRaw] = useState('Ready')
  const [busy, setBusy] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [plyBlob, setPlyBlob] = useState<Blob | null>(null)
  const [maxGaussians, setMaxGaussians] = useState(DEFAULT_MAX_GAUSSIANS)
  const [opacityThreshold, setOpacityThreshold] = useState(DEFAULT_OPACITY_THRESHOLD)
  const [modelLoaded, setModelLoaded] = useState(false)
  const [hasImage, setHasImage] = useState(false)
  const [manualOverride, setManualOverride] = useState(false)
  const [imageName, setImageName] = useState('')
  const [selectedExampleId, setSelectedExampleId] = useState<string | null>(null)

  const workerRef = useRef<SharpWorkerClient | null>(null)
  const selectedFileRef = useRef<File | null>(null)
  const activeProfileRef = useRef<RuntimeProfile | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const modelLoadedRef = useRef(false)

  const setStatus = useCallback((message: string) => {
    setStatusRaw(simplifyStatus(message))
  }, [])

  const applyProfile = useCallback((profile: RuntimeProfile, markManual = false) => {
    setModelUrl(profile.modelUrl)
    setRuntimeLabel(profile.runtimeLabel)
    setBackendHint(profile.summary)
    setRecommended(profile)
    activeProfileRef.current = profile
    if (markManual) {
      setManualOverride(true)
    }
    setModelLoaded(false)
    modelLoadedRef.current = false
  }, [])

  const refreshRecommendation = useCallback(
    async (apply = true): Promise<RuntimeProfile> => {
      const profile = await resolveBestRuntimeProfile()
      setRecommended(profile)
      if (apply && !manualOverride) {
        applyProfile(profile)
      }
      return profile
    },
    [applyProfile, manualOverride],
  )

  const onStatus = useCallback(
    (message: WorkerStatusMessage) => {
      setStatus(message.message)
    },
    [setStatus],
  )

  const onRuntimeHint = useCallback(
    (hint: WorkerRuntimeHintMessage) => {
      const isCompatRetry =
        hint.message.includes('compatibility') || hint.message.includes('retrying with')
      if (hint.failure && !isCompatRetry) {
        recordRuntimeFailure(hint.failure)
      }
      void resolveBestRuntimeProfile().then((profile) => {
        setRecommended(profile)
        if (!manualOverride) {
          if (hint.modelUrl !== undefined) {
            applyProfile({
              ...profile,
              modelUrl: hint.modelUrl,
              runtimeLabel:
                hint.modelUrl === DEFAULT_WEB_MODEL_URL_FP32 ? 'WebGPU + FP32' : profile.runtimeLabel,
              summary: hint.message || profile.summary,
            })
          } else {
            applyProfile(profile)
          }
          setStatus(hint.message ? 'Retrying…' : 'Settings updated — run again')
        }
      })
    },
    [applyProfile, manualOverride, setStatus],
  )

  useEffect(() => {
    void resolveBestRuntimeProfile().then((profile) => {
      applyProfile(profile)
      setStatus('Ready')
    })
  }, [applyProfile, setStatus])

  useEffect(() => {
    workerRef.current = new SharpWorkerClient(onStatus)
    workerRef.current.setRuntimeHintHandler(onRuntimeHint)
    return () => {
      workerRef.current?.dispose()
    }
  }, [onStatus, onRuntimeHint])

  const ensureBestSettings = useCallback(async (): Promise<RuntimeProfile> => {
    const profile = await resolveBestRuntimeProfile()
    setRecommended(profile)
    if (!manualOverride) {
      applyProfile(profile)
      return profile
    }
    const current: RuntimeProfile = {
      modelUrl,
      runtimeLabel,
      summary: backendHint,
      confidence: profile.confidence,
      webGpuAvailable: profile.webGpuAvailable,
      webGpuFloat16: profile.webGpuFloat16,
      riskyWebGpuFp32: profile.riskyWebGpuFp32,
      adapterLabel: profile.adapterLabel,
      allowWebGpuFp32: profile.allowWebGpuFp32,
    }
    activeProfileRef.current = current
    return current
  }, [applyProfile, backendHint, manualOverride, modelUrl, runtimeLabel])

  const handleRuntimeError = useCallback(
    async (error: unknown, retry: () => Promise<void>): Promise<boolean> => {
      const message = error instanceof Error ? error.message : String(error)
      const failure = inferFailureFromError(message)
      if (failure) {
        recordRuntimeFailure(failure)
      }
      const next = await resolveBestRuntimeProfile()
      setRecommended(next)
      const current = activeProfileRef.current ?? {
        modelUrl,
        runtimeLabel: '',
        summary: '',
        confidence: next.confidence,
        webGpuAvailable: next.webGpuAvailable,
        webGpuFloat16: next.webGpuFloat16,
        riskyWebGpuFp32: next.riskyWebGpuFp32,
        adapterLabel: next.adapterLabel,
        allowWebGpuFp32: next.allowWebGpuFp32,
      }
      if (profilesMatch(current, next)) {
        return false
      }
      applyProfile(next)
      setManualOverride(false)
      setStatus('Retrying…')
      try {
        await retry()
        return true
      } catch {
        return false
      }
    },
    [applyProfile, modelUrl, setStatus],
  )

  const ensureModelLoaded = useCallback(
    async (resetRuntimeState = false): Promise<void> => {
      const worker = workerRef.current
      if (!worker) {
        return
      }
      if (modelLoadedRef.current && !resetRuntimeState) {
        return
      }

      await ensureBestSettings()
      const active = activeProfileRef.current
      if (!active) {
        return
      }

      setStatus('Loading model…')

      const load = async () => {
        await worker.loadModel({
          modelUrl: active.modelUrl,
          allowWebGpuFp32: active.allowWebGpuFp32,
          resetRuntimeState: resetRuntimeState || !modelLoadedRef.current,
        })
        setModelLoaded(true)
        modelLoadedRef.current = true
      }

      try {
        await load()
      } catch (error) {
        const retried = await handleRuntimeError(error, load)
        if (!retried) {
          setModelLoaded(false)
          modelLoadedRef.current = false
          throw error
        }
      }
    },
    [ensureBestSettings, handleRuntimeError, setStatus],
  )

  const runPipeline = useCallback(async () => {
    const file = selectedFileRef.current
    const worker = workerRef.current
    if (!file) {
      throw new Error('Load an image first')
    }
    if (!worker) {
      return
    }

    await ensureBestSettings()
    setPlyBlob(null)
    await ensureModelLoaded(false)

    const active = activeProfileRef.current
    if (!active) {
      return
    }

    const runOnce = async () => {
      setStatus('Generating splat…')
      const { tensor, width, height } = await imageFileToSharpTensor(file)
      const focal = await estimateFocalLengthFromFile(file, width, height)
      const disparityFactor = focal.focalPx / width

      const result = await worker.runInference({
        modelUrl: active.modelUrl,
        imageTensor: tensor.buffer,
        imageWidth: width,
        imageHeight: height,
        focalPx: focal.focalPx,
        disparityFactor,
        opacityThreshold,
        maxGaussians,
        allowWebGpuFp32: active.allowWebGpuFp32,
      })

      if (result.effectiveModelUrl && result.effectiveModelUrl !== active.modelUrl) {
        const switched: RuntimeProfile = {
          ...active,
          modelUrl: result.effectiveModelUrl,
          runtimeLabel:
            result.effectiveModelUrl === DEFAULT_WEB_MODEL_URL_FP32
              ? 'WebGPU + FP32'
              : active.runtimeLabel,
          summary: 'Worker switched to a different WebGPU model for this GPU.',
        }
        applyProfile(switched)
        setManualOverride(false)
      }

      const blob = new Blob([result.plyBuffer as ArrayBuffer], {
        type: 'application/octet-stream',
      })
      setPlyBlob(blob)
      recordRuntimeSuccess(activeProfileRef.current ?? active)
      setStatus('Done')
    }

    try {
      await runOnce()
    } catch (error) {
      const retried = await handleRuntimeError(error, async () => {
        await ensureModelLoaded(true)
        await runOnce()
      })
      if (!retried) {
        throw error
      }
    }
  }, [
    applyProfile,
    ensureBestSettings,
    ensureModelLoaded,
    handleRuntimeError,
    maxGaussians,
    opacityThreshold,
    setStatus,
  ])

  const loadModel = useCallback(async () => {
    setBusy(true)
    try {
      await ensureModelLoaded(true)
      setStatus(hasImage ? 'Image ready — click Run' : 'Model ready — load an image and run')
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }, [ensureModelLoaded, hasImage, setStatus])

  const selectImageFile = useCallback(
    async (file: File, name: string, exampleId: string | null, previewRemoteUrl: string | null) => {
      selectedFileRef.current = file
      setHasImage(true)
      setImageName(name)
      setSelectedExampleId(exampleId)
      setPlyBlob(null)
      setPreviewUrl((prev) => {
        if (prev?.startsWith('blob:')) {
          URL.revokeObjectURL(prev)
        }
        return previewRemoteUrl ?? URL.createObjectURL(file)
      })
      await readImageInfo(file)
    },
    [],
  )

  const onFileChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) return
      event.target.value = ''
      setBusy(true)
      try {
        await selectImageFile(file, file.name, null, null)
        await ensureModelLoaded(false)
        setStatus('Image ready — click Run')
      } catch (error) {
        setStatus(error instanceof Error ? error.message : String(error))
      } finally {
        setBusy(false)
      }
    },
    [ensureModelLoaded, selectImageFile, setStatus],
  )

  const loadExample = useCallback(
    async (example: ExampleImage) => {
      setBusy(true)
      try {
        setStatus('Loading example…')
        const file = await fetchExampleImageFile(example)
        await selectImageFile(file, example.label, example.id, example.url)
        await runPipeline()
      } catch (error) {
        setStatus(error instanceof Error ? error.message : String(error))
      } finally {
        setBusy(false)
      }
    },
    [runPipeline, selectImageFile, setStatus],
  )

  const generate = useCallback(async () => {
    if (!selectedFileRef.current) {
      setStatus('Load an image first')
      return
    }
    setBusy(true)
    try {
      await runPipeline()
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }, [runPipeline, setStatus])

  const downloadPly = useCallback(() => {
    if (!plyBlob) return
    const anchor = document.createElement('a')
    anchor.href = URL.createObjectURL(plyBlob)
    anchor.download = 'sharp_output.ply'
    anchor.click()
    URL.revokeObjectURL(anchor.href)
  }, [plyBlob])

  const canRun = !busy && hasImage

  return (
    <div className="app">
      <header className="site-header">
        <h1>SHARP Web</h1>
        <p className="site-tagline">Single photo → 3D Gaussian splat in your browser</p>
      </header>

      <section className="workflow" aria-label="Workflow">
        <div className="workflow-steps">
          <button
            type="button"
            className={`workflow-btn ${modelLoaded ? 'workflow-btn-done' : ''}`}
            onClick={loadModel}
            disabled={busy}
          >
            {modelLoaded ? '✓ Model loaded' : 'Load recommended model'}
          </button>

          <button
            type="button"
            className={`workflow-btn ${hasImage ? 'workflow-btn-done' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            disabled={busy}
          >
            {hasImage ? `✓ ${imageName || 'Image loaded'}` : 'Load image'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={onFileChange}
            disabled={busy}
            hidden
          />

          <button type="button" className="workflow-btn workflow-btn-primary" onClick={generate} disabled={!canRun}>
            Run
          </button>
        </div>

        <div className="example-images" aria-label="Example images">
          <span className="example-images-label">Examples</span>
          <div className="example-images-row">
            {EXAMPLE_IMAGES.map((example) => (
              <button
                key={example.id}
                type="button"
                className={`example-thumb ${selectedExampleId === example.id ? 'example-thumb-active' : ''}`}
                onClick={() => void loadExample(example)}
                disabled={busy}
                title={example.label}
              >
                <img src={example.url} alt={example.label} loading="lazy" />
                <span>{example.label}</span>
              </button>
            ))}
          </div>
        </div>

        <p className={`status-line ${status === 'Done' ? 'status-line-ok' : ''}`} role="status">
          {busy && <span className="status-spinner" aria-hidden="true" />}
          {status}
        </p>

        {recommended?.confidence === 'unsupported' && (
          <p className="status-warn">{recommended.summary}</p>
        )}
      </section>

      {previewUrl && !plyBlob && (
        <figure className="preview-thumb">
          <img src={previewUrl} alt="Input preview" />
        </figure>
      )}

      {plyBlob && (
        <section className="viewer-section" aria-label="Splat preview">
          <SplatViewer plyBlob={plyBlob} budgetMillion={maxGaussians / 1_000_000} />
          <div className="viewer-actions">
            <button type="button" className="btn-secondary" onClick={downloadPly} disabled={busy}>
              Download .ply
            </button>
            <a
              className="btn-secondary viewer-external-link"
              href={SPLATEDIT_VIEWER_URL}
              target="_blank"
              rel="noreferrer"
            >
              Open in SplatEdit viewer
            </a>
          </div>
          <p className="viewer-actions-hint">
            Download your splat, then upload it to{' '}
            <a href={SPLATEDIT_VIEWER_URL} target="_blank" rel="noreferrer">
              splatedit.app/viewer
            </a>
            .
          </p>
        </section>
      )}

      <details className="advanced">
        <summary>Advanced</summary>
        <div className="advanced-body">
          <p className="advanced-hint">
            Recommended: {runtimeLabel}
            {recommended?.adapterLabel ? ` · ${recommended.adapterLabel}` : ''}
          </p>

          <label>
            Max Gaussians
            <input
              type="number"
              min={1000}
              max={5000000}
              step={1000}
              value={maxGaussians}
              onChange={(e) => setMaxGaussians(Number(e.target.value))}
            />
          </label>
          <label>
            Opacity threshold
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={opacityThreshold}
              onChange={(e) => setOpacityThreshold(Number(e.target.value))}
            />
          </label>

          <label>
            Model URL
            <input
              type="text"
              value={modelUrl}
              onChange={(e) => {
                setModelUrl(e.target.value)
                setManualOverride(true)
                setModelLoaded(false)
              }}
              spellCheck={false}
            />
          </label>

          <div className="button-row">
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                clearRuntimeHistory()
                setManualOverride(false)
                setModelLoaded(false)
                workerRef.current?.dispose()
                workerRef.current = new SharpWorkerClient(onStatus)
                workerRef.current.setRuntimeHintHandler(onRuntimeHint)
                void refreshRecommendation(true).then(() => setStatus('Ready'))
              }}
            >
              Reset runtime
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                applyProfile(
                  {
                    modelUrl: DEFAULT_WEB_MODEL_URL_FP32,
                    runtimeLabel: 'WebGPU + FP32',
                    summary: 'Manual override: FP32 on WebGPU.',
                    confidence: 'medium',
                    webGpuAvailable: true,
                    webGpuFloat16: false,
                    riskyWebGpuFp32: false,
                    adapterLabel: '',
                    allowWebGpuFp32: true,
                  },
                  true,
                )
              }}
            >
              FP32 / WebGPU
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                applyProfile(
                  {
                    modelUrl: DEFAULT_WEB_MODEL_URL_FP16,
                    runtimeLabel: 'WebGPU + FP16',
                    summary: 'Manual override: FP16 on WebGPU.',
                    confidence: 'medium',
                    webGpuAvailable: true,
                    webGpuFloat16: true,
                    riskyWebGpuFp32: false,
                    adapterLabel: '',
                    allowWebGpuFp32: true,
                  },
                  true,
                )
              }}
            >
              FP16 / WebGPU
            </button>
          </div>
        </div>
      </details>

      <footer className="license-footer">
        <p className="license-footer-attribution">{APPLE_MODEL_ATTRIBUTION_NOTICE}</p>
        <p>
          Apple SHARP ONNX weights: research use only ·{' '}
          <a href={SHARP_MODEL_LICENSE_URL} target="_blank" rel="noreferrer">
            Model license
          </a>
          {' · '}
          <a href={SHARP_MODEL_DERIVATIVE_NOTICE_URL} target="_blank" rel="noreferrer">
            Derivative notice
          </a>
        </p>
        <p>
          Example photos from{' '}
          <a href={WORDPRESS_PHOTOS_URL} target="_blank" rel="noreferrer">
            WordPress Photos
          </a>{' '}
          (public domain)
        </p>
      </footer>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
