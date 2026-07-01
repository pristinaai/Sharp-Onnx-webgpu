import { useEffect, useRef, useState } from 'react'

import {
  SPLAT_VIEWER_CAMERA_PRESET,
  buildViewerSettings,
  formatCameraPoseJson,
  type ViewerCameraDebugState,
  type ViewerCameraPose,
} from '../lib/viewerCamera'

interface SplatViewerProps {
  plyBlob: Blob
  /** Splat budget in millions (PlayCanvas viewer `budget` URL param). */
  budgetMillion?: number
  /** Override the default preset from viewerCamera.ts */
  cameraPreset?: ViewerCameraPose | null
}

type ViewerIframeWindow = Window & {
  firstFrame?: () => void
  getCameraState?: () => {
    position: [number, number, number]
    angles: [number, number, number]
    distance: number
    fov: number
    mode: string
  }
}

function parseTriple(text: string | null | undefined): [number, number, number] | null {
  if (!text) return null
  const trimmed = text.trim()
  if (!trimmed || trimmed === '—') return null
  const parts = trimmed.split(/[\s,]+/).filter(Boolean).map(Number)
  if (parts.length !== 3 || parts.some((n) => !Number.isFinite(n))) return null
  return [parts[0], parts[1], parts[2]]
}

function readCameraDebug(iframeWindow: ViewerIframeWindow): ViewerCameraDebugState | null {
  const state = iframeWindow.getCameraState?.()
  if (!state) return null

  const focus = parseTriple(
    iframeWindow.document.querySelector('#sse-debug-panel [data-id="focus"]')?.textContent,
  )

  return {
    position: state.position,
    target: focus ?? state.position,
    fov: state.fov,
    mode: state.mode,
    angles: state.angles,
    distance: state.distance,
  }
}

function hideIframeDebugPanel(doc: Document): void {
  if (doc.getElementById('sse-hide-debug-panel')) return
  const style = doc.createElement('style')
  style.id = 'sse-hide-debug-panel'
  style.textContent = '#sse-debug-panel { display: none !important; }'
  doc.head.appendChild(style)
}

export function SplatViewer({
  plyBlob,
  budgetMillion,
  cameraPreset = SPLAT_VIEWER_CAMERA_PRESET,
}: SplatViewerProps) {
  const [viewerSrc, setViewerSrc] = useState<string | null>(null)
  const [cameraDebug, setCameraDebug] = useState<ViewerCameraDebugState | null>(null)
  const plyUrlRef = useRef<string | null>(null)
  const settingsUrlRef = useRef<string | null>(null)
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const pollRef = useRef<number | null>(null)

  useEffect(() => {
    const previousPly = plyUrlRef.current
    const previousSettings = settingsUrlRef.current

    const file = new File([plyBlob], 'sharp_splat.ply', { type: 'application/octet-stream' })
    const plyUrl = URL.createObjectURL(file)
    plyUrlRef.current = plyUrl

    const settingsBlob = new Blob([JSON.stringify(buildViewerSettings(cameraPreset))], {
      type: 'application/json',
    })
    const settingsUrl = URL.createObjectURL(settingsBlob)
    settingsUrlRef.current = settingsUrl

    const params = new URLSearchParams({
      content: plyUrl,
      settings: settingsUrl,
      fullload: '',
      webgl: '',
      debug: '',
      noanim: '',
    })
    if (budgetMillion !== undefined && Number.isFinite(budgetMillion) && budgetMillion > 0) {
      params.set('budget', String(Math.max(budgetMillion, 0.05)))
    }
    setViewerSrc(`${import.meta.env.BASE_URL}supersplat-viewer/index.html?${params}`)
    setCameraDebug(null)

    if (previousPly && previousPly !== plyUrl) URL.revokeObjectURL(previousPly)
    if (previousSettings && previousSettings !== settingsUrl) URL.revokeObjectURL(previousSettings)
  }, [plyBlob, budgetMillion, cameraPreset])

  useEffect(() => {
    return () => {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current)
      }
      if (plyUrlRef.current) {
        URL.revokeObjectURL(plyUrlRef.current)
        plyUrlRef.current = null
      }
      if (settingsUrlRef.current) {
        URL.revokeObjectURL(settingsUrlRef.current)
        settingsUrlRef.current = null
      }
    }
  }, [])

  const startCameraPoll = (iframeWindow: ViewerIframeWindow) => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current)
    }
    const tick = () => {
      const next = readCameraDebug(iframeWindow)
      if (next) {
        setCameraDebug(next)
      }
    }
    tick()
    pollRef.current = window.setInterval(tick, 250)
  }

  const handleIframeLoad = () => {
    const iframeWindow = iframeRef.current?.contentWindow as ViewerIframeWindow | null
    if (!iframeWindow) {
      return
    }

    hideIframeDebugPanel(iframeWindow.document)

    const frameScene = () => {
      if (cameraPreset) return
      iframeWindow.document.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'f', code: 'KeyF', bubbles: true }),
      )
    }

    const switchToOrbitMode = () => {
      iframeWindow.dispatchEvent(
        new KeyboardEvent('keydown', { key: '1', code: 'Digit1', bubbles: true }),
      )
    }

    const priorFirstFrame = iframeWindow.firstFrame
    iframeWindow.firstFrame = () => {
      priorFirstFrame?.()
      frameScene()
      switchToOrbitMode()
      startCameraPoll(iframeWindow)
    }

    iframeWindow.setTimeout(() => {
      frameScene()
      switchToOrbitMode()
      startCameraPoll(iframeWindow)
    }, 2000)
  }

  const copyPreset = async () => {
    if (!cameraDebug) return
    const pose: ViewerCameraPose = {
      position: cameraDebug.position,
      target: cameraDebug.target,
      fov: cameraDebug.fov,
    }
    await navigator.clipboard.writeText(formatCameraPoseJson(pose))
  }

  if (!viewerSrc) {
    return null
  }

  const presetJson = cameraDebug
    ? formatCameraPoseJson({
        position: cameraDebug.position,
        target: cameraDebug.target,
        fov: cameraDebug.fov,
      })
    : null

  return (
    <div className="splat-viewer-wrap">
      <iframe
        ref={iframeRef}
        key={viewerSrc}
        className="splat-viewer"
        src={viewerSrc}
        title="Gaussian splat preview (PlayCanvas)"
        allow="fullscreen"
        onLoad={handleIframeLoad}
      />
      <p className="hint splat-viewer-hint">
        Drag to orbit · scroll to zoom · double-click to focus · press F to re-frame
      </p>

      <div className="camera-debug" aria-live="polite">
        <div className="camera-debug-header">
          <span className="camera-debug-title">Camera debug</span>
          {cameraPreset && <span className="camera-debug-badge">preset active</span>}
          <button
            type="button"
            className="btn-secondary camera-debug-copy"
            onClick={() => void copyPreset()}
            disabled={!cameraDebug}
          >
            Copy preset JSON
          </button>
        </div>
        {cameraDebug ? (
          <>
            <dl className="camera-debug-grid">
              <div>
                <dt>position</dt>
                <dd>
                  {cameraDebug.position.map((v) => v.toFixed(4)).join(', ')}
                </dd>
              </div>
              <div>
                <dt>target</dt>
                <dd>{cameraDebug.target.map((v) => v.toFixed(4)).join(', ')}</dd>
              </div>
              <div>
                <dt>fov</dt>
                <dd>{cameraDebug.fov.toFixed(2)}</dd>
              </div>
              <div>
                <dt>mode</dt>
                <dd>{cameraDebug.mode}</dd>
              </div>
              <div>
                <dt>angles</dt>
                <dd>{cameraDebug.angles.map((v) => v.toFixed(2)).join(', ')}</dd>
              </div>
              <div>
                <dt>distance</dt>
                <dd>{cameraDebug.distance.toFixed(4)}</dd>
              </div>
            </dl>
            {presetJson && (
              <pre className="camera-debug-json">{presetJson}</pre>
            )}
            <p className="camera-debug-hint">
              Paste into <code>SPLAT_VIEWER_CAMERA_PRESET</code> in{' '}
              <code>src/lib/viewerCamera.ts</code> to lock this view on load.
            </p>
          </>
        ) : (
          <p className="camera-debug-hint">Waiting for viewer…</p>
        )}
      </div>
    </div>
  )
}
