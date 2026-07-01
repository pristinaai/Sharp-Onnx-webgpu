/** PlayCanvas SuperSplat settings.json camera pose (position + look-at target). */
export interface ViewerCameraPose {
  position: [number, number, number]
  target: [number, number, number]
  fov: number
}

/**
 * Initial camera for the splat viewer iframe.
 * Set to `null` to auto-frame on load. Tune values using the debug readout under the viewer,
 * then paste into this preset.
 */
export const SPLAT_VIEWER_CAMERA_PRESET: ViewerCameraPose | null = {
  position: [-0.050122334094014995, 0.135977455447434, -0.30668114999315266],
  target: [-0.016, 0.065, 0.705],
  fov: 75,
}

export interface ViewerCameraDebugState {
  position: [number, number, number]
  target: [number, number, number]
  fov: number
  mode: string
  angles: [number, number, number]
  distance: number
}

export function formatCameraPoseJson(pose: ViewerCameraPose): string {
  return JSON.stringify(pose, null, 2)
}

export function buildViewerSettings(preset: ViewerCameraPose | null): object {
  return {
    version: 2,
    tonemapping: 'aces',
    highPrecisionRendering: false,
    background: {
      color: [0.06, 0.07, 0.09],
    },
    postEffectSettings: {
      sharpness: { enabled: false, amount: 0 },
      bloom: { enabled: false, intensity: 1, blurLevel: 2 },
      grading: {
        enabled: false,
        brightness: 0,
        contrast: 1,
        saturation: 1,
        tint: [1, 1, 1],
      },
      vignette: {
        enabled: false,
        intensity: 0.5,
        inner: 0.3,
        outer: 0.75,
        curvature: 1,
      },
      fringing: { enabled: false, intensity: 0.5 },
    },
    animTracks: [],
    cameras: preset ? [{ initial: preset }] : [],
    annotations: [],
    startMode: 'default',
  }
}
