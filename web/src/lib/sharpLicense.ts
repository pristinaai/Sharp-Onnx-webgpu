/** Apple SHARP model license disclosure (see apple/ml-sharp LICENSE_MODEL). */

/** Verbatim copy bundled under public/licenses/ (also in repo at web/licenses/). */
export const SHARP_MODEL_LICENSE_URL = `${import.meta.env.BASE_URL}licenses/APPLE_SHARP_LICENSE_MODEL`

export const SHARP_MODEL_DERIVATIVE_NOTICE_URL = `${import.meta.env.BASE_URL}licenses/MODEL_DERIVATIVE_NOTICE.md`

export const WORDPRESS_PHOTOS_URL = 'https://wordpress.org/photos'

/** Required attribution when redistributing Apple’s model or derivatives (LICENSE_MODEL). */
export const APPLE_MODEL_ATTRIBUTION_NOTICE =
  'Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.'

export const SHARP_MODEL_LICENSE_SHORT =
  'The Apple SHARP model weights (sharp_2572gikvuh.pt and ONNX exports derived from them) are released under the Apple Machine Learning Research Model License Agreement for research purposes only.'

export const SHARP_MODEL_LICENSE_POINTS = [
  'Research use only — non-commercial scientific research and academic development.',
  'Not licensed for commercial exploitation, product development, or use in commercial products or services.',
  'By loading or running the model in this app, you agree to Apple’s LICENSE_MODEL terms.',
] as const
