export interface ExampleImage {
  id: string
  label: string
  /** Same-origin path served from public/examples (source: WordPress.org public domain). */
  url: string
  sourceUrl: string
}

/** Public-domain sample photos bundled under /examples/ (from WordPress.org media library). */
export const EXAMPLE_IMAGES: ExampleImage[] = [
  {
    id: 'example-1',
    label: 'Example 1',
    url: `${import.meta.env.BASE_URL}examples/example-1.jpg`,
    sourceUrl: 'https://pd.w.org/2026/06/7256a437e3102c983.69251117.jpg',
  },
  {
    id: 'example-2',
    label: 'Example 2',
    url: `${import.meta.env.BASE_URL}examples/example-2.jpg`,
    sourceUrl: 'https://pd.w.org/2026/06/1376a38d639bf0033.09499563.jpeg',
  },
  {
    id: 'example-3',
    label: 'Example 3',
    url: `${import.meta.env.BASE_URL}examples/example-3.jpg`,
    sourceUrl: 'https://pd.w.org/2026/06/5356a3e9160c6aee9.58126174-2048x1365.jpg',
  },
]

export async function fetchExampleImageFile(example: ExampleImage): Promise<File> {
  const response = await fetch(example.url)
  if (!response.ok) {
    throw new Error(`Could not load ${example.label} (${response.status})`)
  }
  const blob = await response.blob()
  const extension = example.url.split('.').pop()?.split('?')[0] ?? 'jpg'
  return new File([blob], `${example.id}.${extension}`, {
    type: blob.type || 'image/jpeg',
  })
}
