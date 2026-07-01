#!/usr/bin/env node
/**
 * Vercel Hobby/Pro cannot deploy multi-GB static assets from public/models/.
 * Strip weight blobs from dist so the UI deploys; host models elsewhere and set Model URL.
 */
import { readdir, stat, unlink } from 'node:fs/promises'
import path from 'node:path'

const modelsDir = path.join(process.cwd(), 'dist/models')

async function main() {
  if (!process.env.VERCEL) {
    return
  }

  let removed = 0
  let bytes = 0
  for (const name of await readdir(modelsDir)) {
    if (!/\.(data|part\d+)$/.test(name)) {
      continue
    }
    const filePath = path.join(modelsDir, name)
    const info = await stat(filePath)
    await unlink(filePath)
    removed += 1
    bytes += info.size
  }

  if (removed > 0) {
    const mb = (bytes / (1024 * 1024)).toFixed(0)
    console.warn(
      `prune_vercel_dist: removed ${removed} model weight file(s) (${mb} MB) — too large for Vercel static hosting`,
    )
  }
}

main().catch((error) => {
  console.warn('prune_vercel_dist:', error instanceof Error ? error.message : error)
})
