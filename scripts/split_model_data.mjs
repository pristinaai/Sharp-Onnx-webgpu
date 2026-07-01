#!/usr/bin/env node
/**
 * Split a large .onnx.data sidecar into GitHub-LFS-safe parts (< 2 GiB each).
 * Usage: node scripts/split_model_data.mjs [path/to/model.onnx.data]
 */
import { createReadStream, createWriteStream, promises as fs } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { pipeline } from 'node:stream/promises'

const CHUNK_BYTES = 1_500_000_000 // 1.5 GiB — under GitHub LFS 2 GiB file cap

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const defaultInput = path.join(repoRoot, 'web/public/models/sharp_web_predictor.onnx.data')
const inputPath = path.resolve(process.argv[2] ?? defaultInput)
const baseName = path.basename(inputPath)
const outDir = path.dirname(inputPath)

async function main() {
  const stat = await fs.stat(inputPath)
  const partNames = []
  let offset = 0
  let partIndex = 0

  while (offset < stat.size) {
    const chunkSize = Math.min(CHUNK_BYTES, stat.size - offset)
    const partName = `${baseName}.part${String(partIndex).padStart(2, '0')}`
    const partPath = path.join(outDir, partName)
    partNames.push(partName)

    await pipeline(
      createReadStream(inputPath, { start: offset, end: offset + chunkSize - 1 }),
      createWriteStream(partPath),
    )

    console.log(`Wrote ${partName} (${chunkSize} bytes)`)
    offset += chunkSize
    partIndex += 1
  }

  const manifest = {
    file: baseName,
    size: stat.size,
    chunkBytes: CHUNK_BYTES,
    parts: partNames,
  }
  const manifestPath = path.join(outDir, `${baseName}.manifest.json`)
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8')
  console.log(`Wrote ${path.basename(manifestPath)} (${partNames.length} parts, ${stat.size} bytes total)`)
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
